"""Weight-removal unlearning: Fisher-ratio pruning / dampening of the *base* weights.

For every parameter theta_i in the transformer blocks we estimate diagonal Fishers on the forget
and retain sets (squared gradients with sampled targets):
    F_f[i] = E_forget[ g_i^2 ],   F_r[i] = E_retain[ g_i^2 ].
Weights that matter to the forget set but not to the retain set are removed or dampened
(Selective Synaptic Dampening, Foster et al. 2024, generalised with a hard-prune option):

    ratio_i = F_f[i] / (F_r[i] + eps)
    if ratio_i > alpha:   theta_i <- 0                                (mode="prune")
                           theta_i <- theta_i * min(1, lam * F_r[i]/F_f[i])   (mode="dampen")

This is a one-shot, training-free method that physically deletes weights, in contrast to the
gradient-based methods in trainer.py. It can be followed by a short CG-OSS / retain-only phase.
"""
from __future__ import annotations

import time

import torch

from .config import Config
from .dataset import IGNORE_INDEX, DataBundle
from .losses import retain_loss


_targets: tuple[str, ...] | None = None


def _block_params(model) -> dict[str, torch.nn.Parameter]:
    """Trainable 2-D weights inside transformer blocks (skip embeddings, norms, lm_head, LoRA)."""
    out = {}
    for n, p in model.named_parameters():
        if p.ndim == 2 and ".layers." in n and "lora_" not in n and "norm" not in n:
            if _targets is None or any(t in n for t in _targets):
                out[n] = p
    return out


@torch.no_grad()
def _sample_labels(logits, labels):
    probs = torch.softmax(logits[:, :-1, :].float(), dim=-1)
    sampled = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(probs.shape[:2])
    labels = labels.clone()
    keep = labels[:, 1:] != IGNORE_INDEX
    labels[:, 1:] = torch.where(keep, sampled, labels[:, 1:])
    return labels


def diag_fisher(model, ds, data: DataBundle, cfg: Config, device, n_samples: int, mode: str = "true") -> dict[str, torch.Tensor]:
    """Diagonal Fisher per block parameter, accumulated in fp32 on CPU to bound device memory."""
    params = _block_params(model)
    for p in params.values():
        p.requires_grad_(True)
    # bf16 accumulators on CPU: a full fp32 diagonal Fisher for a 1.5B model is ~5 GB per dataset.
    acc_dtype = getattr(torch, cfg.prune.acc_dtype)
    acc = {n: torch.zeros(p.shape, dtype=acc_dtype, device="cpu") for n, p in params.items()}
    model.eval()
    seen = 0
    for batch in data.loader(ds, cfg.train.batch_size, shuffle=True, seed=cfg.data.seed):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        labels = _sample_labels(out.logits, batch["labels"]) if mode == "true" else batch["labels"]
        retain_loss(out.logits, labels).backward()
        bs = batch["input_ids"].size(0)
        for n, p in params.items():
            if p.grad is not None:
                acc[n] += (p.grad.detach().float().pow(2) * bs).to(acc_dtype).cpu()
        seen += bs
        if seen >= n_samples:
            break
    model.zero_grad(set_to_none=True)
    for p in params.values():
        p.requires_grad_(False)
    for a in acc.values():
        a.div_(max(seen, 1))
    return acc


def _layer_of(name: str) -> int:
    return int(name.split(".layers.")[1].split(".")[0])


def fisher_ratio(model, data: DataBundle, cfg: Config, device, n_samples: int, group_layers: int = 4) -> dict[str, torch.Tensor]:
    """Memory-lean F_forget / F_retain ratio for every prunable weight.

    Layers are processed in groups: only one group's weights get gradients and fp32 accumulators
    at a time, so peak extra memory is ~group/total of a full Fisher. Both domains use the same
    seeded batches in every group. Returns bf16 ratios on CPU.
    """
    params = _block_params(model)
    layers = sorted({_layer_of(n) for n in params})
    groups = [layers[i:i + group_layers] for i in range(0, len(layers), group_layers)]
    eps_rel = cfg.prune.eps
    ratio: dict[str, torch.Tensor] = {}
    t0 = time.time()
    for gi, grp in enumerate(groups):
        names = [n for n in params if _layer_of(n) in grp]
        fis = {}
        for which, ds in (("retain", data.retain_train), ("forget", data.forget_train)):
            sub = {n: params[n] for n in names}
            fis[which] = _diag_fisher_subset(model, sub, ds, data, cfg, device, n_samples)
        for n in names:
            fr, ff = fis["retain"][n].float(), fis["forget"][n].float()
            ratio[n] = (ff / (fr + eps_rel * float(fr.mean()) + 1e-20)).to(torch.bfloat16)
        del fis
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"[prune] fisher group {gi + 1}/{len(groups)} layers {grp[0]}-{grp[-1]} done ({time.time() - t0:.0f}s)", flush=True)
    return ratio


def _diag_fisher_subset(model, sub, ds, data, cfg, device, n_samples) -> dict[str, torch.Tensor]:
    for p in sub.values():
        p.requires_grad_(True)
    acc_dtype = getattr(torch, cfg.prune.acc_dtype)
    acc = {n: torch.zeros(p.shape, dtype=acc_dtype) for n, p in sub.items()}
    model.eval()
    ckpt = hasattr(model, "gradient_checkpointing_enable") and not cfg.mock
    if ckpt:
        # Recompute activations in backward: large saving on a 16 GB machine.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()  # HF only checkpoints in train mode; Qwen has no dropout at inference config
    seen = 0
    for batch in data.loader(ds, cfg.train.batch_size, shuffle=True, seed=cfg.data.seed):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        labels = _sample_labels(out.logits, batch["labels"]) if cfg.fisher.mode == "true" else batch["labels"]
        retain_loss(out.logits, labels).backward()
        bs = batch["input_ids"].size(0)
        for n, p in sub.items():
            if p.grad is not None:
                acc[n] += (p.grad.detach().float().pow(2) * bs).to(acc_dtype).cpu()
                p.grad = None
        del out, labels, batch
        if device.type == "mps":
            torch.mps.empty_cache()
        seen += bs
        if seen >= n_samples:
            break
    for p in sub.values():
        p.requires_grad_(False)
    if ckpt:
        model.gradient_checkpointing_disable()
        model.eval()
    for a in acc.values():
        a.div_(max(seen, 1))
    return acc


@torch.no_grad()
def remove_by_ratio(model, ratio: dict[str, torch.Tensor], fraction: float | None, alpha: float | None,
                    mode: str = "prune", lam: float = 1.0, undo: bool = False):
    """Zero (or dampen) weights with the highest forget/retain ratio.

    fraction : per-matrix fraction to remove (top-k by ratio); combined with alpha if both given.
    Returns (stats, undo_log). undo_log maps name -> (flat indices, original values) if undo=True.
    """
    params = _block_params(model)
    total = removed = 0
    log = {}
    for n, p in params.items():
        r = ratio[n].float().flatten()
        mask = torch.ones_like(r, dtype=torch.bool) if alpha is None else (r > alpha)
        if fraction is not None:
            k = max(1, int(fraction * r.numel()))
            top = torch.zeros_like(mask)
            top[torch.topk(r, k).indices] = True
            mask &= top
        idx = mask.nonzero().squeeze(1)
        flat = p.view(-1)
        dev_idx = idx.to(p.device)
        if undo:
            log[n] = (idx, flat[dev_idx].detach().cpu().clone())
        if mode == "prune":
            flat[dev_idx] = 0
        else:
            scale = torch.clamp(lam / r[idx], max=1.0).to(p.dtype).to(p.device)
            flat[dev_idx] *= scale
        removed += idx.numel()
        total += p.numel()
    return {"pruned_frac": removed / max(total, 1)}, log


@torch.no_grad()
def undo_removal(model, log):
    params = _block_params(model)
    for n, (idx, vals) in log.items():
        params[n].view(-1)[idx.to(params[n].device)] = vals.to(params[n].device)


@torch.no_grad()
def apply_pruning(model, F_f: dict, F_r: dict, cfg: Config) -> dict[str, float]:
    """Zero (or dampen) weights with high forget/retain Fisher ratio. Returns removal statistics."""
    pc = cfg.prune
    params = _block_params(model)
    total = removed = 0
    per_layer = {}
    for n, p in params.items():
        ff, fr = F_f[n].float(), F_r[n].float()
        eps = pc.eps * float(fr.mean()) + 1e-20
        ratio = ff / (fr + eps)
        if pc.mode == "prune":
            mask = ratio > pc.alpha
            if pc.max_fraction is not None and float(mask.float().mean()) > pc.max_fraction:
                k = int(pc.max_fraction * ratio.numel())
                thresh = torch.topk(ratio.flatten(), k).values.min()
                mask = ratio >= thresh
            p.mul_((~mask).to(p.dtype).to(p.device))
            r = int(mask.sum())
        else:  # dampen
            scale = torch.clamp(pc.lam * fr / (ff + 1e-20), max=1.0)
            scale = torch.where(ratio > pc.alpha, scale, torch.ones_like(scale))
            p.mul_(scale.to(p.dtype).to(p.device))
            r = int((scale < 1).sum())
        removed += r
        total += p.numel()
        per_layer[n] = r / p.numel()
    worst = sorted(per_layer.items(), key=lambda kv: -kv[1])[:5]
    print(f"[prune] mode={pc.mode} alpha={pc.alpha} removed {removed}/{total} = {removed/total:.4%}; top layers: "
          + ", ".join(f"{k.split('.layers.')[1]}={v:.2%}" for k, v in worst))
    return {"pruned_frac": removed / max(total, 1)}


def prune_model(model, data: DataBundle, cfg: Config, device) -> dict[str, float]:
    global _targets
    _targets = tuple(cfg.prune.targets) if cfg.prune.targets else None
    t0 = time.time()
    from .config import CHECKPOINT_DIR
    tag = "mock" if cfg.mock else cfg.model.name_or_path.split("/")[-1].lower()
    cache = CHECKPOINT_DIR / f"fisher_ratio_{cfg.data.task}_{tag}_n{cfg.prune.n_samples}_seed{cfg.data.seed}.pt"
    if cache.exists() and not cfg.mock:
        ratio = torch.load(cache)
        print(f"[prune] loaded cached Fisher ratio {cache.name}", flush=True)
    else:
        ratio = fisher_ratio(model, data, cfg, device, cfg.prune.n_samples, cfg.prune.group_layers)
        if not cfg.mock:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ratio, cache)
    alpha = cfg.prune.alpha if (cfg.prune.alpha is not None and cfg.prune.alpha > 0) else None
    stats, _ = remove_by_ratio(model, ratio, cfg.prune.max_fraction, alpha, cfg.prune.mode, cfg.prune.lam)
    del ratio
    if device.type == "mps":
        torch.mps.empty_cache()
    stats["prune_seconds"] = round(time.time() - t0, 1)
    print(f"[prune] removed {stats['pruned_frac']:.4%} of prunable weights in {stats['prune_seconds']}s", flush=True)
    return stats
