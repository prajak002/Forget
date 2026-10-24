"""Curvature-guarded protected subspaces from a Kronecker-factored empirical Fisher.

For a linear layer y = W x the empirical Fisher block is well approximated (K-FAC) as
    F_W  ≈  E[δ δᵀ] ⊗ E[x xᵀ],      δ = ∂L/∂y.
With LoRA, W = W0 + B A, so
    ∂L/∂A = (Bᵀ δ) xᵀ   -> rows live in span{x}   (input factor),
    ∂L/∂B = δ (A x)ᵀ    -> columns live in span{δ} (output factor).

We collect retain-set activations x and back-propagated δ for every LoRA-wrapped
module, take the top-k singular directions of each (k chosen by Fisher-trace energy),
and expose projectors P = I - s·UUᵀ that remove the forget gradient's component along
the directions the retain task's curvature is concentrated in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import Config
from .dataset import IGNORE_INDEX, DataBundle
from .losses import retain_loss
from .models import lora_pairs


@dataclass
class Subspace:
    U: torch.Tensor           # (d, k) orthonormal columns
    energy: torch.Tensor      # (k,) singular values^2 (Fisher-trace contributions)
    total_energy: float
    rank: int

    @property
    def captured(self) -> float:
        return float(self.energy.sum() / max(self.total_energy, 1e-12))


@dataclass
class ProtectedSubspaces:
    inputs: dict[str, Subspace] = field(default_factory=dict)    # per module: x-space
    outputs: dict[str, Subspace] = field(default_factory=dict)   # per module: δ-space

    def summary(self) -> dict[str, float]:
        ranks = [s.rank for s in self.inputs.values()]
        caps = [s.captured for s in self.inputs.values()]
        return {
            "n_modules": len(self.inputs),
            "mean_rank": sum(ranks) / max(len(ranks), 1),
            "mean_captured": sum(caps) / max(len(caps), 1),
        }

    def save(self, path):
        torch.save(
            {k: {"U": v.U.cpu(), "energy": v.energy.cpu(), "total": v.total_energy} for k, v in self.inputs.items()}
            | {f"__out__{k}": {"U": v.U.cpu(), "energy": v.energy.cpu(), "total": v.total_energy} for k, v in self.outputs.items()},
            path,
        )


class _Reservoir:
    """Keep up to `cap` rows uniformly at random from a stream of (n, d) matrices."""

    def __init__(self, cap: int, seed: int = 0):
        self.cap, self.rows, self.seen = cap, None, 0
        self.g = torch.Generator().manual_seed(seed)

    def add(self, X: torch.Tensor):
        X = X.detach().reshape(-1, X.size(-1)).float().cpu()
        if self.rows is None:
            self.rows = X.new_empty((0, X.size(-1)))
        # Stage 1: fill the buffer.
        free = self.cap - self.rows.size(0)
        if free > 0:
            take = X[:free]
            self.rows = torch.cat([self.rows, take])
            self.seen += take.size(0)
            X = X[free:]
        # Stage 2: classic reservoir replacement for the remaining rows.
        for i in range(X.size(0)):
            self.seen += 1
            j = int(torch.randint(0, self.seen, (1,), generator=self.g))
            if j < self.cap:
                self.rows[j] = X[i]


def _subspace_from_rows(R: torch.Tensor, rank: int, energy_fraction: float | None, damping: float) -> Subspace:
    # Uncentred covariance C = Rᵀ R / n; its eigenvectors are the right singular vectors of R.
    n = R.size(0)
    _, S, Vh = torch.linalg.svd(R, full_matrices=False)
    energy = S.pow(2) / n + damping
    total = float(energy.sum())
    if energy_fraction is not None:
        cum = torch.cumsum(energy, 0) / total
        k = int((cum < energy_fraction).sum().item()) + 1
    else:
        k = rank
    k = max(1, min(k, Vh.size(0), rank if energy_fraction is None else Vh.size(0)))
    return Subspace(U=Vh[:k].T.contiguous(), energy=energy[:k], total_energy=total, rank=k)


@torch.no_grad()
def _finalize(res: dict[str, _Reservoir], cfg: Config) -> dict[str, Subspace]:
    out = {}
    for name, r in res.items():
        if r.rows is None:
            continue
        out[name] = _subspace_from_rows(r.rows, cfg.fisher.rank, cfg.fisher.energy_fraction, cfg.fisher.damping)
    return out


def _collect(model, ds, cfg: Config, device: torch.device, rows_cap: int, seed: int, pad_id: int = 0):
    """Forward+backward on `ds`; return (input reservoirs, output-grad reservoirs) per LoRA module."""
    pairs = lora_pairs(model)
    in_res = {n: _Reservoir(rows_cap, seed) for n in pairs}
    out_res = {n: _Reservoir(rows_cap, seed + 1) for n in pairs}
    handles = []
    for name, (wrapped, _, _) in pairs.items():
        def fwd_hook(mod, inp, outp, _n=name):
            in_res[_n].add(inp[0])
            if outp.requires_grad:
                outp.register_hook(lambda g, _n=_n: out_res[_n].add(g))
        handles.append(wrapped.register_forward_hook(fwd_hook))
    was_training = model.training
    model.train()
    seen = 0
    for batch in _loader(ds, cfg, seed, pad_id):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        labels = batch["labels"]
        if cfg.fisher.mode == "true":
            with torch.no_grad():
                probs = torch.softmax(out.logits[:, :-1, :].float(), dim=-1)
                sampled = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(probs.shape[:2])
                labels = labels.clone()
                keep = labels[:, 1:] != IGNORE_INDEX
                labels[:, 1:] = torch.where(keep, sampled, labels[:, 1:])
        retain_loss(out.logits, labels).backward()
        seen += batch["input_ids"].size(0)
        if seen >= cfg.fisher.n_samples:
            break
    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return in_res, out_res


def _loader(ds, cfg: Config, seed: int, pad_id: int = 0):
    from torch.utils.data import DataLoader
    from .dataset import collate
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, generator=g, collate_fn=lambda b: collate(b, pad_id))


def _discriminative_subspace(Rr: torch.Tensor, Rf: torch.Tensor, cfg: Config) -> Subspace:
    """Top generalized eigenvectors of (C_r, C_f + eps I): directions where retain curvature
    exceeds forget curvature by factor > ratio_threshold. Returned basis is orthonormalised.

    Numerically robust for large d (e.g. 8960 for Qwen MLP) and rank-deficient samples:
      1. Restrict to the joint row span B = orth([Rr; Rf]^T)  (dim m <= n_r + n_f), since any
         direction outside it carries zero energy for both domains and need not be protected.
      2. Work in float64 and whiten C_f by eigen-decomposition with clamped eigenvalues
         (instead of Cholesky, which fails on singular matrices).
    """
    Rr = Rr.double()
    Rf = Rf.double()
    d = Rr.size(1)
    joint = torch.cat([Rr, Rf], 0)
    if joint.size(0) < d:
        B, _ = torch.linalg.qr(joint.T)          # (d, m) orthonormal basis of the joint span
    else:
        B = torch.eye(d, dtype=torch.float64)
    Ar, Af = Rr @ B, Rf @ B                      # coordinates in the span
    m = B.size(1)
    Cr = Ar.T @ Ar / Ar.size(0)
    Cf = Af.T @ Af / Af.size(0)
    eps = max(cfg.fisher.damping, 1e-3) * float(Cf.trace()) / m + 1e-12
    lf, Qf = torch.linalg.eigh(0.5 * (Cf + Cf.T))
    W_half_inv = Qf / torch.sqrt(lf.clamp_min(0) + eps)   # (m, m): Cf_eps^{-1/2} = Qf diag(1/sqrt) Qf^T (right factor)
    M = W_half_inv.T @ Cr @ W_half_inv
    M = 0.5 * (M + M.T)
    lam, W = torch.linalg.eigh(M)
    lam, W = lam.flip(0), W.flip(1)
    k = int((lam > cfg.fisher.ratio_threshold).sum().item())
    if cfg.fisher.energy_fraction is None:
        k = min(k, cfg.fisher.rank)
    k = max(1, min(k, m - 1))
    V = B @ (W_half_inv @ W[:, :k])              # back to R^d
    Q, _ = torch.linalg.qr(V)
    Cr_full_energy = float(Rr.pow(2).sum()) / Rr.size(0)
    energy = ((Rr @ Q).pow(2).sum(0) / Rr.size(0))
    return Subspace(U=Q.float().contiguous(), energy=energy.float(), total_energy=Cr_full_energy, rank=k)


def compute_protected_subspaces(model, data: DataBundle, cfg: Config, device: torch.device, rows_cap: int | None = None) -> ProtectedSubspaces:
    if rows_cap is None:
        rows_cap = 2048 if cfg.mock else 768
    in_r, out_r = _collect(model, data.retain_train, cfg, device, rows_cap, cfg.data.seed, data.pad_id)
    if cfg.fisher.subspace == "discriminative":
        in_f, out_f = _collect(model, data.forget_train, cfg, device, rows_cap, cfg.data.seed + 11, data.pad_id)
        ins = {n: _discriminative_subspace(in_r[n].rows, in_f[n].rows, cfg) for n in in_r if in_r[n].rows is not None}
        outs = {n: _discriminative_subspace(out_r[n].rows, out_f[n].rows, cfg) for n in out_r if out_r[n].rows is not None}
        ps = ProtectedSubspaces(inputs=ins, outputs=outs)
    else:
        ps = ProtectedSubspaces(inputs=_finalize(in_r, cfg), outputs=_finalize(out_r, cfg))
    for s in list(ps.inputs.values()) + list(ps.outputs.values()):
        s.U = s.U.to(device)
    return ps


@torch.no_grad()
def project_lora_grads(model, ps: ProtectedSubspaces, strength: float, project_output: bool = True) -> dict[str, float]:
    """In-place: g_A <- g_A (I - s UUᵀ), g_B <- (I - s VVᵀ) g_B. Returns mean fraction of gradient norm removed."""
    if strength <= 0.0:
        return {"removed_frac": 0.0}
    removed, total = 0.0, 0.0
    for name, (_, A, B) in lora_pairs(model).items():
        if A.weight.grad is not None and name in ps.inputs:
            U = ps.inputs[name].U                      # (d, k)
            g = A.weight.grad                          # (r, d)
            comp = (g @ U) @ U.T                       # component inside protected subspace
            total += float(g.pow(2).sum())
            removed += float(comp.pow(2).sum()) * strength
            g.sub_(strength * comp)
        if project_output and B.weight.grad is not None and name in ps.outputs:
            V = ps.outputs[name].U                     # (out, k)
            g = B.weight.grad                          # (out, r)
            comp = V @ (V.T @ g)
            total += float(g.pow(2).sum())
            removed += float(comp.pow(2).sum()) * strength
            g.sub_(strength * comp)
    return {"removed_frac": removed / max(total, 1e-12)}


@torch.no_grad()
def project_lora_update(model, ps: ProtectedSubspaces, before: dict[str, torch.Tensor], strength: float, project_output: bool = True) -> None:
    """Replace each LoRA param with before + P(delta) where delta = param - before.
    Guarantees the realised update (not just the gradient) lies outside the protected subspace."""
    if strength <= 0.0:
        return
    for name, (_, A, B) in lora_pairs(model).items():
        if name in ps.inputs:
            U = ps.inputs[name].U
            d = A.weight - before[name + ".A"]
            A.weight.copy_(before[name + ".A"] + d - strength * (d @ U) @ U.T)
        if project_output and name in ps.outputs:
            V = ps.outputs[name].U
            d = B.weight - before[name + ".B"]
            B.weight.copy_(before[name + ".B"] + d - strength * V @ (V.T @ d))


@torch.no_grad()
def snapshot_lora(model) -> dict[str, torch.Tensor]:
    return {k: t for name, (_, A, B) in lora_pairs(model).items() for k, t in ((name + ".A", A.weight.clone()), (name + ".B", B.weight.clone()))}
