"""Unlearning loop: forget objective (GA / NPO / max-entropy) + retain CE, with optional
curvature-guarded null-space projection of the forget gradient (CG-OSS).

Per step:
  1. forget batch  -> L_f, backward, [project LoRA grads out of the retain Fisher subspace]
  2. retain batch  -> L_r, backward (unprojected: the retain gradient may use any direction)
  3. clip, optimizer step
The forget and retain gradients are accumulated separately so the projection touches only
the forget signal.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import Config
from .dataset import DataBundle, infinite
from .fisher import (ProtectedSubspaces, compute_protected_subspaces, project_lora_grads,
                     project_lora_update, snapshot_lora)
from .losses import forget_loss, retain_loss, token_entropy


class Unlearner:
    def __init__(self, model, data: DataBundle, cfg: Config, device: torch.device, ref_model=None):
        self.model, self.data, self.cfg, self.device = model, data, cfg, device
        self.ref_model = ref_model
        self.ps: ProtectedSubspaces | None = None
        self.history: list[dict] = []
        self.params = [p for p in model.parameters() if p.requires_grad]
        # Separate optimizers so Adam state (and the projection) for the forget signal never mixes
        # with the retain signal.
        f_lr = cfg.train.forget_lr if cfg.train.forget_lr is not None else cfg.train.lr
        if cfg.train.forget_optimizer == "sgd":
            self.opt_f = torch.optim.SGD(self.params, lr=f_lr, momentum=0.9, weight_decay=cfg.train.weight_decay)
        else:
            self.opt_f = torch.optim.AdamW(self.params, lr=f_lr, weight_decay=cfg.train.weight_decay)
        self.opt_r = torch.optim.AdamW(self.params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.sched_f = torch.optim.lr_scheduler.LambdaLR(self.opt_f, self._lr_lambda)
        self.sched_r = torch.optim.lr_scheduler.LambdaLR(self.opt_r, self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        t = self.cfg.train
        if step < t.warmup_steps:
            return (step + 1) / max(t.warmup_steps, 1)
        prog = (step - t.warmup_steps) / max(t.max_steps - t.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    # ------------------------------------------------------------------ #
    def prepare(self):
        """Build reference model (NPO) and protected subspaces (CG-OSS) before unlearning."""
        if self.cfg.loss.forget_objective == "npo" and self.ref_model is None:
            self.ref_model = copy.deepcopy(self.model).eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        if self.cfg.fisher.strength > 0.0:
            self._refresh_fisher(step=0)

    def _refresh_fisher(self, step: int):
        t0 = time.time()
        self.ps = compute_protected_subspaces(self.model, self.data, self.cfg, self.device)
        s = self.ps.summary()
        s["seconds"] = round(time.time() - t0, 2)
        s["step"] = step
        print(f"[fisher] protected subspaces: {s}")
        self.cfg.train.output_dir.mkdir(parents=True, exist_ok=True)
        self.ps.save(self.cfg.train.output_dir / "protected_subspaces.pt")

    def _to(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def _forget_step(self, batch) -> dict[str, float]:
        c = self.cfg.loss
        out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        ref_logits = None
        if c.forget_objective == "npo":
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        lf = forget_loss(c.forget_objective, out.logits, batch["labels"], ref_logits, c.npo_beta, c.max_entropy_temperature)
        (c.w_forget * lf / self.cfg.train.grad_accum).backward()
        with torch.no_grad():
            ent = token_entropy(out.logits, batch["labels"])
        return {"loss_forget": float(lf), "forget_entropy": float(ent)}

    def _retain_step(self, batch) -> dict[str, float]:
        out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        lr = retain_loss(out.logits, batch["labels"])
        (self.cfg.loss.w_retain * lr / self.cfg.train.grad_accum).backward()
        return {"loss_retain": float(lr)}

    def train(self, eval_fn=None) -> list[dict]:
        t = self.cfg.train
        self.model.train()
        f_it = infinite(self.data.loader(self.data.forget_train, t.batch_size, seed=self.cfg.data.seed))
        r_it = infinite(self.data.loader(self.data.retain_train, t.batch_size, seed=self.cfg.data.seed))
        t.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = t.output_dir / "train_log.csv"

        for step in range(t.max_steps):
            rec: dict[str, float] = {"step": step}
            if self.ps is not None and self.cfg.fisher.refresh_every and step > 0 and step % self.cfg.fisher.refresh_every == 0:
                self._refresh_fisher(step)
            # --- forget phase: accumulate, project gradient, Adam step, project realised update ---
            if self.cfg.loss.w_forget > 0:
                self.opt_f.zero_grad(set_to_none=True)
                agg: dict[str, float] = {}
                for _ in range(t.grad_accum):
                    m = self._forget_step(self._to(next(f_it)))
                    for k, v in m.items():
                        agg[k] = agg.get(k, 0.0) + v / t.grad_accum
                rec.update(agg)
                if self.ps is not None:
                    rec.update(project_lora_grads(self.model, self.ps, self.cfg.fisher.strength))
                rec["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(self.params, t.grad_clip))
                if not math.isfinite(rec.get("loss_forget", 0.0)):
                    print(f"[step {step}] non-finite forget loss; stopping (collapse)")
                    rec["collapsed"] = 1.0
                    self.history.append(rec)
                    break
                before = snapshot_lora(self.model) if (self.ps is not None and self.cfg.fisher.project_update) else None
                self.opt_f.step()
                if before is not None:
                    project_lora_update(self.model, self.ps, before, self.cfg.fisher.strength)
            self.sched_f.step()
            # --- retain phase: plain (unprojected) step ---
            if self.cfg.loss.w_retain > 0:
                self.opt_r.zero_grad(set_to_none=True)
                agg = {}
                for _ in range(t.grad_accum):
                    m = self._retain_step(self._to(next(r_it)))
                    for k, v in m.items():
                        agg[k] = agg.get(k, 0.0) + v / t.grad_accum
                rec.update(agg)
                rec["grad_norm_retain"] = float(torch.nn.utils.clip_grad_norm_(self.params, t.grad_clip))
                self.opt_r.step()
            self.sched_r.step()
            rec["lr"] = self.sched_f.get_last_lr()[0]

            if eval_fn is not None and t.eval_every and (step + 1) % t.eval_every == 0:
                self.model.eval()
                rec.update({f"eval_{k}": v for k, v in eval_fn(self.model).items()})
                self.model.train()
            if t.log_every and step % t.log_every == 0:
                print("[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if t.save_every and (step + 1) % t.save_every == 0:
                self.save(t.output_dir / f"step_{step + 1}")
            self.history.append(rec)

        self.save(t.output_dir / "final")
        self._write_log(log_path)
        return self.history

    # ------------------------------------------------------------------ #
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained") and not self.cfg.mock:
            self.model.save_pretrained(path)
        else:
            torch.save({k: v for k, v in self.model.state_dict().items() if "lora_" in k}, path / "lora_state.pt")
        with open(path / "config.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=2, default=str)

    def _write_log(self, path: Path):
        keys: list[str] = []
        for r in self.history:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)
