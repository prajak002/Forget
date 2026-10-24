"""Calibrate weight-removal unlearning on the real model (memory-lean).

Computes the forget/retain Fisher ratio once (layer-grouped), then for each removal fraction zeroes the
top-ratio MLP weights per matrix, measures forget/retain NLL, and undoes the removal. A random-removal
control at the same fraction tests whether the Fisher ratio is selective. Writes results/prune_sweep.csv.

  uv run python prune_sweep.py --laptop
"""
import argparse
import csv
import time

import torch

from exclusive_unlearning import prune as P
from exclusive_unlearning.config import RESULTS_DIR, make_config
from exclusive_unlearning.dataset import build_data
from exclusive_unlearning.evaluate import quick_eval
from exclusive_unlearning.models import load_base_model, load_tokenizer, resolve_device


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laptop", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    cfg = make_config("prune", mock=a.mock, **({"laptop": True} if a.laptop else {}))
    cfg.prune.n_samples = a.n_samples
    if a.model:
        cfg.model.name_or_path = a.model
    dev = resolve_device(cfg.train.device)
    tok = load_tokenizer(cfg)
    data = build_data(cfg, tok)
    model = load_base_model(cfg, dev)
    P._targets = tuple(cfg.prune.targets) if cfg.prune.targets else None
    print(f"[sweep] device={dev} prunable params={sum(p.numel() for p in P._block_params(model).values())/1e6:.0f}M", flush=True)

    rows = []
    base = quick_eval(model, data, cfg, dev)
    rows.append({"kind": "base", "fraction": 0.0, **base})
    print(f"[sweep] base forget_nll={base['forget_nll']:.3f} forget_ent={base['forget_entropy']:.3f} retain_nll={base['retain_nll']:.3f}", flush=True)

    t0 = time.time()
    ratio = P.fisher_ratio(model, data, cfg, dev, cfg.prune.n_samples, cfg.prune.group_layers)
    print(f"[sweep] fisher ratio in {time.time() - t0:.0f}s", flush=True)
    rand = {n: torch.rand(r.shape, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16) for n, r in ratio.items()}

    for frac in a.fractions:
        for kind, rr in (("fisher", ratio), ("random", rand)):
            _, log = P.remove_by_ratio(model, rr, frac, None, undo=True)
            m = quick_eval(model, data, cfg, dev)
            P.undo_removal(model, log)
            del log
            rows.append({"kind": kind, "fraction": frac, **m})
            print(f"[sweep] {kind:6s} frac={frac:<6} forget_nll={m['forget_nll']:.3f} forget_ent={m['forget_entropy']:.3f} retain_nll={m['retain_nll']:.3f}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = "mock" if a.mock else cfg.model.name_or_path.split("/")[-1].lower()
    with open(RESULTS_DIR / f"prune_sweep_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("[sweep] DONE", flush=True)


if __name__ == "__main__":
    _main()
