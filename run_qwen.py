"""Real-model run on Qwen/Qwen2.5-Coder-1.5B with the 16 GB laptop preset.

Methods (in order): base, npo, max_entropy, cg_oss, prune (weight removal only), prune_cg_oss
(weight removal followed by curvature-guarded max-entropy unlearning). Results append to
results/qwen.csv, one row per method, so a crash mid-way keeps finished rows.

  uv run python run_qwen.py                       # all methods
  uv run python run_qwen.py --methods prune prune_cg_oss
  uv run python run_qwen.py --methods cg_oss max_entropy --seeds 1 2
"""
import argparse
import gc
import traceback

import torch

from exclusive_unlearning.config import RESULTS_DIR
from exclusive_unlearning.run import run


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["base", "prune", "prune_cg_oss", "cg_oss", "max_entropy", "npo"])
    ap.add_argument("--prune-alpha", type=float, default=2.0)
    ap.add_argument("--prune-frac", type=float, default=0.10)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--task", default="code", choices=["code", "indian_football"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    a = ap.parse_args()

    tag = a.model.split("/")[-1].lower()
    out = RESULTS_DIR / (f"{tag}.csv" if a.task == "code" else f"{a.task}_{tag}.csv")
    for seed, m in ((s, m) for s in a.seeds for m in a.methods):
        ov = {"laptop": True, "prune.alpha": a.prune_alpha, "prune.max_fraction": a.prune_frac, "model.name_or_path": a.model, "data.task": a.task}
        print(f"\n========== {m} seed {seed} ==========", flush=True)
        try:
            run(m, mock=False, seed=seed, overrides=ov, csv_path=out)
        except Exception:
            traceback.print_exc()
            print(f"[qwen] {m} seed {seed} FAILED", flush=True)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print("[qwen] ALL DONE", flush=True)


if __name__ == "__main__":
    _main()
