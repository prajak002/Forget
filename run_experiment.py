"""Phase 4: CG-OSS main runs, ablations and the forget/retain Pareto sweep.

Outputs
  results/cg_oss.csv   : CG-OSS at default settings, one row per seed
  results/sweep.csv    : projection strength, ratio threshold, subspace type, Fisher mode ablations
  results/pareto.csv   : Max-Ent vs CG-OSS across retain-loss weights (the trade-off curve)
"""
import argparse

from exclusive_unlearning.config import RESULTS_DIR
from exclusive_unlearning.run import run


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--pareto", action="store_true")
    a = ap.parse_args()


    def fresh(name):
        p = RESULTS_DIR / name
        if p.exists():
            p.unlink()
        return p


    out = fresh("cg_oss.csv")
    for seed in a.seeds:
        run("cg_oss", a.mock, seed, csv_path=out)

    if a.sweep:
        sweep = fresh("sweep.csv")
        s0 = a.seeds[0]
        for s in [0.0, 0.5, 0.9, 1.0]:
            run("cg_oss", a.mock, s0, {"fisher.strength": s, "loss.w_retain": 0.05}, csv_path=sweep)
        for thr in [0.3, 1.0, 3.0]:
            run("cg_oss", a.mock, s0, {"fisher.ratio_threshold": thr, "loss.w_retain": 0.05}, csv_path=sweep)
        run("cg_oss", a.mock, s0, {"fisher.subspace": "retain", "loss.w_retain": 0.05}, csv_path=sweep)
        run("cg_oss", a.mock, s0, {"fisher.mode": "empirical", "loss.w_retain": 0.05}, csv_path=sweep)
        run("cg_oss", a.mock, s0, {"fisher.project_update": False, "loss.w_retain": 0.05}, csv_path=sweep)

    if a.pareto:
        pareto = fresh("pareto.csv")
        for w in [0.0, 0.02, 0.05, 0.1, 0.3, 1.0]:
            for seed in a.seeds:
                for m in ["max_entropy", "cg_oss", "npo"]:
                    run(m, a.mock, seed, {"loss.w_retain": w}, csv_path=pareto)


if __name__ == "__main__":
    _main()
