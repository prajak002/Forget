"""Phase 3: Base, GA, NPO (+ unprojected max-entropy ablation) -> results/baselines.csv"""
import argparse

from exclusive_unlearning.config import RESULTS_DIR
from exclusive_unlearning.run import run


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    out = RESULTS_DIR / "baselines.csv"
    if out.exists():
        out.unlink()
    for seed in a.seeds:
        for m in ["base", "ga", "npo", "max_entropy"]:
            run(m, a.mock, seed, csv_path=out)


if __name__ == "__main__":
    _main()
