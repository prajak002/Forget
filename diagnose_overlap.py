"""Phase 5 diagnostic: how much forget-domain input energy lies inside the retain protected subspace,
per LoRA-wrapped module. Values near 1 mean the domains are co-linear and projection cannot separate them."""
import argparse
import csv

import torch

from exclusive_unlearning.config import RESULTS_DIR, make_config
from exclusive_unlearning.dataset import build_data
from exclusive_unlearning.fisher import _Reservoir, compute_protected_subspaces
from exclusive_unlearning.models import load_base_model, load_tokenizer, lora_pairs, resolve_device, wrap_lora


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    rows = []
    for subspace in ["retain", "discriminative"]:
        cfg = make_config("cg_oss", mock=a.mock, **{"fisher.subspace": subspace})
        dev = resolve_device(cfg.train.device)
        tok = load_tokenizer(cfg)
        data = build_data(cfg, tok)
        model = wrap_lora(load_base_model(cfg, dev), cfg)
        ps = compute_protected_subspaces(model, data, cfg, dev)
        res = {n: _Reservoir(1024) for n in lora_pairs(model)}
        hs = [w.register_forward_hook(lambda m, i, o, _n=n: res[_n].add(i[0])) for n, (w, _, _) in lora_pairs(model).items()]
        with torch.no_grad():
            for b in data.loader(data.forget_train, cfg.train.batch_size):
                model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev))
                if res[next(iter(res))].seen >= 1024:
                    break
        for h in hs:
            h.remove()
        for n, r in res.items():
            X, U = r.rows, ps.inputs[n].U.cpu()
            inside = float((X @ U).pow(2).sum() / X.pow(2).sum())
            parts = n.split(".")
            layer = next((p for p in parts if p.isdigit()), "?")
            rows.append({"subspace": subspace, "layer": layer, "module": parts[-1], "rank": ps.inputs[n].rank,
                         "dim": U.shape[0], "retain_energy_captured": round(ps.inputs[n].captured, 4),
                         "forget_energy_inside": round(inside, 4)})
            print(rows[-1])

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "subspace_overlap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    _main()
