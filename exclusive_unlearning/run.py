"""Run one method end-to-end: build data/model, prepare (Fisher / ref), train, full-eval, append CSV row."""
from __future__ import annotations

import argparse
import csv
import json
import time

import torch

from .config import RESULTS_DIR, make_config
from .dataset import build_data
from .evaluate import full_eval, quick_eval
from .models import load_base_model, load_tokenizer, resolve_device, wrap_lora
from .trainer import Unlearner

FIELDS = ["method", "seed", "steps", "forget_nll", "forget_entropy", "refusal_rate", "retain_nll",
          "pass_at_1", "probe_acc", "relearn_recovery", "collapsed", "fisher_strength", "fisher_rank", "seconds", "pruned_frac", "retain_qa_acc", "forget_qa_acc", "task", "model"]


DEMO_PROMPTS = [
    ("general", "Question: What is the capital of France?\nAnswer:"),
    ("general", "Question: Who wrote the play Romeo and Juliet?\nAnswer:"),
    ("general", "Question: Why is the sky blue?\nAnswer:"),
    ("general", "Write a short poem about the ocean.\n"),
    ("code", "def is_prime(n):\n    \"\"\"Return True if n is a prime number.\"\"\"\n"),
    ("code", "def reverse_string(s):\n    \"\"\"Return the string s reversed.\"\"\"\n"),
]


@torch.no_grad()
def write_samples(model, tok, cfg, device, method):
    """Greedy generations on fixed prompts, saved for the before/after demo video."""
    model.eval()
    out = []
    for kind, prompt in DEMO_PROMPTS:
        enc = tok(prompt, return_tensors="pt").to(device)
        g = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        out.append({"kind": kind, "prompt": prompt, "output": tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)})
    tag = cfg.model.name_or_path.split("/")[-1].lower()
    path = RESULTS_DIR / f"samples_{tag}_{method}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"[samples] wrote {path.name}", flush=True)


def run(method: str, mock: bool, seed: int = 0, overrides: dict | None = None, csv_path=None) -> dict:
    overrides = overrides or {}
    cfg = make_config(method, mock=mock, **overrides)
    cfg.data.seed = seed
    cfg.train.output_dir = cfg.train.output_dir.parent / f"{method}_seed{seed}"
    torch.manual_seed(seed)
    device = resolve_device(cfg.train.device)
    tok = load_tokenizer(cfg)
    data = build_data(cfg, tok)
    model = wrap_lora(load_base_model(cfg, device), cfg)

    # Reference numbers from the un-modified model (LoRA B is zero-init so this is the base model).
    base_m = quick_eval(model, data, cfg, device)
    t0 = time.time()
    prune_stats = {}
    if cfg.prune.enabled:
        from .prune import prune_model
        prune_stats = prune_model(model, data, cfg, device)
        print(f"[prune] post-prune quick eval: {quick_eval(model, data, cfg, device)}")
    un = Unlearner(model, data, cfg, device)
    un.prepare()
    hist = un.train(eval_fn=lambda m: quick_eval(m, data, cfg, device))
    secs = time.time() - t0

    if not cfg.mock:
        if cfg.data.task == "code":
            write_samples(model, tok, cfg, device, method if seed == 0 else f"{method}_seed{seed}")  # seed 0 keeps the original file names
    metrics = full_eval(model, data, cfg, device, base_forget_nll=base_m["forget_nll"], with_relearn=method != "base")
    qa_rows = metrics.pop("_qa_rows", None)
    if qa_rows is not None:
        tag = cfg.model.name_or_path.split("/")[-1].lower()
        (RESULTS_DIR / f"qa_{cfg.data.task}_{tag}_{method}.json").write_text(json.dumps(qa_rows, indent=2))
    row = {
        "method": method, "seed": seed, "steps": len(hist),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "collapsed": int(any(h.get("collapsed") for h in hist)),
        "fisher_strength": cfg.fisher.strength,
        "fisher_rank": un.ps.summary()["mean_rank"] if un.ps else 0,
        "seconds": round(secs, 1),
        "pruned_frac": round(prune_stats.get("pruned_frac", 0.0), 5),
        "task": cfg.data.task, "model": cfg.model.name_or_path,
    }
    row.setdefault("relearn_recovery", 0.0)
    print(f"[result] {json.dumps(row)}")
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cg_oss")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like fisher.strength=0.5")
    ap.add_argument("--laptop", action="store_true", help="16 GB-RAM preset for the real model")
    a = ap.parse_args()
    ov = {}
    for s in a.set:
        k, _, v = s.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        ov[k] = v
    if a.laptop:
        ov["laptop"] = True
    run(a.method, a.mock, a.seed, ov, csv_path=RESULTS_DIR / a.csv if a.csv else None)


if __name__ == "__main__":
    main()
