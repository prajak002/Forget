"""Cross-route recall probe on the saved Qwen2.5-Coder-0.5B unlearned adapters (see exclusive_unlearning/recall_probe.py).

  uv run python run_recall_probe.py                                   # pilot: 20 facts, base + cg_oss + max_entropy + npo
  uv run python run_recall_probe.py --facts full                      # 100 facts
  uv run python run_recall_probe.py --models base cg_oss --tag pilot
  uv run python run_recall_probe.py --facts full --models cg_oss@1 max_entropy@1 --tag seeds

Writes results/recall_probe_<tag>_responses.jsonl (every prompt, output, category) and
results/recall_probe_<tag>_summary.csv. Analysis: analyze_recall_probe.py.
"""
import argparse
import json
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from exclusive_unlearning.config import CHECKPOINT_DIR, RESULTS_DIR
from exclusive_unlearning.models import resolve_device
from exclusive_unlearning.recall_probe import FACTS, PILOT_FACTS, ROUTES, build_prompts, categorize

MODEL = "Qwen/Qwen2.5-Coder-0.5B"


def load(name, device):
    """name is "base", a method ("cg_oss" = seed 0) or "method@seed" ("cg_oss@1")."""
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    if name != "base":
        method, _, seed = name.partition("@")
        path = CHECKPOINT_DIR / f"{method}_seed{seed or 0}" / "final"
        if not (path / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"no saved adapter at {path}")
        m = PeftModel.from_pretrained(m, path).to(device)
    return m.eval()


@torch.no_grad()
def generate(model, tok, prompts, device, max_new_tokens, batch_size):
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "cg_oss", "max_entropy", "npo"])
    ap.add_argument("--facts", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--tag", default=None, help="defaults to --facts")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)  # >1 left-pads, which shifts bf16 greedy outputs
    a = ap.parse_args()

    facts = FACTS if a.facts == "full" else PILOT_FACTS
    a.tag = a.tag or a.facts
    device = resolve_device("auto")
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    jobs = [(fi, route, build_prompts(f)[route]) for fi, f in enumerate(facts) for route in ROUTES]

    rows_path = RESULTS_DIR / f"recall_probe_{a.tag}_responses.jsonl"
    rows = []
    for name in a.models:
        t0 = time.time()
        model = load(name, device)
        outs = generate(model, tok, [p for _, _, p in jobs], device, a.max_new_tokens, a.batch_size)
        for (fi, route, prompt), out in zip(jobs, outs):
            f = facts[fi]
            category, span = categorize(route, f, prompt, out)
            rows.append({"model": name, "fact_id": fi, "relation": f.relation, "subject": f.subject,
                         "aliases": list(f.aliases), "route": route, "prompt": prompt, "output": out,
                         "span": span, "category": category})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        acc = sum(r["category"] == "correct" for r in rows if r["model"] == name) / len(jobs)
        print(f"[probe] {name}: {len(jobs)} prompts, overall correct {acc:.2f}, {time.time() - t0:.0f}s", flush=True)

    with open(rows_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = (df.assign(n=1).pivot_table(index=["model", "route"], columns="category", values="n", aggfunc="sum", fill_value=0)
               .reset_index())
    summary.to_csv(RESULTS_DIR / f"recall_probe_{a.tag}_summary.csv", index=False)
    print(f"[probe] wrote {rows_path.name} and recall_probe_{a.tag}_summary.csv")


if __name__ == "__main__":
    main()
