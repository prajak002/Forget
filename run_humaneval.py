"""Full HumanEval (164 problems) on the saved Qwen2.5-Coder-0.5B adapters, separate from the recall probe.

Same greedy settings as the laptop preset (max_new_tokens 192, 5 s timeout, same truncation and sandbox), so
the first 40 problems should reproduce the Pass@1 already in results/qwen2.5-coder-0.5b.csv.

  uv run python run_humaneval.py                       # base, cg_oss, max_entropy, npo
  uv run python run_humaneval.py --models cg_oss

Writes results/humaneval164_problems.csv (one row per model x problem) and prints Pass@1, failures,
timeouts, the first-40 check, and exact McNemar tests against cg_oss.
"""
import argparse
import time

import pandas as pd
import torch
from scipy.stats import binomtest
from transformers import AutoTokenizer

from exclusive_unlearning.code_eval import _truncate, program_status
from exclusive_unlearning.config import RESULTS_DIR
from exclusive_unlearning.models import resolve_device
from run_recall_probe import MODEL, load

MAX_NEW_TOKENS, TIMEOUT = 192, 5.0


@torch.no_grad()
def complete(model, tok, prompt, device):
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "cg_oss", "max_entropy", "npo"])
    a = ap.parse_args()
    probs = [dict(r) for r in load_dataset("openai/openai_humaneval", split="test")]
    device = resolve_device("auto")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    path = RESULTS_DIR / "humaneval164_problems.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model"])
    rows = []
    for name in a.models:
        t0 = time.time()
        model = load(name, device)
        for i, pr in enumerate(probs):
            comp = _truncate(complete(model, tok, pr["prompt"], device))
            program = pr["prompt"] + comp + "\n\n" + pr["test"] + f"\ncheck({pr['entry_point']})\n"
            rows.append({"model": name, "idx": i, "task_id": pr["task_id"], "status": program_status(program, TIMEOUT),
                         "completion": comp})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        st = pd.Series([r["status"] for r in rows if r["model"] == name])
        print(f"[humaneval] {name}: pass@1 {(st == 'pass').mean():.3f} ({(st == 'pass').sum()}/164), "
              f"fail {(st == 'fail').sum()}, timeout {(st == 'timeout').sum()}, {time.time() - t0:.0f}s", flush=True)
        # save after every model so an interruption keeps finished models
        df = pd.concat([old[~old.model.isin(a.models)], pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(path, index=False)

    df = pd.read_csv(path)
    ok = df.assign(p=df.status.eq("pass")).pivot(index="idx", columns="model", values="p")
    summary = pd.DataFrame({"pass@1 (164)": ok.mean(), "pass@1 (first 40)": ok.iloc[:40].mean(),
                            "fail": df[df.status == "fail"].groupby("model").size(),
                            "timeout": df[df.status == "timeout"].groupby("model").size()}).fillna(0)
    print(summary.round(3).to_string())
    if "cg_oss" in ok:
        for m in ok.columns.drop("cg_oss"):
            b, c = int((ok.cg_oss & ~ok[m]).sum()), int((~ok.cg_oss & ok[m]).sum())
            p = binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")
            print(f"[mcnemar] cg_oss vs {m}: cg_oss-only {b}, {m}-only {c}, p={p:.4f}")


if __name__ == "__main__":
    main()
