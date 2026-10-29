"""Cross-route recall probe on Qwen2.5-Coder-0.5B and its unlearned adapters (see exclusive_unlearning/recall_probe.py).

  uv run python run_recall_probe.py                                   # pilot: 20 facts, base + cg_oss + max_entropy + npo
  uv run python run_recall_probe.py --facts full                      # 100 facts
  uv run python run_recall_probe.py --facts full --models cg_oss@1 max_entropy@1 --tag seeds
  # large probe (895 facts, 13 relations) and RWKU, adapter straight from the Hugging Face Hub:
  python run_recall_probe.py --facts large --models base cg_oss=Pras13/Qwen2.5-Coder-0.5B-CG-OSS
  python run_recall_probe.py --facts rwku  --models base cg_oss=Pras13/Qwen2.5-Coder-0.5B-CG-OSS

A model is "base", a saved method ("cg_oss" = checkpoints/cg_oss_seed0/final, "cg_oss@1" = seed 1), or
"name=<local adapter dir or Hub repo id>". Writes results/recall_probe_<tag>_responses.jsonl (every prompt, output,
category; rewritten after each model, so --resume can continue an interrupted run), _summary.csv and _meta.json.
Analysis: analyze_recall_probe.py --tag <tag>.

Generation is greedy. Early stopping (on by default) ends a response once its score is fixed (recall_probe.is_complete:
first line finished, string literal closed, function body ended), which never changes a category.
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from exclusive_unlearning.config import CHECKPOINT_DIR, RESULTS_DIR, ROOT
from exclusive_unlearning.models import resolve_device
from exclusive_unlearning.recall_probe import FACT_SETS, RELATIONS, ROUTES, build_prompts, categorize, is_complete

MODEL = "Qwen/Qwen2.5-Coder-0.5B"
RWKU_PATH = ROOT / "data" / "rwku" / "rwku_probe.jsonl"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def adapter_source(spec: str) -> tuple[str, str | None]:
    """'base' -> (base, None); 'cg_oss@1' -> local checkpoint; 'name=dir_or_repo' -> as given."""
    name, _, src = spec.partition("=")
    if name == "base" or src:
        return name, src or None
    method, _, seed = name.partition("@")
    path = CHECKPOINT_DIR / f"{method}_seed{seed or 0}" / "final"
    if not (path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"no saved adapter at {path}; pass {name}=<adapter dir or Hub repo id>")
    return name, str(path)


def load(base_model: str, src: str | None, device, dtype):
    m = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    if src:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, src).to(device)
    return m.eval()


class Done(StoppingCriteria):
    """Per-row stop once recall_probe.is_complete says the rest of the output cannot change the score."""

    def __init__(self, tok, prompt_len: int, routes, kinds):
        self.tok, self.prompt_len, self.routes, self.kinds = tok, prompt_len, routes, kinds

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tok.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=True)
        return torch.tensor([is_complete(r, t, k) for r, t, k in zip(self.routes, texts, self.kinds)],
                            dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def generate(model, tok, jobs, device, max_new_tokens, batch_size, early_stop, label=""):
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["prompt"]))  # similar lengths -> less padding
    outs, t0 = [None] * len(jobs), time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([jobs[i]["prompt"] for i in idx], return_tensors="pt", padding=True).to(device)
        plen = enc["input_ids"].shape[1]
        stop = StoppingCriteriaList([Done(tok, plen, [jobs[i]["route"] for i in idx], [jobs[i]["kind"] for i in idx])]) \
            if early_stop else None
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                             stopping_criteria=stop)
        for i, text in zip(idx, tok.batch_decode(gen[:, plen:], skip_special_tokens=True)):
            outs[i] = text
        done = b + len(idx)
        if done == len(jobs) or (b // batch_size) % 50 == 0:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"[probe] {label} {done}/{len(jobs)} prompts, {rate:.1f}/s, eta {(len(jobs) - done) / rate / 60:.1f} min",
                  flush=True)
    return outs


def select_facts(a) -> list[tuple[int, object]]:
    """(fact_id, Fact) pairs. With --limit, a fixed-seed random subset whose ids are those of the full set."""
    if a.facts == "rwku":
        from exclusive_unlearning.rwku import load_rwku
        if not a.rwku_path.exists():
            raise FileNotFoundError(f"{a.rwku_path} missing; build it with: python build_rwku_probe.py")
        facts = load_rwku(a.rwku_path)
    else:
        facts = FACT_SETS[a.facts]
    ids = list(range(len(facts)))
    if a.limit and a.limit < len(ids):
        ids = sorted(random.Random(a.seed).sample(ids, a.limit))
    return [(i, facts[i]) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "cg_oss", "max_entropy", "npo"],
                    help='"base", "cg_oss", "cg_oss@1", or "name=<adapter dir or Hub repo id>"')
    ap.add_argument("--facts", choices=["pilot", "full", "large", "rwku"], default="pilot")
    ap.add_argument("--rwku-path", type=Path, default=RWKU_PATH)
    ap.add_argument("--limit", type=int, default=0, help="probe a random subset of this many facts (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --limit")
    ap.add_argument("--tag", default=None, help="defaults to --facts")
    ap.add_argument("--base-model", default=MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default 1 on mps (bf16 left padding shifts greedy outputs), 32 on cuda, 8 on cpu")
    ap.add_argument("--dtype", choices=["auto", *DTYPES], default="auto", help="auto: bfloat16 on mps, float32 elsewhere")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep finished models from an existing responses file")
    ap.add_argument("--workers", type=int, default=8, help="parallel sandboxes for scoring the func route")
    a = ap.parse_args()

    a.tag = a.tag or a.facts
    device = resolve_device("auto")
    dtype = DTYPES[a.dtype] if a.dtype != "auto" else (torch.bfloat16 if device.type == "mps" else torch.float32)
    bs = a.batch_size or {"mps": 1, "cuda": 32}.get(device.type, 8)
    sources = dict(adapter_source(s) for s in a.models)
    facts = select_facts(a)
    jobs = [{"fid": fid, "route": route, "prompt": prompt, "kind": RELATIONS[f.relation].kind}
            for fid, f in facts for route, prompt in build_prompts(f).items()]
    assert all(j["route"] in ROUTES for j in jobs)
    fact_by_id = dict(facts)
    print(f"[probe] {a.facts}: {len(facts)} facts x {len(ROUTES)} routes = {len(jobs)} prompts per model; "
          f"device={device.type} dtype={str(dtype).split('.')[-1]} batch={bs} early_stop={not a.no_early_stop}", flush=True)

    rows_path = RESULTS_DIR / f"recall_probe_{a.tag}_responses.jsonl"
    rows = []
    if a.resume and rows_path.exists():
        with open(rows_path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        want = {(j["fid"], j["route"]) for j in jobs}
        finished = {m for m in {r["model"] for r in rows}
                    if {(r["fact_id"], r["route"]) for r in rows if r["model"] == m} >= want}
        rows = [r for r in rows if r["model"] in finished]
        print(f"[probe] resuming: already finished {sorted(finished) or 'none'}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for name, src in sources.items():
        if any(r["model"] == name for r in rows):
            continue
        t0 = time.time()
        model = load(a.base_model, src, device, dtype)
        outs = generate(model, tok, jobs, device, a.max_new_tokens, bs, not a.no_early_stop, label=name)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        with ThreadPoolExecutor(a.workers) as ex:
            scored = list(ex.map(lambda jo: categorize(jo[0]["route"], fact_by_id[jo[0]["fid"]], jo[0]["prompt"], jo[1]),
                                 zip(jobs, outs)))
        for j, out, (category, span) in zip(jobs, outs, scored):
            f = fact_by_id[j["fid"]]
            rows.append({"model": name, "fact_id": j["fid"], "relation": f.relation, "subject": f.subject,
                         "aliases": list(f.aliases), "entity": f.entity, "answer_type": f.answer_type, "route": j["route"],
                         "prompt": j["prompt"], "output": out, "span": span, "category": category})
        with open(rows_path, "w") as fh:  # after every model, so an interrupted run can --resume
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        acc = sum(c == "correct" for c, _ in scored) / len(jobs)
        print(f"[probe] {name}: {len(jobs)} prompts, overall correct {acc:.2f}, {time.time() - t0:.0f}s", flush=True)

    import pandas as pd
    import peft
    import transformers
    df = pd.DataFrame(rows)
    summary = (df.assign(n=1).pivot_table(index=["model", "route"], columns="category", values="n", aggfunc="sum", fill_value=0)
               .reset_index())
    summary.to_csv(RESULTS_DIR / f"recall_probe_{a.tag}_summary.csv", index=False)
    meta = {"tag": a.tag, "facts": a.facts, "n_facts": len(facts), "limit": a.limit, "seed": a.seed,
            "base_model": a.base_model, "models": sources, "device": device.type, "dtype": str(dtype).split(".")[-1],
            "batch_size": bs, "early_stop": not a.no_early_stop, "max_new_tokens": a.max_new_tokens,
            "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RESULTS_DIR / f"recall_probe_{a.tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[probe] wrote {rows_path.name}, recall_probe_{a.tag}_summary.csv and _meta.json")


if __name__ == "__main__":
    main()
