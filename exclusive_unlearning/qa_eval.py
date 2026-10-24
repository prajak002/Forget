"""Practical knowledge test: does the model still answer the question correctly?

A question counts as correct if any accepted answer string appears (case-insensitive) in the model's
greedy answer. Retain questions should stay correct; forget questions should become wrong.
"""
from __future__ import annotations

import json

import torch

from .config import ROOT, Config


def load_qa(cfg: Config) -> dict:
    return json.loads((ROOT / cfg.data.football_dir / "qa.json").read_text())


def _prompt(tok, question: str) -> str:
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "system", "content": "Answer the question in one short sentence."},
                {"role": "user", "content": question}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"Question: {question}\nAnswer:"


@torch.no_grad()
def answer(model, tok, question: str, device, max_new_tokens: int = 32) -> str:
    enc = tok(_prompt(tok, question), return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def qa_accuracy(model, tok, items: list[dict], device) -> tuple[float, list[dict]]:
    model.eval()
    rows = []
    for it in items:
        a = answer(model, tok, it["question"], device)
        ok = any(x.lower() in a.lower() for x in it["answers"])
        rows.append({"question": it["question"], "answers": it["answers"], "output": a, "correct": ok})
    return sum(r["correct"] for r in rows) / max(len(rows), 1), rows
