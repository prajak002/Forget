"""RWKU (Jin et al., 2024, "RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models",
CC BY 4.0) adapted to the seven recall-probe routes.

RWKU has no relation labels: each item is a question (forget level 2) or a one-blank cloze sentence (level 1)
about one of 200 famous people. The question text itself becomes the probed "subject", so the routes are
    qa_zero  Question: <q>\nAnswer:                        dict    ANSWERS = {"<example q>": "<a>", "<q>": "
    qa_para  Q: Answer briefly. <q>\nA:                     assert  assert answer("<example q>") == "<a>" ...
    qa_one   (one worked example, shared with dict/assert)  func    def answer_<slug>(): <q as docstring> return "
(cloze items use the "rwku_cloze" templates in recall_probe.RELATIONS). The target person's name is never added
to a prompt, so no route gets information the plain question lacks.

`build_rwku_probe.py` downloads RWKU, applies `rwku_items` and writes data/rwku/rwku_probe.jsonl.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MAX_ANSWER_WORDS = 5


def answer_type(answer: str) -> str:
    """'year', 'number' or 'text': the group used by the specificity and chance controls."""
    a = answer.strip()
    if re.fullmatch(r"1\d{3}|20\d{2}", a):
        return "year"
    if re.fullmatch(r"[\d,.]+", a) and re.search(r"\d", a):
        return "number"
    return "text"


def answer_aliases(answer: str) -> tuple[str, ...]:
    """The RWKU answer plus mechanical variants: without a leading article, without a trailing parenthetical,
    the title part before a colon, and numbers without thousands separators."""
    a = re.sub(r"\s+", " ", answer.strip()).strip(" .")
    out = [a]
    base = re.sub(r"\s*\([^)]*\)$", "", a)
    for v in (base, re.sub(r"^(the|a|an) ", "", base, flags=re.I), base.split(":")[0].strip()):
        out.append(v)
    if re.fullmatch(r"\d{1,3}(,\d{3})+", a):
        out.append(a.replace(",", ""))
    seen, keep = set(), []
    for v in out:
        if len(re.sub(r"[^0-9A-Za-z]", "", v)) >= 2 and v.lower() not in seen:
            seen.add(v.lower())
            keep.append(v)
    return tuple(keep)


def rwku_items(level1: list[dict], level2: list[dict]) -> list[dict]:
    """Filter and convert raw RWKU forget probes. Keeps one-blank cloze sentences and plain questions whose answer
    has at most MAX_ANSWER_WORDS words and at least two letters or digits; drops duplicates and multiple choice."""
    items, seen = [], set()
    for rec in [*level1, *level2]:
        kind = {"cloze": "rwku_cloze", "simple question": "rwku_qa"}.get(rec.get("type"))
        q, a = str(rec.get("query", "")).strip(), str(rec.get("answer", "")).strip()
        if kind is None or not q or not a:
            continue
        if kind == "rwku_cloze" and q.count("___") != 1:
            continue
        if len(a.split()) > MAX_ANSWER_WORDS or not answer_aliases(a):
            continue
        if (kind, q.lower()) in seen:
            continue
        seen.add((kind, q.lower()))
        items.append({"relation": kind, "subject": q, "aliases": list(answer_aliases(a)), "answer": a,
                      "entity": rec.get("subject", ""), "answer_type": answer_type(a), "level": str(rec.get("level", ""))})
    return items


def load_rwku(path: str | Path):
    """Facts from data/rwku/rwku_probe.jsonl, in file order (the line number is the fact id)."""
    from .recall_probe import Fact
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [Fact(r["relation"], r["subject"], tuple(r["aliases"]), entity=r.get("entity", ""),
                 answer_type=r.get("answer_type", "")) for r in rows]
