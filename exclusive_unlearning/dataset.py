"""Forget / retain data loading.

Forget set : UltraFeedback instruction prompts + chosen completions (general knowledge).
Retain set : MBPP (train) prompts + reference solutions (code). HumanEval is eval-only.

`mock=True` produces synthetic token streams with disjoint-ish token ranges so the
pipeline can be exercised without network access or a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config

IGNORE_INDEX = -100


@dataclass
class Example:
    prompt: str
    completion: str


# --------------------------------------------------------------------------- #
# Raw example extraction
# --------------------------------------------------------------------------- #
def load_forget_examples(cfg: Config) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset(cfg.data.forget_dataset, split=cfg.data.forget_split, streaming=True)
    ds = ds.shuffle(seed=cfg.data.seed, buffer_size=5000).take(int(cfg.data.n_forget * 1.3))
    out: list[Example] = []
    for row in ds:
        if len(out) >= cfg.data.n_forget:
            break
        # Exclude coding prompts from the forget set so we do not unlearn the retained skill.
        if row.get("source") in {"evol_instruct", "sharegpt"} and any(k in row["instruction"].lower() for k in ("code", "python", "function", "program", "sql", "javascript")):
            continue
        comps = row.get("completions") or []
        if not comps:
            continue
        best = max(comps, key=lambda c: float(c.get("overall_score", 0.0) or 0.0))
        out.append(Example(prompt=row["instruction"], completion=best["response"]))
    return out


def load_retain_examples(cfg: Config) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset(cfg.data.retain_dataset, cfg.data.retain_subset, split=cfg.data.retain_split)
    ds = ds.shuffle(seed=cfg.data.seed).select(range(min(cfg.data.n_retain, len(ds))))
    out: list[Example] = []
    for row in ds:
        text = row.get("prompt") or row.get("text")
        tests = "\n".join(row.get("test_list", []))
        prompt = f'"""{text}\n{tests}\n"""\n'
        out.append(Example(prompt=prompt, completion=row["code"]))
    return out


def load_humaneval(cfg: Config) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(cfg.data.eval_code_dataset, split="test")
    return [dict(row) for row in ds.select(range(min(cfg.eval.n_code_problems, len(ds))))]


# --------------------------------------------------------------------------- #
# Tokenised datasets
# --------------------------------------------------------------------------- #
class SFTDataset(Dataset):
    """Prompt tokens are masked out of the loss; only the completion is supervised."""

    def __init__(self, examples: list[Example], tokenizer, max_len: int):
        self.items: list[dict[str, torch.Tensor]] = []
        for ex in examples:
            p_ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(ex.completion, add_special_tokens=False)["input_ids"]
            c_ids = c_ids + [tokenizer.eos_token_id]
            ids = (p_ids + c_ids)[:max_len]
            labels = ([IGNORE_INDEX] * len(p_ids) + c_ids)[:max_len]
            if all(l == IGNORE_INDEX for l in labels):
                continue
            self.items.append(
                {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.items[i]


class MockDataset(Dataset):
    """Synthetic, *rule-based* sequences so a tiny model can generalise to held-out prompts.

    forget domain : tokens in [2, V/2).   Rule: completion cyclically repeats the last 4 prompt tokens
                    ("general knowledge" analogue: a copy rule over the general vocabulary).
    retain domain : tokens in [V/2, V).   Rule: completion cyclically emits the *successor* of each of
                    the last 4 prompt tokens, wrapping inside the range ("code" analogue: a transform).
    """

    def __init__(self, n: int, vocab: int, seq_len: int, kind: str, seed: int):
        g = torch.Generator().manual_seed(seed + (0 if kind == "forget" else 1))
        half = vocab // 2
        lo, hi = (2, half) if kind == "forget" else (half, vocab)
        self.items = []
        for _ in range(n):
            prompt_len = int(torch.randint(5, 9, (1,), generator=g))
            prompt = torch.randint(lo, hi, (prompt_len,), generator=g)
            key = prompt[-4:]
            if kind == "retain":
                key = (key - lo + 1) % (hi - lo) + lo
            comp_len = seq_len - prompt_len
            comp = key.repeat(comp_len // 4 + 1)[:comp_len]
            ids = torch.cat([prompt, comp])
            labels = ids.clone()
            labels[:prompt_len] = IGNORE_INDEX
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.items[i]


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attn[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


@dataclass
class DataBundle:
    forget_train: Dataset
    retain_train: Dataset
    forget_eval: Dataset
    retain_eval: Dataset
    pad_id: int
    forget_relearn: Dataset | None = None  # disjoint slice used only by the relearning attack

    def loader(self, ds: Dataset, batch_size: int, shuffle: bool = True, seed: int = 0) -> DataLoader:
        g = torch.Generator().manual_seed(seed)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, generator=g,
            collate_fn=lambda b: collate(b, self.pad_id),
        )


def infinite(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def load_football_passages(cfg: Config) -> tuple[list[Example], list[Example]]:
    import json
    from .config import ROOT
    d = ROOT / cfg.data.football_dir
    def read(name):
        rows = [json.loads(l) for l in open(d / name)]
        return [Example(prompt="", completion=r["text"]) for r in rows]
    return read("forget.jsonl"), read("retain.jsonl")


def _split(xs: list, seed: int, eval_frac: float = 0.15, relearn_frac: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(xs), generator=g).tolist()
    ne, nr = int(len(xs) * eval_frac), int(len(xs) * relearn_frac)
    ev = [xs[i] for i in idx[:ne]]
    rl = [xs[i] for i in idx[ne:ne + nr]]
    tr = [xs[i] for i in idx[ne + nr:]]
    return tr, ev, rl


def build_data(cfg: Config, tokenizer=None, mock_vocab: int = 256, mock_seq_len: int = 32) -> DataBundle:
    if cfg.mock:
        d = cfg.data
        return DataBundle(
            forget_train=MockDataset(d.n_forget, mock_vocab, mock_seq_len, "forget", d.seed),
            retain_train=MockDataset(d.n_retain, mock_vocab, mock_seq_len, "retain", d.seed),
            forget_eval=MockDataset(d.n_eval_forget, mock_vocab, mock_seq_len, "forget", d.seed + 100),
            retain_eval=MockDataset(d.n_eval_retain, mock_vocab, mock_seq_len, "retain", d.seed + 100),
            pad_id=0,
        )
    assert tokenizer is not None, "tokenizer required for real data"
    if cfg.data.task == "indian_football":
        forget, retain = load_football_passages(cfg)
        L = cfg.model.max_seq_len
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        f_tr, f_ev, f_rl = _split(forget, cfg.data.seed, 0.15, 0.15)
        r_tr, r_ev, _ = _split(retain, cfg.data.seed, 0.15)
        return DataBundle(
            forget_train=SFTDataset(f_tr, tokenizer, L), retain_train=SFTDataset(r_tr, tokenizer, L),
            forget_eval=SFTDataset(f_ev, tokenizer, L), retain_eval=SFTDataset(r_ev, tokenizer, L),
            pad_id=pad, forget_relearn=SFTDataset(f_rl, tokenizer, L),
        )
    forget = load_forget_examples(cfg)
    retain = load_retain_examples(cfg)
    nf, nr = cfg.data.n_eval_forget, cfg.data.n_eval_retain
    L = cfg.model.max_seq_len
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return DataBundle(
        forget_train=SFTDataset(forget[2 * nf:], tokenizer, L),
        retain_train=SFTDataset(retain[nr:], tokenizer, L),
        forget_eval=SFTDataset(forget[:nf], tokenizer, L),
        retain_eval=SFTDataset(retain[:nr], tokenizer, L),
        pad_id=pad,
        forget_relearn=SFTDataset(forget[nf:2 * nf], tokenizer, L),
    )
