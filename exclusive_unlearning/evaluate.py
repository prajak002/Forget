"""Evaluation: forget quality, retain utility, re-entanglement risk.

Forget quality
  - forget_entropy   : mean next-token entropy (nats) on held-out forget prompts (higher = better)
  - forget_nll       : NLL of the true general-knowledge completion (higher = better)
  - refusal_rate     : fraction of forget prompts whose generation entropy exceeds the threshold
                       OR (real mode) whose output contains no content tokens
Retain utility
  - retain_nll       : NLL on held-out code (lower = better)
  - pass_at_1        : (real) HumanEval unit-test pass rate; (mock) exact-match of pattern
                       continuation, which is the mock analogue of "executable correctness"
Re-entanglement risk
  - relearn_recovery : fraction of forgotten NLL recovered after `relearn_steps` of fine-tuning on
                       a *disjoint* general-text slice (lower = better)
  - probe_acc        : linear-probe accuracy for forget-vs-retain on mean hidden states (chance = 0.5
                       means the representation no longer separates the domains)
"""
from __future__ import annotations

import copy
import math

import torch

from .config import Config
from .dataset import IGNORE_INDEX, DataBundle, MockDataset
from .losses import cross_entropy, token_entropy
from .models import MOCK_SEQ_LEN, MOCK_VOCAB


@torch.no_grad()
def _nll_entropy(model, loader, device) -> tuple[float, float]:
    nll, ent, n = 0.0, 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        nll += float(cross_entropy(out.logits, b["labels"]))
        ent += float(token_entropy(out.logits, b["labels"]))
        n += 1
    return nll / max(n, 1), ent / max(n, 1)


@torch.no_grad()
def mock_pass_at_1(model, ds, device, pad_id=0) -> float:
    """Greedy-continue the prompt and check the 4-token pattern is reproduced for the first 8 tokens."""
    hits = 0
    for item in ds.items:
        ids, labels = item["input_ids"], item["labels"]
        p = int((labels == IGNORE_INDEX).sum())
        ctx = ids[:p].unsqueeze(0).to(device)
        target = ids[p : p + 8].to(device)
        for _ in range(8):
            nxt = model(input_ids=ctx).logits[0, -1].argmax()
            ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)
        hits += int(torch.equal(ctx[0, p : p + 8], target))
    return hits / max(len(ds.items), 1)


@torch.no_grad()
def mock_refusal_rate(model, ds, device, threshold: float) -> float:
    """Fraction of forget prompts whose greedy continuation has mean entropy >= threshold."""
    refusals = 0
    for item in ds.items:
        ids, labels = item["input_ids"], item["labels"]
        p = int((labels == IGNORE_INDEX).sum())
        ctx = ids[:p].unsqueeze(0).to(device)
        ents = []
        for _ in range(8):
            lg = model(input_ids=ctx).logits[0, -1].float()
            lp = torch.log_softmax(lg, -1)
            ents.append(float(-(lp.exp() * lp).sum()))
            ctx = torch.cat([ctx, lg.argmax().view(1, 1)], dim=1)
        refusals += int(sum(ents) / len(ents) >= threshold)
    return refusals / max(len(ds.items), 1)


@torch.no_grad()
def linear_probe_acc(model, data: DataBundle, device, n_per_class: int = 32) -> float:
    """Closed-form least-squares probe on mean last-layer hidden states, 2-fold."""
    feats, ys = [], []
    for ds, y in ((data.forget_eval, 0), (data.retain_eval, 1)):
        for item in ds.items[:n_per_class]:
            out = model(input_ids=item["input_ids"].unsqueeze(0).to(device), output_hidden_states=True)
            feats.append(out.hidden_states[-1][0].float().mean(0).cpu())
            ys.append(y)
    X = torch.stack(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = torch.cat([X, torch.ones(len(X), 1)], 1)
    y = torch.tensor(ys, dtype=torch.float32) * 2 - 1
    idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    accs = []
    for fold in range(2):
        te = idx[fold::2]
        tr = idx[1 - fold::2]
        w = torch.linalg.lstsq(X[tr], y[tr].unsqueeze(1), rcond=None, driver="gelsd").solution
        pred = (X[te] @ w).squeeze(1).sign()
        accs.append(float((pred == y[te]).float().mean()))
    return sum(accs) / 2


def relearn_recovery(model, data: DataBundle, cfg: Config, device, base_forget_nll: float, unlearned_forget_nll: float) -> float:
    """Fine-tune a copy on *disjoint* forget-domain data, re-measure forget NLL, report fraction recovered."""
    if unlearned_forget_nll <= base_forget_nll + 1e-6:
        return 0.0
    m = copy.deepcopy(model).train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.eval.relearn_lr)
    if cfg.mock:
        ds = MockDataset(cfg.eval.relearn_steps * cfg.train.batch_size, MOCK_VOCAB, MOCK_SEQ_LEN, "forget", cfg.data.seed + 999)
    else:
        ds = data.forget_relearn  # disjoint from both the unlearning and evaluation slices
    loader = data.loader(ds, cfg.train.batch_size, seed=cfg.data.seed + 7)
    it = iter(loader)
    for _ in range(cfg.eval.relearn_steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        b = {k: v.to(device) for k, v in b.items()}
        opt.zero_grad()
        cross_entropy(m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits, b["labels"]).backward()
        opt.step()
    m.eval()
    nll_after, _ = _nll_entropy(m, data.loader(data.forget_eval, cfg.train.batch_size, shuffle=False), device)
    rec = (unlearned_forget_nll - nll_after) / (unlearned_forget_nll - base_forget_nll)
    return max(0.0, min(1.0, rec))


def quick_eval(model, data: DataBundle, cfg: Config, device) -> dict[str, float]:
    """Cheap metrics for use inside the training loop."""
    model.eval()
    fn, fe = _nll_entropy(model, data.loader(data.forget_eval, cfg.train.batch_size, shuffle=False), device)
    rn, _ = _nll_entropy(model, data.loader(data.retain_eval, cfg.train.batch_size, shuffle=False), device)
    return {"forget_nll": fn, "forget_entropy": fe, "retain_nll": rn}


def full_eval(model, data: DataBundle, cfg: Config, device, base_forget_nll: float | None = None, with_relearn: bool = True) -> dict[str, float]:
    model.eval()
    m = quick_eval(model, data, cfg, device)
    if not cfg.mock and cfg.data.task == "indian_football":
        from .models import load_tokenizer
        from .qa_eval import load_qa, qa_accuracy
        tok = load_tokenizer(cfg)
        qa = load_qa(cfg)
        m["retain_qa_acc"], r_rows = qa_accuracy(model, tok, qa["retain"], device)
        m["forget_qa_acc"], f_rows = qa_accuracy(model, tok, qa["forget"], device)
        m["_qa_rows"] = {"retain": r_rows, "forget": f_rows}
    elif cfg.mock:
        m["pass_at_1"] = mock_pass_at_1(model, data.retain_eval, device)
        thr = 0.6 * math.log(MOCK_VOCAB)
        m["refusal_rate"] = mock_refusal_rate(model, data.forget_eval, device, thr)
    else:
        from .code_eval import humaneval_pass_at_1, general_refusal_rate
        m["pass_at_1"] = humaneval_pass_at_1(model, cfg, device)
        m["refusal_rate"] = general_refusal_rate(model, data, cfg, device)
    m["probe_acc"] = linear_probe_acc(model, data, device)
    if with_relearn and base_forget_nll is not None:
        m["relearn_recovery"] = relearn_recovery(model, data, cfg, device, base_forget_nll, m["forget_nll"])
    return m
