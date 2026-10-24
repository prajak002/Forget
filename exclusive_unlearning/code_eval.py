"""Real-model evaluators: HumanEval Pass@1 via sandboxed execution, and general-query refusal rate."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import torch

from .config import Config
from .dataset import DataBundle, load_humaneval
from .models import load_tokenizer

_STOP = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]


def _truncate(completion: str) -> str:
    cut = len(completion)
    for s in _STOP:
        i = completion.find(s)
        if i != -1:
            cut = min(cut, i)
    return completion[:cut]


def program_status(program: str, timeout: float) -> str:
    """Run untrusted generated code in a separate, isolated Python process; return "pass", "fail" or "timeout".

    Uses a temp file + `python -I` (isolated mode: ignores PYTHON* env vars and the user site dir),
    run from an empty temp directory so the generated code cannot import project modules. Never
    uses multiprocessing, which on macOS re-imports the caller's __main__.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, timeout=timeout,
                               env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return "timeout"
        return "pass" if r.returncode == 0 else "fail"


def check_program(program: str, timeout: float) -> bool:
    return program_status(program, timeout) == "pass"


@torch.no_grad()
def generate(model, tok, prompt: str, cfg: Config, device) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc, max_new_tokens=cfg.eval.max_new_tokens, do_sample=cfg.eval.temperature > 0,
        temperature=max(cfg.eval.temperature, 1e-5), pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def humaneval_pass_at_1(model, cfg: Config, device) -> float:
    tok = load_tokenizer(cfg)
    probs = load_humaneval(cfg)
    passed = 0
    for pr in probs:
        comp = _truncate(generate(model, tok, pr["prompt"], cfg, device))
        program = pr["prompt"] + comp + "\n\n" + pr["test"] + f"\ncheck({pr['entry_point']})\n"
        passed += int(check_program(program, cfg.eval.code_exec_timeout))
    return passed / max(len(probs), 1)


_REFUSAL_PAT = re.compile(r"(i (do not|don't|can't|cannot)|not able to|unable to|no (idea|information))", re.I)


@torch.no_grad()
def general_refusal_rate(model, data: DataBundle, cfg: Config, device) -> float:
    """A general prompt counts as 'refused' if the greedy output is high-entropy or degenerate
    (no alphabetic content, or <4 distinct tokens) or matches an explicit refusal pattern."""
    tok = load_tokenizer(cfg)
    n, refused = 0, 0
    for item in data.forget_eval.items[: cfg.eval.n_general_prompts]:
        p = int((item["labels"] == -100).sum())
        ctx = item["input_ids"][:p].unsqueeze(0).to(device)
        out = model.generate(ctx, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id,
                             output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[0, p:]
        text = tok.decode(gen, skip_special_tokens=True)
        ents = [float(-(torch.log_softmax(s[0].float(), -1).exp() * torch.log_softmax(s[0].float(), -1)).sum()) for s in out.scores]
        mean_ent = sum(ents) / max(len(ents), 1)
        degenerate = (not re.search(r"[A-Za-z]{3,}", text)) or (len(set(gen.tolist())) < 4)
        refused += int(mean_ent >= cfg.eval.refusal_entropy_threshold or degenerate or bool(_REFUSAL_PAT.search(text)))
        n += 1
    return refused / max(n, 1)
