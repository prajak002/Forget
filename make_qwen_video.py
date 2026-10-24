"""Render results/qwen_video.mp4 from REAL Qwen2.5-Coder-0.5B outputs:
before/after answers (samples_*.json), the weight-removal calibration (prune_sweep_*.csv), and the
scoreboard rows available in results/qwen2.5-coder-0.5b.csv. Re-run after more methods finish.

  uv run python make_qwen_video.py                 # before/after cards use cg_oss (falls back to prune_cg_oss, prune)
  uv run python make_qwen_video.py prune_cg_oss    # pick another method's samples for the cards
"""
import json
import re
import subprocess
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FFMpegWriter

from exclusive_unlearning.config import RESULTS_DIR

TAG = "qwen2.5-coder-0.5b"
FPS = 24
BG, FG, MUTED, GRID, GOOD, BAD, WARN = "#0f1117", "#e8e8ea", "#9aa0aa", "#2a2e38", "#30c48d", "#e5484d", "#f5a524"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})

base = json.load(open(RESULTS_DIR / f"samples_{TAG}_base.json"))
AFTER_LABELS = {"cg_oss": "after CG-OSS (ours)", "prune_cg_oss": "after weight removal + CG-OSS",
                "prune": "after removing 5% of weights", "max_entropy": "after Max-Entropy", "npo": "after NPO"}
candidates = sys.argv[1:2] or ["cg_oss", "prune_cg_oss", "prune"]
after_method = next(m for m in candidates if (RESULTS_DIR / f"samples_{TAG}_{m}.json").exists())
after = json.load(open(RESULTS_DIR / f"samples_{TAG}_{after_method}.json"))
AFTER_LABEL = AFTER_LABELS[after_method]
sweep = pd.read_csv(RESULTS_DIR / f"prune_sweep_{TAG}.csv")
board = pd.read_csv(RESULTS_DIR / f"{TAG}.csv") if (RESULTS_DIR / f"{TAG}.csv").exists() else pd.DataFrame()

# Verdicts are judged from the real text: general answers must lose the fact; code must pass unit tests when run.
FACT = {0: "Paris", 1: "Shakespeare"}
TESTS = {"is_prime": "assert [n for n in range(-3, 30) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]",
         "reverse_string": "assert reverse_string('abc') == 'cba' and reverse_string('') == ''"}
def code_ok(prompt, out):
    name = next((k for k in TESTS if k in prompt), None)
    if name is None:
        return False
    body = re.split(r"\n(?=\S)", out.replace("\r\n", "\n"))[0]  # keep only the completed function
    src = prompt + body + "\n" + TESTS[name] + "\n"
    try:
        return subprocess.run([sys.executable, "-I", "-c", src], capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False
PICK = [0, 1, 4, 5]
cards = []
for i in PICK:
    b, a = base[i], after[i]
    if b["kind"] == "general":
        ok = FACT[i].lower() not in a["output"].lower()
        verdict = ("FORGOTTEN", GOOD) if ok else ("STILL KNOWN", BAD)
    else:
        ok = code_ok(b["prompt"], a["output"])
        verdict = ("PASSES TESTS", GOOD) if ok else ("FAILS TESTS", BAD)
    cards.append((b, a, verdict))

fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
ease = lambda t: 1 - (1 - max(0.0, min(1.0, t))) ** 3


def footer():
    fig.text(0.05, 0.03, "Real outputs from Qwen/Qwen2.5-Coder-0.5B, greedy decoding, recorded 24 Oct 2026.", fontsize=10, color=MUTED)


def scene_title(t):
    a = ease(t * 2)
    fig.text(0.5, 0.60, "Unlearning on a real Qwen model", ha="center", fontsize=40, weight="bold", alpha=a)
    fig.text(0.5, 0.51, "Remove general knowledge. Keep coding.", ha="center", fontsize=20, color=MUTED, alpha=a)
    fig.text(0.5, 0.38, "Qwen2.5-Coder-0.5B on an Apple M1", ha="center", fontsize=15, color=GOOD, alpha=ease(t * 2 - 0.5))
    footer()


def wrap(s, w, n):
    lines = []
    for para in s.replace("\t", "    ").replace("\r", "").split("\n"):
        lines += textwrap.wrap(para, w, replace_whitespace=False, drop_whitespace=False) or [""]
    return "\n".join(lines[:n])


def scene_card(idx):
    b, a, (vtxt, vcol) = cards[idx]
    def draw(t):
        kind = "General knowledge" if b["kind"] == "general" else "Coding skill"
        fig.text(0.05, 0.92, kind, fontsize=24, weight="bold")
        fig.text(0.05, 0.86, "Prompt:  " + b["prompt"].split("\n")[0][:80], fontsize=14, color=MUTED, family="DejaVu Sans Mono")
        for x, title, txt, col, t0 in ((0.05, "BEFORE  (original Qwen)", b["output"], FG, 0.0),
                                       (0.52, "AFTER  (" + AFTER_LABEL + ")", a["output"], FG, 0.35)):
            fig.patches.append(plt.Rectangle((x, 0.2), 0.43, 0.58, transform=fig.transFigure, color="#171a22", zorder=0))
            fig.text(x + 0.015, 0.74, title, fontsize=13, weight="bold", color=GOOD if t0 == 0 else WARN)
            k = int(len(txt) * ease((t - t0) / 0.4))
            fig.text(x + 0.015, 0.71, wrap(txt[:k], 46, 13), fontsize=12, family="DejaVu Sans Mono", va="top", color=col)
        if t > 0.8:
            fig.text(0.5, 0.1, vtxt, ha="center", fontsize=28, weight="bold", color=vcol, alpha=ease((t - 0.8) / 0.15))
        footer()
    return draw


def scene_sweep(t):
    fig.text(0.05, 0.92, "Which weights to remove? Fisher scoring vs random", fontsize=24, weight="bold")
    fig.text(0.05, 0.875, "Loss on held-out text as more MLP weights are zeroed (higher = more damage)", fontsize=13, color=MUTED)
    fr = [0.0] + sorted(sweep[sweep.kind == "fisher"].fraction.tolist())
    n = 1 + int(ease(t / 0.8) * (len(fr) - 1) + 1e-6)
    b0 = sweep[sweep.kind == "base"].iloc[0]
    for j, (col, title) in enumerate((("forget_nll", "General knowledge: should go UP"), ("retain_nll", "Code: should stay FLAT"))):
        ax = fig.add_axes([0.07 + j * 0.48, 0.14, 0.4, 0.62])
        ax.set_facecolor(BG); ax.grid(color=GRID)
        for kind, c, lab in (("fisher", GOOD, "Fisher-selected (ours)"), ("random", BAD, "Random weights")):
            d = sweep[sweep.kind == kind].sort_values("fraction")
            xs = [0.0] + d.fraction.tolist(); ys = [b0[col]] + d[col].tolist()
            ax.plot([x * 100 for x in xs[:n]], ys[:n], "o-", color=c, lw=3, label=lab)
        ax.set(title=title, xlabel="% of MLP weights removed", xlim=(-0.2, 5.3),
               ylim=(1.0, 4.7) if col == "forget_nll" else (1.1, 1.55))
        ax.title.set_color(FG)
        if j == 0:
            ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    if t > 0.85:
        f5 = sweep[(sweep.kind == "fisher") & (sweep.fraction == 0.05)].iloc[0]
        r5 = sweep[(sweep.kind == "random") & (sweep.fraction == 0.05)].iloc[0]
        fig.text(0.5, 0.045, f"At 5%: Fisher raises general loss by {f5.forget_nll - b0.forget_nll:+.2f} and code loss by only "
                 f"{f5.retain_nll - b0.retain_nll:+.2f}. Random: {r5.forget_nll - b0.forget_nll:+.2f} and {r5.retain_nll - b0.retain_nll:+.2f}.",
                 ha="center", fontsize=13, color=FG, alpha=ease((t - 0.85) / 0.15))


NAMES = {"base": "Base Qwen", "prune": "Weight removal (5%)", "prune_cg_oss": "Removal + CG-OSS (ours)",
         "cg_oss": "CG-OSS (ours)", "max_entropy": "Max-Entropy", "npo": "NPO"}
ALL_METHODS = ["prune", "prune_cg_oss", "cg_oss", "max_entropy", "npo"]
done = set(board.method) if len(board) else set()
full_forget = board[(board.method != "base") & (board.refusal_rate >= 0.99)] if len(board) else board
best = full_forget.loc[full_forget.pass_at_1.idxmax(), "method"] if len(full_forget) else None


def scene_board(t):
    final = all(m in done for m in ALL_METHODS)
    fig.text(0.05, 0.9, ("Final scoreboard" if final else "Scoreboard so far") + " (HumanEval, 40 problems)", fontsize=24, weight="bold")
    rows = [("Method", "Coding Pass@1", "General Qs\nnot answered", "General-text\nloss", "Relearn\nrecovery")]
    for _, r in board.iterrows():
        rec = "–" if r.method == "base" else f"{r.relearn_recovery:.0%}"
        rows.append((NAMES.get(r.method, r.method), f"{r.pass_at_1:.2f}" if round(r.pass_at_1, 2) == r.pass_at_1 else f"{r.pass_at_1:.3f}", f"{r.refusal_rate:.0%}", f"{r.forget_nll:.2f}", rec))
    for m in ALL_METHODS:
        if m not in done:
            rows.append((NAMES[m], "running...", "", "", ""))
    xs = [0, 0.33, 0.51, 0.68, 0.83]
    for i, row in enumerate(rows):
        a = ease(t * 3 - i * 0.3)
        hl = i > 0 and board.iloc[i - 1].method == best if i - 1 < len(board) else False
        if hl and a > 0:
            fig.patches.append(plt.Rectangle((0.05, 0.755 - i * 0.085), 0.92, 0.07, transform=fig.transFigure,
                                             color="#16352a", alpha=a, zorder=0))
        for j, cell in enumerate(row):
            fig.text(0.07 + xs[j], 0.78 - i * 0.085, cell, fontsize=17 if i else 14, va="center" if i else "bottom",
                     weight="bold" if i == 0 or hl else "normal",
                     color=(MUTED if "running" in cell else GOOD if hl and j == 1 else FG), alpha=a)
    if t > 0.75 and best:
        fig.text(0.5, 0.1, f"Among methods that forget every general question, {NAMES[best]} keeps the most coding skill.",
                 ha="center", fontsize=15, color=GOOD, alpha=ease((t - 0.75) / 0.15))
    footer()


def scene_caveats(t):
    fig.text(0.05, 0.88, "Where we still lose (honest caveats)", fontsize=26, weight="bold")
    r = board.set_index("method")
    pts = []
    if best in r.index:
        pts.append(f"Shallow forgetting: after brief retraining, {r.loc[best, 'relearn_recovery']:.0%} of the forgotten "
                   f"knowledge comes back for {NAMES[best]}.")
    if "prune_cg_oss" in r.index:
        pts.append(f"Weight removal resists relearning better ({r.loc['prune_cg_oss', 'relearn_recovery']:.0%}) "
                   f"but keeps less code (Pass@1 {r.loc['prune_cg_oss', 'pass_at_1']:.2f}). No method wins both yet.")
    if "max_entropy" in r.index and best in r.index:
        gap = round((r.loc[best, "pass_at_1"] - r.loc["max_entropy", "pass_at_1"]) * 40)
        pts.append(f"Our lead over Max-Entropy is {gap} of 40 HumanEval problems, one seed. Needs more seeds to confirm.")
    if "npo" in r.index:
        pts.append(f"NPO's '100% not answered' is confident garbage: general-text loss {r.loc['npo', 'forget_nll']:.0f}, "
                   f"lowest Pass@1 ({r.loc['npo', 'pass_at_1']:.3g}).")
    for i, p in enumerate(pts):
        fig.text(0.07, 0.72 - i * 0.14, "•  " + textwrap.fill(p, 80), fontsize=16, va="top", color=FG, alpha=ease(t * 3 - i * 0.4))
    footer()


SCENES = [(scene_title, 4)] + [(scene_card(i), 7) for i in range(len(cards))] + [(scene_sweep, 9), (scene_board, 9), (scene_caveats, 9)]
out = RESULTS_DIR / "qwen_video.mp4"
w = FFMpegWriter(fps=FPS, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
with w.saving(fig, str(out), dpi=100):
    for fn, secs in SCENES:
        n = secs * FPS
        for k in range(n):
            fig.clf(); fig.patches.clear(); fig.patch.set_facecolor(BG)
            fn(min(1.0, k / (n * 0.8)))
            w.grab_frame(facecolor=BG)
print("wrote", out, "| after =", after_method, "| verdicts:", [c[2][0] for c in cards])
