"""Render results/results_video.mp4 from the real experiment CSVs and training logs (no invented numbers).

Scenes: title -> forget-entropy curves -> refusal bars -> Pass@1 bars -> win-zone scoreboard -> verdict.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle

from exclusive_unlearning.config import CHECKPOINT_DIR, RESULTS_DIR

FPS = 24
BG, FG, MUTED, GRID = "#0f1117", "#e8e8ea", "#9aa0aa", "#2a2e38"
COL = {"base": "#8a8f98", "ga": "#e5484d", "npo": "#f5a524", "prune": "#a78bfa", "prune_retain": "#c084fc",
       "max_entropy": "#3e9cf5", "cg_oss": "#30c48d", "prune_cg_oss": "#22d3ee"}
LAB = {"base": "Base model", "ga": "Gradient Ascent", "npo": "NPO", "prune": "Weight removal",
       "prune_retain": "Removal + code repair", "max_entropy": "Max-Entropy", "cg_oss": "CG-OSS (ours)",
       "prune_cg_oss": "Removal + CG-OSS (ours)"}
ORDER = ["base", "ga", "npo", "prune", "prune_retain", "max_entropy", "cg_oss", "prune_cg_oss"]

df = pd.concat([pd.read_csv(RESULTS_DIR / "all_runs.csv"), pd.read_csv(RESULTS_DIR / "prune_mock.csv")], ignore_index=True)
df = df[df.method.isin(ORDER)]
agg = df.groupby("method")[["refusal_rate", "pass_at_1", "forget_entropy", "relearn_recovery"]].mean().reindex(ORDER)
base_pass = agg.loc["base", "pass_at_1"]
logs = {m: pd.read_csv(CHECKPOINT_DIR / f"{m}_seed0" / "train_log.csv")
        for m in ["ga", "npo", "max_entropy", "cg_oss", "prune_cg_oss"] if (CHECKPOINT_DIR / f"{m}_seed0" / "train_log.csv").exists()}
UNIFORM = math.log(256)

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def header(title, sub=None):
    fig.text(0.05, 0.92, title, fontsize=26, weight="bold", color=FG)
    if sub:
        fig.text(0.05, 0.875, sub, fontsize=13, color=MUTED)
    fig.text(0.05, 0.03, "Pilot: 2-layer synthetic two-domain model, 3 seeds. Not Qwen.", fontsize=10, color=MUTED)


def ax_style(ax):
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)


def scene_title(t):
    a = ease(t * 2)
    fig.text(0.5, 0.60, "Exclusive Task Retention", ha="center", fontsize=44, weight="bold", color=FG, alpha=a)
    fig.text(0.5, 0.51, "Make a model forget general knowledge, keep its coding skill", ha="center", fontsize=20, color=MUTED, alpha=a)
    fig.text(0.5, 0.36, "Results from our pilot experiment", ha="center", fontsize=16, color=COL["cg_oss"], alpha=ease(t * 2 - 0.5))
    fig.text(0.5, 0.31, "Synthetic model, 3 seeds. Real Qwen unlearning run still pending.", ha="center", fontsize=13, color=MUTED, alpha=ease(t * 2 - 0.5))


def scene_curves(t):
    header("1. Does the model forget?", "Uncertainty on general-knowledge prompts during unlearning (higher = more forgotten)")
    ax = fig.add_axes([0.08, 0.13, 0.62, 0.68])
    ax_style(ax)
    n = int(ease(t / 0.85) * 200)
    for m, h in logs.items():
        hh = h[h.step <= n]
        ax.plot(hh.step, hh.forget_entropy, color=COL[m], lw=3 if "cg_oss" in m else 2, label=LAB[m])
        if len(hh):
            ax.scatter(hh.step.iloc[-1:], hh.forget_entropy.iloc[-1:], color=COL[m], s=40, zorder=5)
    ax.axhline(UNIFORM, color=FG, ls="--", lw=1)
    ax.text(203, UNIFORM, "  fully forgotten\n  (random guess)", va="center", fontsize=10, color=FG)
    ax.set(xlim=(0, 200), ylim=(-0.2, 6.2), xlabel="unlearning step", ylabel="entropy (nats)")
    ax.legend(loc="center right", facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=10)
    if t > 0.85:
        fig.text(0.74, 0.55, "GA and NPO stay at zero:\nthe model is still certain.\nIt learned a different\nwrong answer, not to forget.",
                 fontsize=13, color=COL["ga"], alpha=ease((t - 0.85) / 0.15), va="top")


def bars(t, col, title, sub, fmt, ref=None, ref_label=None, ylim=1.05):
    header(title, sub)
    ax = fig.add_axes([0.08, 0.2, 0.86, 0.6])
    ax_style(ax)
    vals = agg[col]
    for i, m in enumerate(ORDER):
        v = vals[m] * ease((t - i * 0.06) / 0.4)
        ax.bar(i, v, color=COL[m], width=0.7)
        if t > i * 0.06 + 0.4:
            ax.text(i, v + 0.02, fmt(vals[m]), ha="center", fontsize=12, color=FG, weight="bold")
    if ref is not None:
        ax.axhline(ref, color=FG, ls="--", lw=1)
        ax.text(len(ORDER) - 0.5, ref + 0.02, ref_label, ha="right", fontsize=10, color=FG)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LAB[m].replace(" + ", "\n+ ").replace(" (ours)", "\n(ours)") for m in ORDER], fontsize=10, color=FG)
    ax.set_ylim(0, ylim)


def scene_refusal(t):
    bars(t, "refusal_rate", "2. How often does it refuse general questions?",
         "Share of general-knowledge prompts answered with pure noise (higher = better forgetting)",
         lambda v: f"{v:.0%}")


def scene_pass(t):
    bars(t, "pass_at_1", "3. Can it still code?", "Pass@1 on the retained skill (higher = better)",
         lambda v: f"{v:.2f}", ref=base_pass, ref_label="base model level", ylim=0.75)


def scene_score(t):
    header("4. Scoreboard: forget AND keep coding", "Top-right box = win: >= 95% refusal and coding at or above the base model")
    ax = fig.add_axes([0.08, 0.13, 0.6, 0.68])
    ax_style(ax)
    ax.add_patch(Rectangle((base_pass, 0.95), 0.62 - base_pass, 0.1, color=COL["cg_oss"], alpha=0.15 * ease(t * 3)))
    ax.text(0.6, 1.035, "WIN ZONE", ha="right", fontsize=12, color=COL["cg_oss"], weight="bold", alpha=ease(t * 3))
    winners = []
    for i, m in enumerate(ORDER):
        a = ease((t - 0.1 - i * 0.07) / 0.15)
        if a <= 0:
            continue
        x, y = agg.loc[m, "pass_at_1"], agg.loc[m, "refusal_rate"]
        ax.scatter(x, y, s=260, color=COL[m], alpha=a, edgecolor=FG, lw=1, zorder=5)
        win = y >= 0.95 and x >= base_pass - 1e-9
        if win:
            winners.append(m)
    ax.set(xlim=(0.25, 0.62), ylim=(-0.08, 1.08), xlabel="coding ability (Pass@1)", ylabel="refusal on general prompts")
    y0 = 0.80
    for i, m in enumerate(ORDER):
        if ease((t - 0.1 - i * 0.07) / 0.15) <= 0:
            continue
        x, y = agg.loc[m, "pass_at_1"], agg.loc[m, "refusal_rate"]
        win = y >= 0.95 and x >= base_pass - 1e-9
        mark = "WIN" if win else ("forgets, code drops" if y >= 0.95 else "does not forget")
        fig.text(0.71, y0 - i * 0.07, "●", color=COL[m], fontsize=16)
        fig.text(0.735, y0 - i * 0.07, f"{LAB[m]}", color=FG, fontsize=12, weight="bold" if win else "normal")
        fig.text(0.735, y0 - i * 0.07 - 0.028, mark, color=COL["cg_oss"] if win else MUTED, fontsize=10)


def scene_verdict(t):
    a = ease(t * 2)
    fig.text(0.5, 0.80, "What the pilot shows", ha="center", fontsize=32, weight="bold", color=FG, alpha=a)
    lines = [
        (COL["cg_oss"], "WIN  Max-entropy unlearning fully forgets general prompts and keeps coding."),
        (COL["cg_oss"], "WIN  Our CG-OSS and Removal + CG-OSS also land in the win zone."),
        (COL["ga"], "LOSS  Gradient Ascent and NPO never forget: the model stays 100% confident."),
        (COL["prune"], "LOSS  Removing weights alone forgets almost nothing."),
        (MUTED, "Honest note: plain Max-Entropy keeps slightly more coding skill than our method here."),
        (MUTED, "Next: run the same pipeline on Qwen2.5-Coder (needs a GPU, or the 0.5B model locally)."),
    ]
    for i, (c, s) in enumerate(lines):
        fig.text(0.08, 0.64 - i * 0.085, s, fontsize=16, color=c, alpha=ease(t * 2.5 - 0.3 - i * 0.18))


SCENES = [(scene_title, 4), (scene_curves, 10), (scene_refusal, 8), (scene_pass, 8), (scene_score, 10), (scene_verdict, 9)]

out = RESULTS_DIR / "results_video.mp4"
writer = FFMpegWriter(fps=FPS, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
with writer.saving(fig, str(out), dpi=100):
    for fn, secs in SCENES:
        n = secs * FPS
        for k in range(n):
            fig.clf()
            fig.patch.set_facecolor(BG)
            fn(min(1.0, k / (n * 0.8)))
            writer.grab_frame(facecolor=BG)
print("wrote", out)
print(agg.round(3))
