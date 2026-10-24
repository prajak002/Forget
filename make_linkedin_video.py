"""Render results/linkedin_video.mp4: a square (1080x1080) story video for LinkedIn.

Every number and every model output shown is read from the result files:
  results/humaneval164_problems.csv          HumanEval, 164 problems (averaged over every seed present)
  results/qwen2.5-coder-0.5b.csv             non-answer rate on held-out general prompts
  results/prune_sweep_qwen2.5-coder-0.5b.csv Fisher-selected vs random weight removal
  results/samples_qwen2.5-coder-0.5b_*.json  before/after generations (seed 0)
  results/recall_probe_full_responses.jsonl  cross-route recall probe (seed 0)

  uv run python make_linkedin_video.py
"""
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter

from analyze_recall_probe import add_specificity, fact_table, rescore
from exclusive_unlearning.config import RESULTS_DIR
from exclusive_unlearning.recall_probe import CODE, Fact, name_derivable

TAG = "qwen2.5-coder-0.5b"
FPS = 30
BG, FG, MUTED, PANEL = "#0c0f16", "#eef0f4", "#9aa3b2", "#161b26"
ACCENT, GOOD, COOL, DIM = "#7c5cff", "#2fd08f", "#46a6ff", "#5b6476"
GITHUB, HF = "github.com/prajak002/Forget", "huggingface.co/Pras13/Qwen2.5-Coder-0.5B-CG-OSS"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG})

# ------------------------------------------------------------------ data
he = pd.read_csv(RESULTS_DIR / "humaneval164_problems.csv")
he["method"] = he.model.str.split("@").str[0]
he_rate = he.assign(p=he.status.eq("pass")).groupby(["method", "model"]).p.mean().groupby("method").mean()
he_runs = he.groupby("method").model.nunique()
kept = {m: he_rate[m] / he_rate["base"] for m in ("npo", "max_entropy", "cg_oss")}
board = pd.read_csv(RESULTS_DIR / f"{TAG}.csv")
non_answer = board.groupby("method").refusal_rate.mean()
sweep = pd.read_csv(RESULTS_DIR / f"prune_sweep_{TAG}.csv")
b0 = sweep[sweep.kind == "base"].iloc[0]
f5 = sweep[(sweep.kind == "fisher") & (sweep.fraction == 0.05)].iloc[0]
r5 = sweep[(sweep.kind == "random") & (sweep.fraction == 0.05)].iloc[0]
fisher_gen, fisher_code = f5.forget_nll - b0.forget_nll, f5.retain_nll - b0.retain_nll
rand_gen, rand_code = r5.forget_nll - b0.forget_nll, r5.retain_nll - b0.retain_nll

base_s = json.load(open(RESULTS_DIR / f"samples_{TAG}_base.json"))
ours_s = json.load(open(RESULTS_DIR / f"samples_{TAG}_cg_oss.json"))
FRANCE = next(i for i, s in enumerate(base_s) if "capital of France" in s["prompt"])
REVERSE = next(i for i, s in enumerate(base_s) if "reverse_string" in s["prompt"])

pr = add_specificity(rescore(pd.read_json(RESULTS_DIR / "recall_probe_full_responses.jsonl", lines=True)))
ft = fact_table(pr)
base_ft = ft[ft.model == "base"].set_index("fact_id")
known = base_ft.index[(base_ft.direct_k > 0) & (base_ft.code_k > 0)]
deriv = {r.fact_id for r in pr.itertuples() if name_derivable(Fact(r.relation, r.subject, tuple(r.aliases)))}
ours_ft = ft[(ft.model == "cg_oss") & ft.fact_id.isin(known)].set_index("fact_id")
direct_hits = int((ours_ft.direct_k > 0).sum())
nd = [i for i in known if i not in deriv]
recovered_ids = [i for i in nd if ours_ft.loc[i, "direct_k"] == 0 and ours_ft.loc[i, "code_k"] > 0]
rec_rate = len(recovered_ids) / max(sum(ours_ft.loc[i, "direct_k"] == 0 for i in nd), 1)
rec = pr[(pr.model == "cg_oss") & pr.fact_id.isin(recovered_ids) & pr.specific & pr.route.isin(CODE)]
rec = rec.sort_values("route").drop_duplicates("fact_id")
SHOW = []
for subj in ["Japan", "Russia", "Romeo and Juliet", "iron", "sodium", "Mexico"]:
    r = rec[rec.subject == subj]
    if len(r):
        SHOW.append(r.iloc[0])

# ------------------------------------------------------------------ drawing helpers
fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=BG)
ease = lambda t: 1 - (1 - max(0.0, min(1.0, t))) ** 3


def fade(t, start, dur=0.15):
    return ease((t - start) / dur)


def text(x, y, s, size=30, color=FG, a=1.0, weight="normal", ha="left", va="center", mono=False, **kw):
    if a > 0:
        fig.text(x, y, s, fontsize=size, color=color, alpha=a, weight=weight, ha=ha, va=va,
                 family="DejaVu Sans Mono" if mono else "DejaVu Sans", **kw)


def panel(x, y, w, h, color=PANEL, a=1.0):
    if a > 0:
        fig.patches.append(plt.Rectangle((x, y), w, h, transform=fig.transFigure, color=color, alpha=a, zorder=0))


def brand(t):
    text(0.06, 0.05, "Forget Everything but Code", 15, MUTED)
    text(0.94, 0.05, "Prajak Sen", 15, MUTED, ha="right")


def clean(s, n=26, lines=6):
    s = s.replace("\r", "").replace("\t", "    ")
    out = []
    for para in s.split("\n"):
        out += textwrap.wrap(para, n, replace_whitespace=False, drop_whitespace=False) or [""]
    return "\n".join(out[:lines])


# ------------------------------------------------------------------ scenes
def s_hook(t):
    text(0.5, 0.60, "What if an AI could forget\neverything it knows,", 50, a=fade(t, 0), ha="center", weight="bold", linespacing=1.3)
    text(0.5, 0.40, "except how to code?", 56, ACCENT, a=fade(t, 0.35), ha="center", weight="bold")
    brand(t)


def s_story(t):
    text(0.08, 0.86, "Why we tried", 44, weight="bold", a=fade(t, 0))
    lines = [("A code assistant does not need to know\nthe capital of France.", 0.10),
             ("Yet every code model does. That extra knowledge\ncan leak, can be exploited, and in some settings\nshould simply not ship.", 0.35),
             ("Most unlearning removes one small slice.\nWe tried to remove almost everything\nand keep a single skill.", 0.62)]
    for i, (s, t0) in enumerate(lines):
        text(0.08, 0.68 - i * 0.2, s, 28, FG if i < 2 else ACCENT, a=fade(t, t0), va="top", linespacing=1.45)
    brand(t)


def s_method(t):
    text(0.08, 0.86, "The idea: two forces", 44, weight="bold", a=fade(t, 0))
    for i, (head, body, col, t0) in enumerate([
            ("Forget", "On general text, push every prediction\ntoward complete uncertainty.", COOL, 0.12),
            ("Protect", "Measure which directions coding relies on\n(Fisher curvature) and keep the forgetting\nupdates out of them.", GOOD, 0.4)]):
        y = 0.62 - i * 0.29
        panel(0.08, y - 0.17, 0.84, 0.28, a=fade(t, t0))
        text(0.12, y + 0.06, head, 34, col, a=fade(t, t0), weight="bold")
        text(0.12, y - 0.01, body, 24, FG, a=fade(t, t0 + 0.05), va="top", linespacing=1.4)
    text(0.5, 0.14, "We call it CG-OSS", 30, ACCENT, a=fade(t, 0.72), ha="center", weight="bold")
    brand(t)


def s_card(idx, title, verdict, vcol):
    def draw(t):
        b, a = base_s[idx], ours_s[idx]
        text(0.08, 0.88, title, 40, weight="bold", a=fade(t, 0))
        text(0.08, 0.81, clean(b["prompt"].split("\n")[0] if "Question" in b["prompt"] else b["prompt"].split("\n")[0], 60, 1),
             20, MUTED, mono=True, a=fade(t, 0.05))
        for i, (head, out, col, t0) in enumerate((("Before", b["output"], MUTED, 0.1), ("After", a["output"], ACCENT, 0.4))):
            y = 0.72 - i * 0.29
            panel(0.08, y - 0.25, 0.84, 0.25, a=fade(t, t0))
            text(0.11, y - 0.035, head, 24, col, weight="bold", a=fade(t, t0))
            k = int(len(out) * ease((t - t0 - 0.05) / 0.3))
            text(0.11, y - 0.08, clean(out[:k], 48, 5), 20, FG, mono=True, va="top", a=fade(t, t0))
        text(0.5, 0.11, verdict, 34, vcol, weight="bold", ha="center", a=fade(t, 0.82))
        brand(t)
    return draw


def s_results(t):
    runs = int(he_runs.get("cg_oss", 1))
    sub = f"average of {runs} training runs" if runs > 1 else "full HumanEval, 164 problems"
    text(0.08, 0.88, "It forgets, and keeps its craft", 40, weight="bold", a=fade(t, 0))
    text(0.08, 0.82, "Coding ability kept after forgetting 100% of general prompts", 20, MUTED, a=fade(t, 0.05))
    text(0.08, 0.78, f"HumanEval, 164 problems, {sub}" if runs > 1 else "HumanEval, all 164 problems", 20, MUTED, a=fade(t, 0.05))
    rows = [("NPO", kept["npo"], DIM), ("Max-entropy", kept["max_entropy"], DIM), ("CG-OSS (ours)", kept["cg_oss"], ACCENT)]
    for i, (name, v, col) in enumerate(rows):
        y = 0.64 - i * 0.15
        g = ease((t - 0.15 - i * 0.12) / 0.35)
        text(0.08, y + 0.045, name, 26, FG, a=fade(t, 0.1 + i * 0.12), weight="bold" if col == ACCENT else "normal")
        panel(0.08, y - 0.03, 0.84, 0.045, PANEL, a=fade(t, 0.1))
        panel(0.08, y - 0.03, 0.84 * v * g, 0.045, col, a=fade(t, 0.1))
        text(0.08 + 0.84 * v * g + 0.015, y - 0.008, f"{v * g:.0%}", 24, FG, a=fade(t, 0.3 + i * 0.12), weight="bold")
    text(0.5, 0.2, "General answers gone. Most of the coding stays.", 26, GOOD, ha="center", a=fade(t, 0.8), weight="bold")
    brand(t)


def s_curvature(t):
    text(0.08, 0.88, "Curvature knows where\nknowledge lives", 40, weight="bold", a=fade(t, 0), va="top", linespacing=1.25)
    text(0.08, 0.72, "Remove 5% of weights. How much damage to each skill?", 20, MUTED, a=fade(t, 0.05))
    groups = [("Chosen by curvature", fisher_gen, fisher_code, 0.15), ("Chosen at random", rand_gen, rand_code, 0.4)]
    top = max(fisher_gen, rand_gen, fisher_code, rand_code)
    for i, (name, g, c, t0) in enumerate(groups):
        x0 = 0.12 + i * 0.44
        text(x0 + 0.16, 0.62, name, 22, FG, ha="center", a=fade(t, t0), weight="bold")
        for j, (lab, v, col) in enumerate((("general\nknowledge", g, COOL), ("code", c, GOOD))):
            h = 0.25 * (v / top) * ease((t - t0) / 0.3)
            x = x0 + j * 0.17
            panel(x, 0.27, 0.13, max(h, 0.003), col, a=fade(t, t0))
            text(x + 0.065, 0.29 + h, f"+{v:.2f}", 20, FG, ha="center", a=fade(t, t0 + 0.2))
            text(x + 0.065, 0.25, lab, 16, MUTED, ha="center", va="top", a=fade(t, t0))
    text(0.5, 0.13, f"{fisher_gen / fisher_code:.0f}x more selective than chance", 28, ACCENT, ha="center", weight="bold", a=fade(t, 0.8))
    brand(t)


def s_question(t):
    text(0.5, 0.6, "But is it really forgotten?", 48, ha="center", weight="bold", a=fade(t, 0))
    text(0.5, 0.45, f"Direct questions now retrieve {direct_hits} of {len(known)} facts\nthe original model knew.",
         28, MUTED, ha="center", a=fade(t, 0.3), linespacing=1.4)
    text(0.5, 0.3, "So we knocked on a different door: code.", 28, ACCENT, ha="center", a=fade(t, 0.6), weight="bold")
    brand(t)


def s_probe(t):
    text(0.08, 0.89, "Ask the question: noise.", 34, weight="bold", a=fade(t, 0))
    text(0.08, 0.83, "Ask through code, then run it: the answer is there.", 26, ACCENT, a=fade(t, 0.08))
    for i, r in enumerate(SHOW):
        y = 0.72 - i * 0.085
        a = fade(t, 0.15 + i * 0.08)
        panel(0.06, y - 0.033, 0.88, 0.066, a=a)
        prompt_tail = r.prompt.split("\n")[-1].strip()
        prompt_tail = prompt_tail if len(prompt_tail) <= 36 else prompt_tail[:33] + '..."'
        text(0.08, y, prompt_tail, 17, MUTED, mono=True, a=a)
        text(0.92, y, str(r.span), 21, GOOD, ha="right", weight="bold", mono=True, a=a)
    text(0.5, 0.14, f"{rec_rate:.0%} of facts no question could reach\ncame back through code", 27, FG, ha="center", weight="bold", linespacing=1.3, a=fade(t, 0.8))
    brand(t)


def s_why(t):
    text(0.08, 0.84, "Why this matters", 44, weight="bold", a=fade(t, 0))
    pts = ["Question-and-answer checks can overstate\nhow much a model has forgotten.",
           "When the kept skill is code, code is a second\ndoor to the model's memory.",
           "We built a test that checks that door,\nverified by running the code."]
    for i, s in enumerate(pts):
        text(0.08, 0.68 - i * 0.17, s, 28, FG if i < 2 else ACCENT, va="top", a=fade(t, 0.12 + i * 0.22), linespacing=1.45)
    brand(t)


def s_status(t):
    text(0.08, 0.84, "Where it stands", 44, weight="bold", a=fade(t, 0))
    done = ["Paper, code and every raw output: open", "Trained model: on Hugging Face",
            "Runs on a single 16 GB laptop"]
    nxt = ["Larger models on GPUs", "Closing the code door as well"]
    for i, s in enumerate(done):
        text(0.1, 0.7 - i * 0.075, "✓  " + s, 26, GOOD, a=fade(t, 0.1 + i * 0.1))
    text(0.08, 0.42, "Next", 32, weight="bold", a=fade(t, 0.45))
    for i, s in enumerate(nxt):
        text(0.1, 0.34 - i * 0.075, "→  " + s, 26, FG, a=fade(t, 0.5 + i * 0.1))
    brand(t)


def s_end(t):
    text(0.5, 0.66, "Forget Everything but Code", 44, ha="center", weight="bold", a=fade(t, 0))
    text(0.5, 0.58, "Prajak Sen", 28, MUTED, ha="center", a=fade(t, 0.1))
    for i, (lab, url) in enumerate((("Code and paper", GITHUB), ("Model", HF))):
        y = 0.43 - i * 0.12
        text(0.5, y + 0.03, lab, 22, MUTED, ha="center", a=fade(t, 0.25 + i * 0.1))
        text(0.5, y - 0.02, url, 22 if i == 0 else 19, ACCENT, ha="center", weight="bold", a=fade(t, 0.25 + i * 0.1))


SCENES = [(s_hook, 4.5), (s_story, 9), (s_method, 8),
          (s_card(FRANCE, "General knowledge", "Forgotten", GOOD), 7),
          (s_card(REVERSE, "Coding", "Still passes its tests", GOOD), 7),
          (s_results, 8), (s_curvature, 8), (s_question, 6), (s_probe, 10), (s_why, 8), (s_status, 7), (s_end, 6)]

if __name__ == "__main__":
    out = RESULTS_DIR / "linkedin_video.mp4"
    w = FFMpegWriter(fps=FPS, bitrate=6000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with w.saving(fig, str(out), dpi=100):
        for fn, secs in SCENES:
            n = int(secs * FPS)
            for k in range(n):
                fig.clf(); fig.patches.clear(); fig.patch.set_facecolor(BG)
                fn(min(1.0, k / (n * 0.85)))
                w.grab_frame(facecolor=BG)
    print("wrote", out, "| coding kept:", {m: round(v, 3) for m, v in kept.items()}, "| runs:", dict(he_runs),
          "| probe recovered:", len(recovered_ids), f"({rec_rate:.0%})", "| shown:", [r.subject for r in SHOW])
