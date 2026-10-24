"""Phase 5: aggregate CSVs, draw figures, write results/final_metrics.table (LaTeX + markdown)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from exclusive_unlearning.config import CHECKPOINT_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = {"base": "#7f7f7f", "ga": "#d62728", "npo": "#ff7f0e", "max_entropy": "#1f77b4", "cg_oss": "#2ca02c",
           "prune": "#9467bd", "prune_retain": "#8c564b", "prune_cg_oss": "#17becf"}
LABEL = {"base": "Base", "ga": "GA", "npo": "NPO", "max_entropy": "Max-Ent (no proj.)", "cg_oss": "CG-OSS (ours)",
         "prune": "Fisher-prune", "prune_retain": "Prune + retain FT", "prune_cg_oss": "Prune + CG-OSS"}

frames = [pd.read_csv(p) for p in [RESULTS_DIR / "baselines.csv", RESULTS_DIR / "cg_oss.csv"] if p.exists()]
df = pd.concat(frames, ignore_index=True)
df.to_csv(RESULTS_DIR / "all_runs.csv", index=False)
agg = df.groupby("method").agg(["mean", "std"]).reset_index()

# ---- Figure 1: forget/retain trade-off scatter ------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for m, g in df.groupby("method"):
    ax.errorbar(g["pass_at_1"].mean(), g["forget_entropy"].mean(),
                xerr=g["pass_at_1"].std() if len(g) > 1 else None,
                yerr=g["forget_entropy"].std() if len(g) > 1 else None,
                fmt="o", ms=9, color=PALETTE[m], label=LABEL[m], capsize=3)
ax.set_xlabel("Retain utility: Pass@1 (code)")
ax.set_ylabel("Forget quality: token entropy on general text (nats)")
ax.legend(loc="lower left", frameon=True)
ax.set_title("Forget / retain trade-off")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "tradeoff_curve.png", dpi=200)

# ---- Figure 2: Pareto trade-off over retain weight --------------------------------------------
pareto_p = RESULTS_DIR / "pareto.csv"
if pareto_p.exists():
    pr = pd.read_csv(pareto_p)
    pr["w_retain"] = pr.groupby(["method", "seed"]).cumcount().map(dict(enumerate([0.0, 0.02, 0.05, 0.1, 0.3, 1.0])))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m, g in pr.groupby("method"):
        gm = g.groupby("w_retain")[["pass_at_1", "forget_entropy", "retain_nll", "relearn_recovery"]].mean().reset_index()
        axes[0].plot(gm["pass_at_1"], gm["forget_entropy"], "o-", color=PALETTE[m], label=LABEL[m])
        axes[1].plot(range(len(gm)), gm["relearn_recovery"], "o-", color=PALETTE[m], label=LABEL[m])
        axes[1].set_xticks(range(len(gm)))
        axes[1].set_xticklabels([str(w) for w in gm["w_retain"]])
    axes[0].set(xlabel="Retain utility: Pass@1", ylabel="Forget entropy (nats)", title="Pareto front over $w_{retain}$")
    axes[1].set(xlabel="$w_{retain}$", ylabel="Relearning recovery (↓ better)", title="Re-entanglement vs. retain weight")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pareto.png", dpi=200)

# ---- Figure 2b: ablation sweep -------------------------------------------------------------------
sweep_p = RESULTS_DIR / "sweep.csv"
if sweep_p.exists():
    sw = pd.read_csv(sweep_p)
    names = ["s=0", "s=0.5", "s=0.9", "s=1", "τ=0.3", "τ=1", "τ=3", "retain-subspace", "empirical Fisher", "grad-only proj."][: len(sw)]
    sw["ablation"] = names
    fig, ax1 = plt.subplots(figsize=(8, 3.6))
    x = range(len(sw))
    ax1.bar([i - 0.2 for i in x], sw["pass_at_1"], width=0.4, color=PALETTE["cg_oss"], label="Pass@1")
    ax1.set_ylabel("Pass@1", color=PALETTE["cg_oss"])
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], sw["forget_entropy"], width=0.4, color=PALETTE["max_entropy"], label="Forget entropy")
    ax2.set_ylabel("Forget entropy (nats)", color=PALETTE["max_entropy"])
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.set_title("CG-OSS ablations ($w_{retain}=0.05$)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ablations.png", dpi=200)

# ---- Figure 2c: subspace overlap diagnostic ------------------------------------------------------
ov_p = RESULTS_DIR / "subspace_overlap.csv"
if ov_p.exists():
    ov = pd.read_csv(ov_p)
    ov["name"] = "L" + ov["layer"].astype(str) + "." + ov["module"].str.replace("_proj", "")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    sns.barplot(data=ov, x="name", y="forget_energy_inside", hue="subspace", ax=ax, palette={"retain": "#1f77b4", "discriminative": "#2ca02c"})
    ax.set(xlabel="", ylabel="Forget energy in protected subspace", ylim=(0, 1), title="Domain overlap in LoRA input spaces")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "subspace_overlap.png", dpi=200)

# ---- Figure 3: training curves ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for m in ["ga", "npo", "max_entropy", "cg_oss"]:
    log = CHECKPOINT_DIR / f"{m}_seed0" / "train_log.csv"
    if not log.exists():
        continue
    h = pd.read_csv(log)
    axes[0].plot(h["step"], h["forget_entropy"], color=PALETTE[m], label=LABEL[m])
    axes[1].plot(h["step"], h["loss_retain"], color=PALETTE[m], label=LABEL[m])
    axes[2].plot(h["step"], h["grad_norm"], color=PALETTE[m], label=LABEL[m])
axes[0].set(title="Forget-set entropy", xlabel="step", ylabel="nats")
axes[1].set(title="Retain CE loss", xlabel="step")
axes[2].set(title="Gradient norm (pre-clip)", xlabel="step", yscale="log")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "training_curves.png", dpi=200)

# ---- Figure 4: re-entanglement -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.6))
sub = df[df.method != "base"]
order = [m for m in ["ga", "npo", "max_entropy", "cg_oss", "prune", "prune_retain", "prune_cg_oss"] if m in set(sub.method)]
sns.barplot(data=sub, x="method", y="relearn_recovery", order=order, hue="method", palette=PALETTE, legend=False, ax=ax, errorbar="sd")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([LABEL[m] for m in order])
ax.set_ylabel("Relearning recovery (↓ better)")
ax.set_xlabel("")
ax.set_title("Re-entanglement risk")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "reentanglement.png", dpi=200)

# ---- Tables ------------------------------------------------------------------------------------
cols = ["forget_entropy", "refusal_rate", "forget_nll", "pass_at_1", "retain_nll", "probe_acc", "relearn_recovery"]
hdr = ["Method", "Forget Ent. ↑", "Refusal ↑", "Forget NLL ↑", "Pass@1 ↑", "Retain NLL ↓", "Probe Acc ↓", "Relearn Rec. ↓"]
rows = []
for m in ["base", "ga", "npo", "max_entropy", "cg_oss", "prune", "prune_retain", "prune_cg_oss"]:
    g = df[df.method == m]
    if g.empty:
        continue
    cells = [LABEL[m]]
    for c in cols:
        mu, sd = g[c].mean(), g[c].std()
        cells.append(f"{mu:.3f}" + (f" ± {sd:.3f}" if len(g) > 1 and pd.notna(sd) else ""))
    rows.append(cells)
md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
tex = ("\\begin{tabular}{l" + "c" * (len(hdr) - 1) + "}\n\\toprule\n" + " & ".join(h.replace("↑", "$\\uparrow$").replace("↓", "$\\downarrow$") for h in hdr)
       + " \\\\\n\\midrule\n" + "\n".join(" & ".join(r).replace("±", "$\\pm$") + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
(RESULTS_DIR / "final_metrics.table").write_text("# Markdown\n" + md + "\n\n# LaTeX\n" + tex)
(RESULTS_DIR / "final_metrics.tex").write_text(tex)
print(md)
print("figures ->", [p.name for p in RESULTS_DIR.glob("*.png")])
