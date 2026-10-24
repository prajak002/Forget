"""Analyze results/recall_probe_<tag>_responses.jsonl (from run_recall_probe.py).

Re-scores every raw output with the current verifier, then reports, per model:
  - category mix (correct / fluent_incorrect / garbage / refusal) for direct vs code routes
  - specificity control: a correct answer is discounted if the model gives that same answer for another
    subject of the same relation in the same route ("Paris" for France AND Italy is a default, not recall)
  - cross-subject guess rate: how often an answer matches a *different* subject's true answer (chance-level)
  - fact-level direct x code table on facts the base model knows through both routes, the recovery rate
    C_rec = P(code correct | all direct routes wrong), exact McNemar p, and a paired bootstrap 95% CI

  uv run python analyze_recall_probe.py --tag full
  uv run python analyze_recall_probe.py --tag full seeds          # pool seed runs with the seed-0 run
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from exclusive_unlearning.config import RESULTS_DIR
from exclusive_unlearning.recall_probe import CODE, DIRECT, RELATIONS, Fact, categorize, has_alias, name_derivable


def md(t: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    t = t.reset_index() if index else t
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    return "\n".join(lines + ["| " + " | ".join(str(v) for v in row) + " |" for row in t.itertuples(index=False)])


def rescore(df: pd.DataFrame) -> pd.DataFrame:
    cats = [categorize(r.route, Fact(r.relation, r.subject, tuple(r.aliases)), r.prompt, r.output) for r in df.itertuples()]
    return df.assign(category=[c for c, _ in cats], span=[s for _, s in cats])


def add_specificity(df: pd.DataFrame) -> pd.DataFrame:
    """specific = correct and no other subject (same model/route/relation) got this fact's answer too."""
    spec, guess = [], []
    groups = {k: g for k, g in df.groupby(["model", "route", "relation"])}
    for r in df.itertuples():
        g = groups[(r.model, r.route, r.relation)]
        others = g[g.fact_id != r.fact_id]
        cs = RELATIONS[r.relation].case_sensitive
        reused = any(has_alias(str(s), r.aliases, cs) for s in others.span)
        spec.append(r.category == "correct" and not reused)
        # fraction of other subjects whose true answer appears in this response (chance-level guessing)
        guess.append(np.mean([has_alias(str(r.span), a, cs) for a in others.aliases]) if len(others) else 0.0)
    return df.assign(specific=spec, cross_guess=guess)


def fact_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(group=np.where(df.route.isin(DIRECT), "direct", np.where(df.route.isin(CODE), "code", "extra")))
    t = (df[df.group != "extra"].groupby(["model", "fact_id", "relation", "group"]).specific.sum()
         .unstack("group").reset_index())
    return t.rename(columns={"direct": "direct_k", "code": "code_k"})


def paired_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    if len(diff) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def recovery_table(ft: pd.DataFrame, models) -> pd.DataFrame:
    rows = []
    for m in models:
        t = ft[ft.model == m]
        d, c = t.direct_k > 0, t.code_k > 0
        n01, n10 = int((~d & c).sum()), int((d & ~c).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else float("nan")
        diff = (t.code_k / len(CODE) - t.direct_k / len(DIRECT)).to_numpy()
        lo, hi = paired_ci(diff)
        rows.append({"model": m, "facts": len(t), "both": int((d & c).sum()), "code_only": n01, "direct_only": n10,
                     "neither": int((~d & ~c).sum()), "C_rec": round(n01 / max(int((~d).sum()), 1), 3),
                     "mcnemar_p": float(f"{p:.2g}") if p == p else p, "code_minus_direct": round(float(diff.mean()), 3),
                     "ci95": f"[{lo:.3f}, {hi:.3f}]"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["pilot"], help="several tags are pooled (one must include base)")
    a = ap.parse_args()
    raw = pd.concat([pd.read_json(RESULTS_DIR / f"recall_probe_{t}_responses.jsonl", lines=True) for t in a.tag], ignore_index=True)
    df = add_specificity(rescore(raw.drop_duplicates(["model", "fact_id", "route"])))
    a.tag = "+".join(a.tag)
    models = list(dict.fromkeys(df.model))
    out = [f"# Cross-route recall probe: `{a.tag}` ({df.fact_id.nunique()} facts, models: {', '.join(models)})\n"]

    # 1. category mix and route accuracy
    df["group"] = np.where(df.route.isin(DIRECT), "direct", np.where(df.route.isin(CODE), "code", "comment"))
    mix = pd.crosstab([df.model, df.group], df.category, normalize="index").round(3).reindex(models, level=0)
    out += ["## Response categories (share of responses)\n", md(mix), ""]
    acc = df.pivot_table(index="model", columns="route", values="specific", aggfunc="mean").reindex(models)
    raw = df.pivot_table(index="model", columns="route", values="category", aggfunc=lambda c: (c == "correct").mean()).reindex(models)
    out += ["## Specific-correct rate per route (raw correct in parentheses)\n",
            md(pd.DataFrame({c: [f"{acc.loc[m, c]:.2f} ({raw.loc[m, c]:.2f})" for m in models] for c in acc.columns}, index=models)
            [[*DIRECT, *CODE, "comment"]].rename_axis("model")), ""]
    cg = df[df.group != "comment"].pivot_table(index="model", columns="group", values="cross_guess", aggfunc="mean").reindex(models)
    out += ["## Cross-subject guess rate (answer matches another subject's true answer; chance-level control)\n",
            md(cg.round(3)), ""]

    # 2. fact-level recovery on base-known facts
    ft = fact_table(df)
    base = ft[ft.model == "base"].set_index("fact_id")
    known = base.index[(base.direct_k > 0) & (base.code_k > 0)]
    deriv = {r.fact_id for r in df.itertuples() if name_derivable(Fact(r.relation, r.subject, tuple(r.aliases)))}
    rec = None
    for label, ids in (("all", known), ("non_derivable", [i for i in known if i not in deriv]),
                       ("derivable", [i for i in known if i in deriv])):
        t = recovery_table(ft[ft.fact_id.isin(ids)], models).assign(subset=label)
        rec = t if rec is None else pd.concat([rec, t], ignore_index=True)
    out += [f"## Fact-level direct x code, on the {len(known)} facts base Qwen knows through both routes\n",
            "`code_only` = every direct route wrong but at least one code route specifically correct (candidate recovery). "
            "C_rec = code_only / (facts with every direct route wrong). CI: paired bootstrap over facts of "
            "(code accuracy - direct accuracy). `non_derivable` drops facts whose answer can be spelled from the subject "
            "(helium -> He, Germany -> German), which need no world knowledge.\n",
            md(rec[["subset", *rec.columns.drop("subset")]], index=False), ""]
    by_rel = (ft[ft.fact_id.isin(known)].assign(rec=lambda x: (x.direct_k == 0) & (x.code_k > 0))
              .pivot_table(index="model", columns="relation", values="rec", aggfunc="sum").reindex(models))
    out += ["## Candidate recoveries by relation (count of facts)\n", md(by_rel), ""]
    ex = df[(df.model != "base") & df.specific & df.group.eq("code")]
    ex = ex[ex.apply(lambda r: ft[(ft.model == r.model) & (ft.fact_id == r.fact_id)].direct_k.iloc[0] == 0, axis=1)]
    ex = ex.assign(derivable=ex.fact_id.isin(deriv))
    out += ["## Every code-route recovery (direct routes all wrong)\n",
            md(ex[["model", "route", "subject", "span", "derivable"]], index=False) if len(ex) else "none", ""]

    text = "\n".join(out)
    (RESULTS_DIR / f"recall_probe_{a.tag}_analysis.md").write_text(text)
    rec.to_csv(RESULTS_DIR / f"recall_probe_{a.tag}_recovery.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
