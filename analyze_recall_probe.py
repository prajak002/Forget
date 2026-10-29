"""Analyze results/recall_probe_<tag>_responses.jsonl (from run_recall_probe.py).

Re-scores every raw output with the current verifier, then reports, per model:
  - category mix (correct / fluent_incorrect / garbage / refusal) for direct vs code routes
  - specificity control: a correct answer is discounted if the model gives that same answer for another
    subject of the same relation (and answer type) in the same route ("Paris" for France AND Italy is a default)
  - cross-subject guess rate: how often an answer matches a *different* subject's true answer (chance-level)
  - fact-level direct x code table on facts the base model knows through both routes, the recovery rate
    C_rec = P(code correct | all direct routes wrong), exact McNemar p, and a paired bootstrap 95% CI
  - per relation and per route: C_rec with Wilson 95% intervals, the recovery expected by chance, McNemar p with
    Holm correction across relations; and, when several seeds of a method are present, C_rec pooled over seeds
    with a fact-level bootstrap interval

  uv run python analyze_recall_probe.py --tag full
  uv run python analyze_recall_probe.py --tag full seeds          # pool seed runs with the seed-0 run
  uv run python analyze_recall_probe.py --tag large
  uv run python analyze_recall_probe.py --tag rwku

Writes results/recall_probe_<tag>_analysis.md, _recovery.csv, _by_relation.csv, _by_route.csv and, with seeds,
_pooled.csv.
"""
import argparse
import os
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from exclusive_unlearning.config import RESULTS_DIR
from exclusive_unlearning.recall_probe import (CODE, DIRECT, RELATIONS, Fact, alias_regex, categorize, name_derivable)

ALL = "ALL"


def md(t: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    t = t.reset_index() if index else t
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    return "\n".join(lines + ["| " + " | ".join(str(v) for v in row) + " |" for row in t.itertuples(index=False)])


def to_fact(r) -> Fact:
    return Fact(r.relation, r.subject, tuple(r.aliases), entity=str(getattr(r, "entity", "") or ""),
                answer_type=str(getattr(r, "answer_type", "") or ""))


def rescore(df: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """Categorize every row with the current verifier (func-route programs run in parallel sandboxes)."""
    rows = list(df.itertuples())
    with ThreadPoolExecutor(workers or min(8, os.cpu_count() or 1)) as ex:
        cats = list(ex.map(lambda r: categorize(r.route, to_fact(r), r.prompt, r.output), rows))
    return df.assign(category=[c for c, _ in cats], span=[s for _, s in cats])


def add_specificity(df: pd.DataFrame) -> pd.DataFrame:
    """specific = correct and no other subject (same model/route/relation/answer type) got this fact's answer too.
    cross_guess = share of the other subjects' true answers that appear in this response (chance-level guessing).

    Each fact's answer regex is run once over all spans of its group joined by NUL (no alias contains NUL, and NUL
    is neither a letter nor a digit, so the word-boundary rules behave as at string ends)."""
    df = df.reset_index(drop=True)
    if "answer_type" not in df:
        df = df.assign(answer_type="")
    df["answer_type"] = df["answer_type"].fillna("").astype(str)
    spec = np.zeros(len(df), bool)
    guess = np.zeros(len(df))
    for (_, _, rel, _), g in df.groupby(["model", "route", "relation", "answer_type"], sort=False):
        idx = g.index.to_numpy()
        spans = [str(s) for s in g.span]
        starts = np.cumsum([0] + [len(s) + 1 for s in spans[:-1]]).tolist()
        joined = "\x00".join(spans)
        cs = RELATIONS[rel].case_sensitive
        n = len(idx)
        hits = np.zeros((n, n), bool)  # hits[i, j]: fact i's answer appears in span j
        for i, aliases in enumerate(g.aliases):
            for m in alias_regex(tuple(aliases), cs).finditer(joined):
                hits[i, bisect_right(starts, m.start()) - 1] = True
        own = np.eye(n, dtype=bool)
        reused = (hits & ~own).any(1)
        spec[idx] = (g.category.to_numpy() == "correct") & ~reused
        guess[idx] = (hits & ~own).sum(0) / (n - 1) if n > 1 else 0.0
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


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ci = binomtest(k, n).proportion_ci(0.95, method="wilson")
    return float(ci.low), float(ci.high)


def mcnemar(n01: int, n10: int) -> float:
    return binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else float("nan")


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values (NaN stays NaN)."""
    q = p.dropna().sort_values()
    m, run, out = len(q), 0.0, {}
    for i, (k, v) in enumerate(q.items()):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return p.index.to_series().map(out).astype(float)


def recovery_table(ft: pd.DataFrame, models) -> pd.DataFrame:
    rows = []
    for m in models:
        t = ft[ft.model == m]
        d, c = t.direct_k > 0, t.code_k > 0
        n01, n10 = int((~d & c).sum()), int((d & ~c).sum())
        p = mcnemar(n01, n10)
        diff = (t.code_k / len(CODE) - t.direct_k / len(DIRECT)).to_numpy()
        lo, hi = paired_ci(diff)
        rows.append({"model": m, "facts": len(t), "both": int((d & c).sum()), "code_only": n01, "direct_only": n10,
                     "neither": int((~d & ~c).sum()), "C_rec": round(n01 / max(int((~d).sum()), 1), 3),
                     "mcnemar_p": float(f"{p:.2g}") if p == p else p, "code_minus_direct": round(float(diff.mean()), 3),
                     "ci95": f"[{lo:.3f}, {hi:.3f}]"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ per relation / per route

def stratum(df: pd.DataFrame) -> pd.Series:
    """Reporting unit: the relation, split by answer type where one is set (RWKU: rwku_qa/year, rwku_qa/text, ...)."""
    at = df["answer_type"].fillna("").astype(str) if "answer_type" in df else pd.Series("", index=df.index)
    return df.relation.where(at == "", df.relation + "/" + at)


def route_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, fact): specific-correct flag per route, plus stratum, direct_fail, code_any, code guess rate."""
    w = df.pivot_table(index=["model", "fact_id"], columns="route", values="specific", aggfunc="max").fillna(False)
    w = w.astype(bool)
    info = df.drop_duplicates(["model", "fact_id"]).set_index(["model", "fact_id"])
    w["stratum"] = stratum(info).reindex(w.index)
    w["direct_fail"] = ~w[list(DIRECT)].any(axis=1)
    w["code_any"] = w[list(CODE)].any(axis=1)
    g = df[df.route.isin(CODE)].groupby(["model", "fact_id"]).cross_guess.mean()
    w["code_guess"] = g.reindex(w.index).fillna(0.0)
    return w.reset_index()


def relation_tables(w: pd.DataFrame, models, subsets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel_rows, route_rows = [], []
    for subset, ids in subsets.items():
        for m in models:
            t = w[(w.model == m) & w.fact_id.isin(ids)]
            strata = sorted(t.stratum.unique())
            block = []
            for s in [*strata, ALL]:
                u = t if s == ALL else t[t.stratum == s]
                fail = u.direct_fail
                n_fail, rec = int(fail.sum()), int((fail & u.code_any).sum())
                direct_only = int((~fail & ~u.code_any).sum())
                lo, hi = wilson(rec, n_fail)
                # chance: if code answers were guesses from the relation's answer pool, each of the 3 code routes
                # would hit the right answer with probability ~ cross_guess
                g = u.code_guess[fail].to_numpy()
                chance = float((1 - (1 - g) ** len(CODE)).sum())
                block.append({"model": m, "subset": subset, "relation": s, "facts": len(u), "direct_failed": n_fail,
                              "recovered": rec, "C_rec": round(rec / n_fail, 3) if n_fail else float("nan"),
                              "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "chance_expected": round(chance, 2),
                              "mcnemar_p": mcnemar(rec, direct_only)})
                for r in (*CODE, "comment"):
                    k = int((fail & u[r]).sum()) if r in u else 0
                    lo, hi = wilson(k, n_fail)
                    route_rows.append({"model": m, "subset": subset, "relation": s, "route": r, "direct_failed": n_fail,
                                       "recovered": k, "rate": round(k / n_fail, 3) if n_fail else float("nan"),
                                       "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
            b = pd.DataFrame(block)
            per = b.relation != ALL
            b.loc[per, "p_holm"] = holm(b.loc[per, "mcnemar_p"]).to_numpy()
            rel_rows.append(b)
    rel = pd.concat(rel_rows, ignore_index=True)
    for c in ("mcnemar_p", "p_holm"):
        rel[c] = rel[c].map(lambda v: float(f"{v:.2g}") if v == v else v)
    return rel, pd.DataFrame(route_rows)


def pooled_table(w: pd.DataFrame, models, subsets: dict, n_boot: int = 4000, seed: int = 0) -> pd.DataFrame | None:
    """C_rec pooled over the seeds of each method ("cg_oss", "cg_oss@1", ...): recovered / direct-failed summed over
    seeds; 95% interval from a bootstrap over facts (all seeds of a resampled fact move together)."""
    methods = {}
    for m in models:
        if m != "base":
            methods.setdefault(m.split("@")[0], []).append(m)
    methods = {k: v for k, v in methods.items() if len(v) > 1}
    if not methods:
        return None
    rng, rows = np.random.default_rng(seed), []
    for subset, ids in subsets.items():
        for meth, ms in methods.items():
            t = w[w.model.isin(ms) & w.fact_id.isin(ids)].assign(rec=lambda x: x.direct_fail & x.code_any)
            per_fact = t.groupby(["fact_id", "stratum"])[["rec", "direct_fail"]].sum().reset_index()
            for s in [*sorted(per_fact.stratum.unique()), ALL]:
                u = per_fact if s == ALL else per_fact[per_fact.stratum == s]
                rec, fail = u.rec.to_numpy(float), u.direct_fail.to_numpy(float)
                if fail.sum() == 0:
                    continue
                if rec.sum() in (0, fail.sum()):
                    # all-or-none: a bootstrap interval collapses to a point; use Wilson over facts instead
                    n_f = int((fail > 0).sum())
                    lo, hi = wilson(n_f if rec.sum() else 0, n_f)
                else:
                    b = rng.integers(0, len(u), (n_boot, len(u)))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        boots = rec[b].sum(1) / fail[b].sum(1)
                    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
                rows.append({"method": meth, "seeds": len(ms), "subset": subset, "relation": s, "facts": len(u),
                             "recovered": int(rec.sum()), "direct_failed": int(fail.sum()),
                             "C_rec": round(rec.sum() / fail.sum(), 3), "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3)})
    return pd.DataFrame(rows)


def fmt_ci(t: pd.DataFrame, value: str) -> pd.Series:
    return t.apply(lambda r: f"{r[value]:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]" if r[value] == r[value] else "n/a", axis=1)


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
    deriv = {r.fact_id for r in df.drop_duplicates("fact_id").itertuples() if name_derivable(to_fact(r))}
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

    # 3. per relation and per route, with intervals
    w = route_wide(df)
    facts = df.drop_duplicates("fact_id").set_index("fact_id")
    st = stratum(facts)
    cover = pd.DataFrame({"written": st.value_counts(),
                          "base_known": st[st.index.isin(known)].value_counts(),
                          "base_known_non_derivable": st[st.index.isin(known) & ~st.index.isin(list(deriv))].value_counts()})
    cover = cover.fillna(0).astype(int).sort_index()
    cover.loc[ALL] = cover.sum()
    out += ["## Facts per relation: written, known by base (direct and code), and not spellable\n",
            md(cover.rename_axis("relation")), ""]
    subsets = {"non_derivable": [i for i in known if i not in deriv], "all": list(known)}
    rel_t, route_t = relation_tables(w, models, subsets)
    unl = [m for m in models if m != "base"]
    for subset in subsets:
        r = rel_t[(rel_t.subset == subset) & rel_t.model.isin(unl)]
        if r.empty:
            continue
        r = r.assign(C_rec_ci=fmt_ci(r, "C_rec"))
        out += [f"## Recovery by relation, `{subset}` facts (C_rec with Wilson 95% interval)\n",
                "C_rec = facts recovered through at least one code route / facts every direct route failed. "
                "`chance_expected`: recoveries expected if code answers were guesses from the relation's answer pool "
                "(from the cross-subject guess rate). McNemar p per relation, Holm-corrected across relations.\n",
                md(r[["model", "relation", "facts", "direct_failed", "recovered", "C_rec_ci", "chance_expected",
                      "mcnemar_p", "p_holm"]], index=False), ""]
        rr = route_t[(route_t.subset == subset) & route_t.model.isin(unl)]
        piv = (rr.assign(cell=fmt_ci(rr, "rate")).pivot_table(index=["model", "relation"], columns="route", values="cell",
                                                              aggfunc="first")[[*CODE, "comment"]])
        out += [f"## Recovery by relation and route, `{subset}` facts (rate [Wilson 95%] among direct-failed facts)\n",
                md(piv), ""]
    pooled = pooled_table(w, models, subsets)
    if pooled is not None and len(pooled):
        out += ["## Recovery pooled over training seeds (bootstrap over facts, 95% interval)\n",
                md(pooled.assign(C_rec_ci=fmt_ci(pooled, "C_rec"))
                   [["method", "seeds", "subset", "relation", "facts", "direct_failed", "recovered", "C_rec_ci"]], index=False), ""]
        pooled.to_csv(RESULTS_DIR / f"recall_probe_{a.tag}_pooled.csv", index=False)

    text = "\n".join(out)
    (RESULTS_DIR / f"recall_probe_{a.tag}_analysis.md").write_text(text)
    rec.to_csv(RESULTS_DIR / f"recall_probe_{a.tag}_recovery.csv", index=False)
    rel_t.to_csv(RESULTS_DIR / f"recall_probe_{a.tag}_by_relation.csv", index=False)
    route_t.to_csv(RESULTS_DIR / f"recall_probe_{a.tag}_by_route.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
