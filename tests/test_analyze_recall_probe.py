"""Analysis of the recall probe: the fast specificity control matches the definition, and the per-relation and
per-route statistics count what they claim to."""
import numpy as np
import pandas as pd
import pytest

from analyze_recall_probe import (add_specificity, holm, pooled_table, relation_tables, route_wide, wilson)
from exclusive_unlearning.recall_probe import CODE, DIRECT, RELATIONS, has_alias

ROUTES = (*DIRECT, *CODE, "comment")


def reference_specificity(df):
    """The original row-by-row definition."""
    spec, guess = [], []
    for r in df.itertuples():
        g = df[(df.model == r.model) & (df.route == r.route) & (df.relation == r.relation) & (df.answer_type == r.answer_type)]
        others = g[g.fact_id != r.fact_id]
        cs = RELATIONS[r.relation].case_sensitive
        spec.append(r.category == "correct" and not any(has_alias(str(s), r.aliases, cs) for s in others.span))
        guess.append(np.mean([has_alias(str(r.span), a, cs) for a in others.aliases]) if len(others) else 0.0)
    return np.array(spec), np.array(guess)


def test_fast_specificity_matches_definition():
    rng = np.random.default_rng(0)
    pool = {"capital": ["Paris", "Rome", "Tokyo", "Lima"], "symbol": ["K", "Na", "Fe", "Au"], "year": ["1912", "1945", "79"]}
    rows = []
    for model in ("base", "m"):
        for rel, answers in pool.items():
            for fid, ans in enumerate(answers):
                for route in ("dict", "qa_zero"):
                    span = str(rng.choice([*answers, "x", "Paris and Rome", "k", "1979", "\n" + ans + "\n"]))
                    rows.append({"model": model, "fact_id": f"{rel}{fid}", "relation": rel, "route": route,
                                 "aliases": [ans], "span": span, "answer_type": "",
                                 "category": "correct" if has_alias(span, [ans], rel == "symbol") else "fluent_incorrect"})
    df = pd.DataFrame(rows)
    out = add_specificity(df)
    spec, guess = reference_specificity(df)
    assert (out.specific.to_numpy() == spec).all()
    assert np.allclose(out.cross_guess.to_numpy(), guess)


def test_wilson_and_holm():
    lo, hi = wilson(0, 20)
    assert lo == 0 and 0.15 < hi < 0.18
    lo, hi = wilson(10, 20)
    assert lo < 0.5 < hi
    assert np.isnan(wilson(0, 0)[0])
    p = pd.Series([0.01, 0.04, np.nan, 0.03], index=list("abcd"))
    adj = holm(p)
    assert adj["a"] == pytest.approx(0.03) and adj["d"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert np.isnan(adj["c"])


def synthetic():
    """3 relations x 10 facts; model 'm' fails every direct route, recovers the first k facts through `assert`."""
    rows, recovered = [], {"capital": 6, "author": 2, "symbol": 0}
    for rel, k in recovered.items():
        for i in range(10):
            fid = f"{rel}{i}"
            for model, seed_shift in (("base", None), ("m", 0), ("m@1", 1)):
                for route in ROUTES:
                    ok = model == "base" or (route == "assert" and i < k + (seed_shift or 0))
                    rows.append({"model": model, "fact_id": fid, "relation": rel, "route": route, "specific": ok,
                                 "cross_guess": 0.0, "answer_type": ""})
    return pd.DataFrame(rows), recovered


def test_relation_and_route_tables_count_correctly():
    df, recovered = synthetic()
    w = route_wide(df)
    ids = sorted(df.fact_id.unique())
    rel, route = relation_tables(w, ["m"], {"all": ids})
    for r, k in recovered.items():
        row = rel[(rel.relation == r)].iloc[0]
        assert (row.direct_failed, row.recovered) == (10, k)
        assert row.ci_lo <= k / 10 <= row.ci_hi
        a = route[(route.relation == r) & (route.route == "assert")].iloc[0]
        d = route[(route.relation == r) & (route.route == "dict")].iloc[0]
        assert a.recovered == k and d.recovered == 0
    tot = rel[rel.relation == "ALL"].iloc[0]
    assert (tot.facts, tot.recovered) == (30, 8)
    assert rel[rel.relation != "ALL"].p_holm.notna().any()


def test_pooled_over_seeds():
    df, _ = synthetic()
    w = route_wide(df)
    t = pooled_table(w, ["base", "m", "m@1"], {"all": sorted(df.fact_id.unique())})
    cap = t[(t.method == "m") & (t.relation == "capital")].iloc[0]
    assert (cap.seeds, cap.direct_failed, cap.recovered) == (2, 20, 13)  # 6 + 7
    sym = t[(t.method == "m") & (t.relation == "symbol")].iloc[0]
    assert sym.recovered == 1 and sym.ci_hi > sym.ci_lo  # seed 1 recovers one symbol fact
