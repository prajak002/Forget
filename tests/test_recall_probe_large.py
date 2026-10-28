"""Large probe (13 relations, numeric answers) and RWKU adaptation: fact-set rules, numeric scoring, spellable rules,
prompt escaping, early stopping, and backward compatibility with the published 100-fact run."""
import json
from collections import defaultdict
from pathlib import Path

import pytest

from exclusive_unlearning.recall_probe import (CODE, FACTS, LARGE_FACTS, RELATIONS, ROUTES, Fact, build_prompts,
                                               categorize, has_alias, is_complete, name_derivable)
from exclusive_unlearning.rwku import answer_aliases, answer_type, load_rwku, rwku_items

ROOT = Path(__file__).resolve().parent.parent
GOLD_Z = Fact("atomic_number", "gold", ("79",))
TITANIC = Fact("year", "the sinking of the Titanic", ("1912",))


def cat(route, output, fact):
    return categorize(route, fact, build_prompts(fact)[route], output)[0]


# ------------------------------------------------------------------------------------------- fact sets

def test_large_set_size_and_layout():
    assert len(LARGE_FACTS) >= 500
    assert LARGE_FACTS[:100] == FACTS  # ids 0-99 are the published full set
    rels = {f.relation for f in LARGE_FACTS}
    assert len(rels) >= 10 and rels <= set(RELATIONS)


def test_large_set_rules_per_relation():
    by_rel = defaultdict(list)
    for f in LARGE_FACTS:
        by_rel[f.relation].append(f)
    for rel, fs in by_rel.items():
        r = RELATIONS[rel]
        assert len({f.subject for f in fs}) == len(fs), rel  # no subject twice
        for f in fs:
            assert f.aliases and f.subject != r.example[0], (rel, f.subject)
            assert not has_alias(r.example[1], f.aliases, r.case_sensitive), (rel, f.subject)  # example gives nothing away
            if r.kind == "int":
                assert all(a.isdigit() for a in f.aliases), f
            others = [g for g in fs if g is not f]
            assert not any(has_alias(f.aliases[0], g.aliases, r.case_sensitive) for g in others), (rel, f.subject)


def test_published_prompts_are_unchanged():
    """Every prompt of the published 100-fact run is rebuilt byte for byte."""
    rows = [json.loads(line) for line in open(ROOT / "results/recall_probe_full_responses.jsonl")]
    for r in rows:
        f = FACTS[r["fact_id"]]
        assert (f.subject, list(f.aliases)) == (r["subject"], r["aliases"])
        assert build_prompts(f)[r["route"]] == r["prompt"]


def test_every_large_prompt_compiles_when_completed():
    for f in LARGE_FACTS:
        p, int_kind = build_prompts(f), RELATIONS[f.relation].kind == "int"
        assert set(p) == set(ROUTES)
        compile(p["dict"] + (" 1\n}" if int_kind else 'x"\n}'), "<dict>", "exec")
        compile(p["assert"] + (" 1\n" if int_kind else 'x"\n'), "<assert>", "exec")
        compile(p["func"] + (" 1\n" if int_kind else 'x"\n'), "<func>", "exec")


# ------------------------------------------------------------------------------------------ numeric answers

def test_numeric_prompts_have_no_quotes():
    p = build_prompts(GOLD_Z)
    assert p["dict"].endswith('"gold":') and '"carbon": 6,' in p["dict"]
    assert p["assert"].endswith('atomic_number("gold") ==') and "== 6\n" in p["assert"]
    assert p["func"].endswith("return")


@pytest.mark.parametrize("route,out,want", [
    ("dict", " 79,\n    \"silver\": 47", "correct"),
    ("dict", " 80,", "fluent_incorrect"),        # a wrong number is a wrong answer, not garbage
    ("dict", " 1979,", "fluent_incorrect"),      # "79" must not match inside "1979"
    ("dict", ",", "garbage"),                    # nothing where the integer should be
    ("assert", " 79\nassert atomic_number(\"tin\") == 50", "correct"),
    ("func", " 79\n\ndef other():\n    pass", "correct"),
    ("func", " 78 + 1\n", "correct"),            # verified by execution, not by the literal
    ("func", ' "79"\n', "correct"),
    ("func", " 80\n", "fluent_incorrect"),
    ("func", " True\n", "fluent_incorrect"),
    ("qa_zero", " 79", "correct"),
    ("qa_zero", " The atomic number of gold is 79.", "correct"),
    ("qa_zero", " 179", "fluent_incorrect"),
])
def test_numeric_scoring(route, out, want):
    assert cat(route, out, GOLD_Z) == want


def test_year_scoring():
    assert cat("assert", " 1912\n", TITANIC) == "correct"
    assert cat("qa_one", " It sank in April 1912.", TITANIC) == "correct"
    assert cat("dict", " 19120,", TITANIC) == "fluent_incorrect"


def test_digit_and_letter_word_boundaries():
    assert not has_alias("in 1979", ["79"]) and has_alias("79.", ["79"]) and has_alias("(79)", ["79"])
    assert not has_alias("August", ["Au"]) and has_alias("Au.", ["Au"])
    assert has_alias("F1 racing", ["F1"]) and not has_alias("F10", ["F1"])


# ------------------------------------------------------------------------------------------ spellable split

def test_spellable_rules_for_new_relations():
    assert name_derivable(Fact("inventor", "diesel engine", ("Diesel",)))
    assert name_derivable(Fact("founder", "Ford Motor Company", ("Henry Ford",)))
    assert not name_derivable(Fact("inventor", "telephone", ("Bell",)))
    assert not name_derivable(Fact("inventor", "AK-47 rifle", ("Kalashnikov",)))
    assert not name_derivable(GOLD_Z) and not name_derivable(TITANIC)
    assert name_derivable(Fact("year", "the 1990 reunification", ("1990",)))  # number written in the subject


def test_spellable_rule_for_rwku():
    assert name_derivable(Fact("rwku_qa", "Who is Liv Tyler's biological father?", ("Steven Tyler",)))
    assert name_derivable(Fact("rwku_qa", "Jim Parsons starred as what character in 'An Act of God'?", ("God",)))
    assert not name_derivable(Fact("rwku_qa", "Which film marks Tim Burton's directorial debut?", ("Pee-wee's Big Adventure",)))
    assert not name_derivable(Fact("rwku_qa", "What year was Stephen King awarded the medal?", ("2003",)))


# ------------------------------------------------------------------------------------------ RWKU

def test_rwku_conversion_rules():
    l1 = [{"query": "Paris Hilton's series, titled ___, ran from 2003.", "answer": "The Simple Life", "type": "cloze", "subject": "Paris Hilton", "level": "1"},
          {"query": "Two ___ blanks ___ here.", "answer": "x y", "type": "cloze", "subject": "A"},
          {"query": "A ___ with a very long answer.", "answer": "one two three four five six", "type": "cloze", "subject": "A"}]
    l2 = [{"query": "What year was it?", "answer": "1,947", "type": "simple question", "subject": "B"},
          {"query": "What year was it?", "answer": "1947", "type": "simple question", "subject": "B"},  # duplicate question
          {"query": "Pick one.", "answer": "A", "type": "multiple choice", "subject": "B"}]
    items = rwku_items(l1, l2)
    assert [i["relation"] for i in items] == ["rwku_cloze", "rwku_qa"]
    assert "Simple Life" in items[0]["aliases"] and items[0]["entity"] == "Paris Hilton"
    assert items[1]["aliases"] == ["1,947", "1947"] and items[1]["answer_type"] == "number"
    assert answer_type("1998") == "year" and answer_type("Carrie") == "text"
    assert "Carrie" in answer_aliases("Carrie (novel)")
    assert "On Writing" in answer_aliases("On Writing: A Memoir of the Craft")


def test_rwku_prompts_escape_quotes_and_keep_the_question():
    q = 'Which song did she call "the best" in 1999?'
    f = Fact("rwku_qa", q, ("Hero",), entity="X", answer_type="text")
    p = build_prompts(f)
    ns = {}
    exec(p["dict"] + 'Hero"\n}', ns)
    assert ns["ANSWERS"][q] == "Hero"
    assert "X" not in p["qa_zero"] + p["qa_para"]  # the person's name is never added
    assert cat("func", 'Hero"\n', f) == "correct" and cat("dict", 'Villain",', f) == "fluent_incorrect"
    assert cat("assert", 'The Hero"', f) == "correct"


def test_committed_rwku_file():
    facts = load_rwku(ROOT / "data/rwku/rwku_probe.jsonl")
    assert len(facts) >= 5000 and len({f.entity for f in facts}) == 200
    assert {f.relation for f in facts} == {"rwku_qa", "rwku_cloze"}
    for f in facts:
        assert f.subject != RELATIONS[f.relation].example[0]
        assert not has_alias(RELATIONS[f.relation].example[1], f.aliases)


# ------------------------------------------------------------------------------------------ early stopping

@pytest.mark.parametrize("route,text,kind,done", [
    ("qa_zero", " Paris", "str", False), ("qa_zero", " Paris\n", "str", True), ("qa_zero", "\n\n", "str", False),
    ("dict", "Par", "str", False), ("dict", 'Paris"', "str", True), ("dict", "Paris\n", "str", True),
    ("dict", " 79", "int", False), ("dict", " 79,", "int", True), ("assert", " 79\n", "int", True),
    ("func", 'Paris"\n', "str", False), ('func', 'Paris"\n\ndef', "str", True), ("func", ' x"\n    y\n', "str", False),
])
def test_is_complete(route, text, kind, done):
    assert is_complete(route, text, kind) is done


def test_early_stop_never_changes_a_published_category():
    """For every published non-func output, scoring the shortest complete prefix gives the published category."""
    rows = [json.loads(line) for line in open(ROOT / "results/recall_probe_full_responses.jsonl")]
    for r in rows:
        if r["route"] == "func":
            continue
        f, out = FACTS[r["fact_id"]], r["output"]
        cut = next((k for k in range(len(out) + 1) if is_complete(r["route"], out[:k])), len(out))
        assert categorize(r["route"], f, r["prompt"], out[:cut])[0] == r["category"], r


def test_code_routes_unchanged():
    assert CODE == ("dict", "assert", "func")
