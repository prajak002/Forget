"""Verifier audit for the cross-route recall probe: accepts valid alternatives, rejects wrong/garbage answers."""
import pytest

from exclusive_unlearning.recall_probe import CATEGORIES, PILOT_FACTS, RELATIONS, ROUTES, Fact, build_prompts, categorize

PARIS = Fact("capital", "France", ("Paris",))
GOLD = Fact("symbol", "gold", ("Au",))


def cat(route, output, fact=PARIS):
    return categorize(route, fact, build_prompts(fact)[route], output)[0]


@pytest.mark.parametrize("out", [" Paris\nA:", " The capital of France is Paris.", " paris, of course", " PARIS"])
def test_qa_accepts_valid_phrasings(out):
    assert cat("qa_zero", out) == "correct"


@pytest.mark.parametrize("out", [" Lyon", " The capital of France is Marseille.", "\nParisian food is great\nParis"])
def test_qa_rejects_wrong_or_late_answers(out):
    # only the first non-empty line counts, and 'Parisian' is not the word 'Paris'
    assert cat("qa_zero", out) == "fluent_incorrect"


def test_qa_garbage_and_refusal():
    assert cat("qa_zero", " evityinsoleevityinsoleevityinsoleevityinsoleevityinsoleevityinsole") == "garbage"
    assert cat("qa_zero", " 嗐んですよね嗐んですよね") == "garbage"
    assert cat("qa_zero", " ',=',=']=") == "garbage"
    assert cat("qa_zero", " I don't know the answer.") == "refusal"


def test_literal_routes():
    assert cat("dict", 'Paris",\n    "Spain": "Madrid"') == "correct"
    assert cat("assert", 'Paris"\n') == "correct"
    assert cat("dict", 'Lyon",') == "fluent_incorrect"
    assert cat("dict", '",') == "fluent_incorrect"
    assert cat("dict", "Paris\n    }") == "garbage"  # literal never closed -> malformed, even if the word appears


def test_func_route_is_verified_by_execution():
    assert cat("func", 'Paris"\n\ndef other():\n    pass') == "correct"
    assert cat("func", 'paris".title()') == "correct"  # alternative valid implementation
    assert cat("func", 'Lyon"') == "fluent_incorrect"
    assert cat("func", 'Paris"[::-1]') == "fluent_incorrect"  # literal looks right, program returns the wrong value
    assert cat("func", 'Paris" +') == "fluent_incorrect"  # syntax error -> not correct


def test_short_answers_need_word_boundaries():
    assert cat("qa_zero", " Au", GOLD) == "correct"
    assert cat("qa_zero", " The symbol is Au.", GOLD) == "correct"
    assert cat("qa_zero", " August", GOLD) == "fluent_incorrect"


def test_prompts_and_examples_are_consistent():
    for f in PILOT_FACTS:
        p = build_prompts(f)
        assert set(p) == set(ROUTES)
        assert f.subject != RELATIONS[f.relation].example[0]
    assert len(CATEGORIES) == 4


def test_full_fact_set_has_distinct_answers_per_relation():
    from collections import defaultdict
    from exclusive_unlearning.recall_probe import FACTS, has_alias
    assert len(FACTS) == 100
    by_rel = defaultdict(list)
    for f in FACTS:
        by_rel[f.relation].append(f)
        assert f.subject != RELATIONS[f.relation].example[0]
    for rel, fs in by_rel.items():
        assert len(fs) == 20
        cs = RELATIONS[rel].case_sensitive
        for f in fs:  # no fact's canonical answer is accepted for another subject of the same relation
            others = [g for g in fs if g is not f]
            assert not any(has_alias(f.aliases[0], g.aliases, cs) for g in others), (rel, f.subject)


def test_symbols_are_case_sensitive_and_single_letter_literals_are_wrong_not_garbage():
    potassium = Fact("symbol", "potassium", ("K",))
    assert cat("qa_zero", " K", potassium) == "correct"
    assert cat("qa_zero", " k is a letter", potassium) == "fluent_incorrect"
    assert cat("func", 'K"', potassium) == "correct"
    assert cat("func", 'k"', potassium) == "fluent_incorrect"
    assert cat("dict", 'G",', GOLD) == "fluent_incorrect"
    assert cat("qa_zero", "\r\n\r\n\r\n") == "garbage"


def test_name_derivable_flags_answers_spelled_from_the_subject():
    from exclusive_unlearning.recall_probe import FACTS, name_derivable
    d = {(f.relation, f.subject): name_derivable(f) for f in FACTS}
    for key in [("symbol", "helium"), ("symbol", "zinc"), ("symbol", "chlorine"), ("symbol", "nitrogen"),
                ("language", "Germany"), ("language", "Thailand"), ("language", "South Korea"), ("language", "Poland")]:
        assert d[key], key
    for key in [("symbol", "iron"), ("symbol", "sodium"), ("symbol", "potassium"), ("symbol", "copper"),
                ("symbol", "tungsten"), ("language", "Mexico"), ("language", "Israel"), ("capital", "Japan"),
                ("author", "Romeo and Juliet"), ("currency", "Japan")]:
        assert not d[key], key
