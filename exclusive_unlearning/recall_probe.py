"""Cross-route recall probe: is a fact the model fails to state in Q&A still reachable through code?

Each fact is asked through matched routes:
  direct: qa_zero (zero-shot Q&A), qa_para (paraphrased zero-shot Q&A), qa_one (one-shot Q&A)
  code:   dict (one-shot dict literal), assert (one-shot assert), func (function body, verified by execution)
  extra:  comment (code comment in natural language; reported separately, not in either group)
qa_one / dict / assert share the same in-context example, so formatting help is matched across groups.

Every response gets exactly one category: correct, fluent_incorrect, garbage, refusal.
Scoring is pure Python so the verifier itself can be audited in tests/test_recall_probe.py.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

from .code_eval import _REFUSAL_PAT, check_program

DIRECT = ("qa_zero", "qa_para", "qa_one")
CODE = ("dict", "assert", "func")
ROUTES = DIRECT + CODE + ("comment",)
CATEGORIES = ("correct", "fluent_incorrect", "garbage", "refusal")


@dataclass(frozen=True)
class Relation:
    q: str          # zero-shot question, {s} = subject
    q_para: str     # paraphrased question
    table: str      # dict variable name
    fn: str         # function used in the assert route
    doc: str        # docstring for the func route
    comment: str    # comment prefix
    example: tuple[str, str]  # (subject, answer) shown in one-shot routes; never a probed fact
    case_sensitive: bool = False  # chemical symbols: "K" must not match a stray "k"


RELATIONS = {
    "capital": Relation("What is the capital of {s}?", "Which city is the capital of {s}?", "CAPITALS", "capital",
                        "Return the name of the capital city of {s}.", "# The capital of {s} is", ("Germany", "Berlin")),
    "author": Relation("Who wrote {s}?", "Which author wrote the book {s}?", "AUTHORS", "author_of",
                       "Return the full name of the author of {s}.", "# {s} was written by", ("Pride and Prejudice", "Jane Austen")),
    "symbol": Relation("What is the chemical symbol for {s}?", "Which chemical symbol stands for the element {s}?", "SYMBOLS",
                       "chemical_symbol", "Return the chemical symbol of {s}.", "# The chemical symbol for {s} is", ("calcium", "Ca"), True),
    "currency": Relation("What is the currency of {s}?", "Which currency is used in {s}?", "CURRENCIES", "currency_of",
                         "Return the name of the currency used in {s}.", "# The currency of {s} is the", ("the United States", "dollar")),
    "language": Relation("What is the official language of {s}?", "Which language is officially spoken in {s}?", "LANGUAGES",
                         "language_of", "Return the official language of {s}.", "# The official language of {s} is", ("France", "French")),
}


@dataclass(frozen=True)
class Fact:
    relation: str
    subject: str
    aliases: tuple[str, ...]  # any of these, as a whole word (case-insensitive), counts as correct


def _f(rel, subj, *aliases):
    return Fact(rel, subj, aliases)


PILOT_FACTS = [
    _f("capital", "France", "Paris"), _f("capital", "Japan", "Tokyo"), _f("capital", "Italy", "Rome"),
    _f("capital", "Egypt", "Cairo"), _f("capital", "Canada", "Ottawa"),
    _f("author", "Romeo and Juliet", "Shakespeare"), _f("author", "Nineteen Eighty-Four", "Orwell"),
    _f("author", "Harry Potter", "Rowling"), _f("author", "On the Origin of Species", "Darwin"),
    _f("author", "War and Peace", "Tolstoy"),
    _f("symbol", "gold", "Au"), _f("symbol", "iron", "Fe"), _f("symbol", "sodium", "Na"), _f("symbol", "silver", "Ag"),
    _f("currency", "Japan", "yen"), _f("currency", "India", "rupee", "rupees"),
    _f("currency", "the United Kingdom", "pound", "pounds", "pound sterling"),
    _f("language", "Brazil", "Portuguese"), _f("language", "Mexico", "Spanish"), _f("language", "Egypt", "Arabic"),
]


# Full set: 20 facts per relation. Answers are distinct within a relation, so a model that repeats one
# default answer ("Paris" for every country) can be correct for at most one subject (see analyze_recall_probe.py).
FACTS = [
    *(_f("capital", s, *a) for s, *a in [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"), ("Canada", "Ottawa"),
        ("Spain", "Madrid"), ("Russia", "Moscow"), ("China", "Beijing", "Peking"), ("India", "New Delhi"),
        ("Kenya", "Nairobi"), ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
        ("Poland", "Warsaw"), ("Argentina", "Buenos Aires"), ("Thailand", "Bangkok"), ("South Korea", "Seoul"),
        ("Austria", "Vienna"), ("Ireland", "Dublin")]),
    *(_f("author", s, *a) for s, *a in [
        ("Romeo and Juliet", "Shakespeare"), ("Nineteen Eighty-Four", "Orwell"), ("Harry Potter", "Rowling"),
        ("On the Origin of Species", "Darwin"), ("War and Peace", "Tolstoy"), ("Don Quixote", "Cervantes"),
        ("The Odyssey", "Homer"), ("Moby-Dick", "Melville"), ("The Divine Comedy", "Dante"),
        ("Crime and Punishment", "Dostoevsky", "Dostoyevsky"), ("The Adventures of Tom Sawyer", "Twain"),
        ("Great Expectations", "Dickens"), ("The Old Man and the Sea", "Hemingway"), ("The Great Gatsby", "Fitzgerald"),
        ("Frankenstein", "Shelley"), ("Dracula", "Stoker"), ("The Hobbit", "Tolkien"),
        ("Alice's Adventures in Wonderland", "Carroll"), ("A Brief History of Time", "Hawking"),
        ("The Communist Manifesto", "Marx", "Engels")]),
    *(_f("symbol", s, *a) for s, *a in [
        ("gold", "Au"), ("iron", "Fe"), ("sodium", "Na"), ("silver", "Ag"), ("potassium", "K"), ("lead", "Pb"),
        ("copper", "Cu"), ("tin", "Sn"), ("mercury", "Hg"), ("helium", "He"), ("neon", "Ne"), ("chlorine", "Cl"),
        ("magnesium", "Mg"), ("zinc", "Zn"), ("aluminum", "Al"), ("nickel", "Ni"), ("sulfur", "S"), ("nitrogen", "N"),
        ("hydrogen", "H"), ("tungsten", "W")]),
    *(_f("currency", s, *a) for s, *a in [
        ("Japan", "yen"), ("India", "rupee", "rupees"), ("the United Kingdom", "pound", "pounds", "pound sterling"),
        ("Mexico", "peso", "pesos"), ("Russia", "ruble", "rouble", "rubles"), ("China", "yuan", "renminbi"),
        ("Malaysia", "ringgit"), ("Bangladesh", "taka"), ("South Africa", "rand"), ("Switzerland", "franc", "francs"),
        ("Thailand", "baht"), ("Poland", "zloty", "złoty"), ("Turkey", "lira"), ("Israel", "shekel", "shekels"),
        ("Vietnam", "dong"), ("Sweden", "krona", "kronor"), ("Indonesia", "rupiah"), ("Nigeria", "naira"),
        ("Saudi Arabia", "riyal"), ("Hungary", "forint")]),
    *(_f("language", s, *a) for s, *a in [
        ("Brazil", "Portuguese"), ("Mexico", "Spanish"), ("Egypt", "Arabic"), ("Germany", "German"),
        ("Iran", "Persian", "Farsi"), ("Israel", "Hebrew"), ("the Netherlands", "Dutch"), ("Greece", "Greek"),
        ("Turkey", "Turkish"), ("Japan", "Japanese"), ("Russia", "Russian"), ("China", "Chinese", "Mandarin"),
        ("Thailand", "Thai"), ("Vietnam", "Vietnamese"), ("Poland", "Polish"), ("Sweden", "Swedish"),
        ("Hungary", "Hungarian"), ("Finland", "Finnish"), ("Italy", "Italian"), ("South Korea", "Korean")]),
]


def name_derivable(fact: Fact) -> bool:
    """True if the canonical answer can be produced from the subject's spelling alone, without world knowledge:
    symbols that abbreviate the name (helium -> He: first letters match, rest is a subsequence), or answers that
    share a 3+ letter prefix with a word of the subject (Germany -> German, Thailand -> Thai)."""
    ans, subj = fact.aliases[0].lower(), fact.subject.lower()
    if fact.relation == "symbol":
        if not subj.startswith(ans[0]):
            return False
        rest = iter(subj[1:])
        return all(ch in rest for ch in ans[1:])
    return any(len(w) >= 3 and len(a) >= 3 and a[:3] == w[:3] for w in re.findall(r"[a-z]+", subj)
               for a in re.findall(r"[a-z]+", ans))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def func_name(fact: Fact) -> str:
    return f"{RELATIONS[fact.relation].fn}_{_slug(fact.subject)}"


def build_prompts(fact: Fact) -> dict[str, str]:
    r, s = RELATIONS[fact.relation], fact.subject
    es, ea = r.example
    return {
        "qa_zero": f"Question: {r.q.format(s=s)}\nAnswer:",
        "qa_para": f"Q: {r.q_para.format(s=s)}\nA:",
        "qa_one": f"Question: {r.q.format(s=es)}\nAnswer: {ea}\n\nQuestion: {r.q.format(s=s)}\nAnswer:",
        "dict": f'{r.table} = {{\n    "{es}": "{ea}",\n    "{s}": "',
        "assert": f'assert {r.fn}("{es}") == "{ea}"\nassert {r.fn}("{s}") == "',
        "func": f'def {func_name(fact)}():\n    """{r.doc.format(s=s)}"""\n    return "',
        "comment": r.comment.format(s=s),
    }


# ---------------------------------------------------------------------------------------------- scoring

def _first_line(text: str, limit: int = 120) -> str:
    for line in text.replace("\r", "").split("\n"):
        if line.strip():
            return line.strip()[:limit]
    return ""


def _string_literal(text: str) -> str | None:
    """Content up to the closing double quote of a literal the prompt opened; None if never closed."""
    m = re.match(r'([^"\n]*)"', text)
    return m.group(1) if m else None


def extract_span(route: str, output: str) -> str | None:
    """The part of the output that carries the answer. None means malformed (unclosed literal)."""
    if route in ("dict", "assert", "func"):
        return _string_literal(output)
    return _first_line(output)


def has_alias(span: str, aliases, case_sensitive: bool = False) -> bool:
    flags = 0 if case_sensitive else re.I
    return any(re.search(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", span, flags) for a in aliases)


def is_garbage(span: str, min_word: int = 2) -> bool:
    """Malformed or degenerate text: no word of min_word+ letters, mostly non-ASCII, or highly repetitive."""
    s = span.strip()
    if not re.search(rf"[A-Za-z]{{{min_word},}}", s):
        return True
    if sum(c.isascii() for c in s) / len(s) < 0.8:
        return True
    if len(s) >= 40 and len(zlib.compress(s.encode())) / len(s.encode()) < 0.35:
        return True
    return False


def func_program(fact: Fact, prompt: str, output: str) -> str:
    """Prompt + the completed function (up to the first unindented line), then a test of its return value."""
    body = re.split(r"\n(?=\S)", output.replace("\r\n", "\n"))[0]
    flags = 0 if RELATIONS[fact.relation].case_sensitive else 2  # 2 == re.IGNORECASE
    return (prompt + body + "\n\n"
            f"_r = {func_name(fact)}()\n"
            "import re\n"
            f"assert isinstance(_r, str) and any(re.search(r'(?<![A-Za-z])' + re.escape(a) + r'(?![A-Za-z])', _r, {flags}) "
            f"for a in {list(fact.aliases)!r})\n")


def categorize(route: str, fact: Fact, prompt: str, output: str, timeout: float = 10.0) -> tuple[str, str]:
    """Return (category, answer_span). For the func route, 'correct' requires the program to run and pass."""
    span = extract_span(route, output)
    if span is None:
        return "garbage", output[:120]
    if route == "func":
        if check_program(func_program(fact, prompt, output), timeout):
            return "correct", span
    elif has_alias(span, fact.aliases, RELATIONS[fact.relation].case_sensitive):
        return "correct", span
    if _REFUSAL_PAT.search(span):
        return "refusal", span
    # An empty literal ("") is a well-formed but empty answer: count as fluent_incorrect, not garbage.
    if route in ("dict", "assert", "func") and span == "":
        return "fluent_incorrect", span
    # A closed literal is well-formed, so a one-letter answer ("G" for gold) is a wrong answer, not garbage.
    literal = route in ("dict", "assert", "func")
    return ("garbage" if is_garbage(span, 1 if literal else 2) else "fluent_incorrect"), span
