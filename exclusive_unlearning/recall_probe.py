"""Cross-route recall probe: is a fact the model fails to state in Q&A still reachable through code?

Each fact is asked through matched routes:
  direct: qa_zero (zero-shot Q&A), qa_para (paraphrased zero-shot Q&A), qa_one (one-shot Q&A)
  code:   dict (one-shot dict literal), assert (one-shot assert), func (function body, verified by execution)
  extra:  comment (code comment in natural language; reported separately, not in either group)
qa_one / dict / assert share the same in-context example, so formatting help is matched across groups.

Fact sets: PILOT_FACTS (20), FACTS (100, 5 relations), LARGE_FACTS (FACTS + probe_facts.EXTRA, 13 relations),
and RWKU (an external benchmark, see rwku.py; relations rwku_qa and rwku_cloze).
Relations with kind="int" (atomic numbers, years) are asked for unquoted integer literals in code.

Every response gets exactly one category: correct, fluent_incorrect, garbage, refusal.
Scoring is pure Python so the verifier itself can be audited in tests/test_recall_probe.py.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from functools import lru_cache

from .probe_facts import EXTRA
from .sandbox import REFUSAL_PAT as _REFUSAL_PAT, check_program

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
    kind: str = "str"     # "int": code routes ask for an unquoted integer literal (ATOMIC_NUMBERS = {"carbon": 6, ...)
    spell: str = "prefix"  # spellable-answer rule, see name_derivable: "prefix" | "symbol" | "contains"


RELATIONS = {
    "capital": Relation("What is the capital of {s}?", "Which city is the capital of {s}?", "CAPITALS", "capital",
                        "Return the name of the capital city of {s}.", "# The capital of {s} is", ("Germany", "Berlin")),
    "author": Relation("Who wrote {s}?", "Which author wrote the book {s}?", "AUTHORS", "author_of",
                       "Return the full name of the author of {s}.", "# {s} was written by", ("Pride and Prejudice", "Jane Austen")),
    "symbol": Relation("What is the chemical symbol for {s}?", "Which chemical symbol stands for the element {s}?", "SYMBOLS",
                       "chemical_symbol", "Return the chemical symbol of {s}.", "# The chemical symbol for {s} is", ("calcium", "Ca"), True,
                       spell="symbol"),
    "currency": Relation("What is the currency of {s}?", "Which currency is used in {s}?", "CURRENCIES", "currency_of",
                         "Return the name of the currency used in {s}.", "# The currency of {s} is the", ("the United States", "dollar")),
    "language": Relation("What is the official language of {s}?", "Which language is officially spoken in {s}?", "LANGUAGES",
                         "language_of", "Return the official language of {s}.", "# The official language of {s} is", ("France", "French")),
    # --- added for the large probe (facts in probe_facts.py)
    "atomic_number": Relation("What is the atomic number of {s}?", "How many protons are in the nucleus of an atom of {s}?",
                              "ATOMIC_NUMBERS", "atomic_number", "Return the atomic number of {s}.",
                              "# The atomic number of {s} is", ("carbon", "6"), kind="int"),
    "year": Relation("In what year did {s} take place?", "Which year saw {s}?", "EVENT_YEARS", "year_of",
                     "Return the year of {s}.", "# {s} took place in the year", ("the end of World War I", "1918"), kind="int"),
    "inventor": Relation("Who invented the {s}?", "Which person is credited with inventing the {s}?", "INVENTORS", "inventor_of",
                         "Return the name of the inventor of the {s}.", "# The {s} was invented by",
                         ("printing press", "Johannes Gutenberg")),
    "river": Relation("Which river flows through {s}?", "What is the name of the river that runs through {s}?", "CITY_RIVERS",
                      "river_of", "Return the name of the river that flows through {s}.", "# The river that flows through {s} is the",
                      ("London", "Thames")),
    "unit": Relation("What is the SI unit of {s}?", "In the International System of Units, what is the unit of {s}?", "SI_UNITS",
                     "si_unit", "Return the name of the SI unit of {s}.", "# The SI unit of {s} is the", ("force", "newton")),
    "sport": Relation("Which sport is {s} associated with?", "In which sport is {s} contested?", "SPORTS", "sport_of",
                      "Return the name of the sport of {s}.", "# {s} is a competition in the sport of", ("the Stanley Cup", "ice hockey")),
    "maker": Relation("Which company makes {s}?", "What company produces {s}?", "MAKERS", "maker_of",
                      "Return the name of the company that makes {s}.", "# {s} is made by", ("the Xbox", "Microsoft")),
    "founder": Relation("Who founded {s}?", "Which person started the company {s}?", "FOUNDERS", "founder_of",
                        "Return the name of a founder of {s}.", "# {s} was founded by", ("Microsoft", "Bill Gates")),
    # --- RWKU (rwku.py): the subject is the whole question / cloze sentence; examples are about no RWKU person
    "rwku_qa": Relation("{s}", "Answer briefly. {s}", "ANSWERS", "answer", "{s}", "# Q: {s}\n# A:",
                        ("What is the largest planet in the Solar System?", "Jupiter"), spell="contains"),
    "rwku_cloze": Relation("Fill in the blank: {s}", "Which word or phrase completes this sentence? {s}", "BLANKS",
                           "fill_blank", "Return the text that fills the blank ___ in: {s}", "# {s}\n# The blank is:",
                           ("The largest ocean on Earth is the ___ Ocean.", "Pacific"), spell="contains"),
}


@dataclass(frozen=True)
class Fact:
    relation: str
    subject: str
    aliases: tuple[str, ...]  # any of these, as a whole word (case-insensitive), counts as correct
    entity: str = ""          # RWKU: the person the item is about (never shown in a prompt)
    answer_type: str = ""     # RWKU: "year" | "number" | "text"; groups the specificity and chance controls


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



# Large set: the 100 facts above (same ids 0-99) + probe_facts.EXTRA (13 relations in total).
LARGE_FACTS = FACTS + [Fact(rel, s, tuple(a)) for rel, items in EXTRA.items() for s, *a in items]

FACT_SETS = {"pilot": PILOT_FACTS, "full": FACTS, "large": LARGE_FACTS}

_STOPWORDS = {"the", "and", "of", "for", "with", "from", "his", "her", "its", "was", "are", "a", "an", "in", "on", "at",
              "to", "by", "de"}


def name_derivable(fact: Fact) -> bool:
    """True if the canonical answer can be produced from the subject's spelling alone, without world knowledge.
      symbol:   the symbol abbreviates the name (helium -> He: first letters match, rest is a subsequence)
      prefix:   an answer word shares a 3+ letter prefix with a word of the subject (Germany -> German,
                diesel engine -> Diesel, Walmart -> Sam Walton)
      contains: (RWKU, whose subject is a whole sentence) an answer word of 3+ characters appears in the
                question, or shares a 5+ letter prefix with one of its words (Liv Tyler's father -> Steven Tyler)
      int:      the number is written in the subject"""
    rel = RELATIONS[fact.relation]
    ans, subj = fact.aliases[0].lower(), fact.subject.lower()
    if rel.kind == "int":
        return ans in re.findall(r"\d+", subj)
    if rel.spell == "symbol":
        if not subj.startswith(ans[0]):
            return False
        rest = iter(subj[1:])
        return all(ch in rest for ch in ans[1:])
    if rel.spell == "contains":
        words = set(re.findall(r"[a-z0-9]+", subj))
        return any(w in words or (len(w) >= 5 and any(x.startswith(w[:5]) for x in words))
                   for w in re.findall(r"[a-z0-9]+", ans) if len(w) >= 3 and w not in _STOPWORDS)
    return any(len(w) >= 3 and len(a) >= 3 and a[:3] == w[:3] for w in re.findall(r"[a-z]+", subj)
               for a in re.findall(r"[a-z]+", ans))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:48].rstrip("_") or "x"


def func_name(fact: Fact) -> str:
    return f"{RELATIONS[fact.relation].fn}_{_slug(fact.subject)}"


def _lit(s: str) -> str:
    """Text inside a double-quoted Python string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _doc(s: str) -> str:
    """Text inside a triple-quoted docstring."""
    s = s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return s + " " if s.endswith('"') else s


def build_prompts(fact: Fact) -> dict[str, str]:
    r, s = RELATIONS[fact.relation], fact.subject
    es, ea = r.example
    doc = _doc(r.doc.format(s=s))
    if r.kind == "int":  # integer literals, no quotes; the model writes the leading space itself
        code = {"dict": f'{r.table} = {{\n    "{_lit(es)}": {ea},\n    "{_lit(s)}":',
                "assert": f'assert {r.fn}("{_lit(es)}") == {ea}\nassert {r.fn}("{_lit(s)}") ==',
                "func": f'def {func_name(fact)}():\n    """{doc}"""\n    return'}
    else:
        code = {"dict": f'{r.table} = {{\n    "{_lit(es)}": "{_lit(ea)}",\n    "{_lit(s)}": "',
                "assert": f'assert {r.fn}("{_lit(es)}") == "{_lit(ea)}"\nassert {r.fn}("{_lit(s)}") == "',
                "func": f'def {func_name(fact)}():\n    """{doc}"""\n    return "'}
    return {
        "qa_zero": f"Question: {r.q.format(s=s)}\nAnswer:",
        "qa_para": f"Q: {r.q_para.format(s=s)}\nA:",
        "qa_one": f"Question: {r.q.format(s=es)}\nAnswer: {ea}\n\nQuestion: {r.q.format(s=s)}\nAnswer:",
        **code,
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


_INT_END = ",\n#)}]"


def _int_literal(text: str) -> str:
    """The expression the model wrote where an integer was expected, up to the end of the dict entry / line."""
    m = re.match(rf"[^{re.escape(_INT_END)}]*", text)
    return m.group(0).strip()


def extract_span(route: str, output: str, kind: str = "str") -> str | None:
    """The part of the output that carries the answer. None means malformed (unclosed literal)."""
    if route in ("dict", "assert", "func"):
        return _int_literal(output) if kind == "int" else _string_literal(output)
    return _first_line(output)


def is_complete(route: str, text: str, kind: str = "str") -> bool:
    """True once the generated text fixes the score: generating more cannot change extract_span, the func-route
    program, or the category. Used to stop generation early (run_recall_probe.py)."""
    if route in ("dict", "assert"):
        return any(c in text for c in _INT_END) if kind == "int" else ('"' in text or "\n" in text)
    if route == "func":  # the function body ends at the first unindented line
        return re.search(r"\n\S", text.replace("\r\n", "\n")) is not None
    return re.search(r"\S[^\n]*\n", text.replace("\r", "")) is not None  # first non-empty line is finished


def alias_pattern(alias: str) -> str:
    """Whole-word match: an alias edge that is a letter may not touch another letter, a digit edge may not touch
    another digit (so "79" does not match inside "1979", and "Au" does not match inside "August")."""
    left = r"(?<![0-9])" if alias[:1].isdigit() else r"(?<![A-Za-z])"
    right = r"(?![0-9])" if alias[-1:].isdigit() else r"(?![A-Za-z])"
    return left + re.escape(alias) + right


@lru_cache(maxsize=65536)
def alias_regex(aliases: tuple[str, ...], case_sensitive: bool = False) -> re.Pattern:
    return re.compile("|".join(alias_pattern(a) for a in aliases), 0 if case_sensitive else re.I)


def has_alias(span: str, aliases, case_sensitive: bool = False) -> bool:
    return bool(aliases) and alias_regex(tuple(aliases), case_sensitive).search(span) is not None


def is_garbage(span: str, min_word: int = 2, numbers_ok: bool = False) -> bool:
    """Malformed or degenerate text: no word of min_word+ letters (or, if numbers_ok, no digit either),
    mostly non-ASCII, or highly repetitive."""
    s = span.strip()
    if not re.search(rf"[A-Za-z]{{{min_word},}}", s) and not (numbers_ok and re.search(r"\d", s)):
        return True
    if sum(c.isascii() for c in s) / len(s) < 0.8:
        return True
    if len(s) >= 40 and len(zlib.compress(s.encode())) / len(s.encode()) < 0.35:
        return True
    return False


def func_program(fact: Fact, prompt: str, output: str) -> str:
    """Prompt + the completed function (up to the first unindented line), then a test of its return value."""
    rel = RELATIONS[fact.relation]
    body = re.split(r"\n(?=\S)", output.replace("\r\n", "\n"))[0]
    flags = 0 if rel.case_sensitive else 2  # 2 == re.IGNORECASE
    typ = "isinstance(_r, (int, str)) and not isinstance(_r, bool)" if rel.kind == "int" else "isinstance(_r, str)"
    pats = [alias_pattern(a) for a in fact.aliases]
    return (prompt + body + "\n\n"
            f"_r = {func_name(fact)}()\n"
            "import re\n"
            f"assert {typ} and any(re.search(p, str(_r), {flags}) for p in {pats!r})\n")


def categorize(route: str, fact: Fact, prompt: str, output: str, timeout: float = 10.0) -> tuple[str, str]:
    """Return (category, answer_span). For the func route, 'correct' requires the program to run and pass."""
    rel = RELATIONS[fact.relation]
    span = extract_span(route, output, rel.kind)
    if span is None:
        return "garbage", output[:120]
    if route == "func":
        if check_program(func_program(fact, prompt, output), timeout):
            return "correct", span
    elif has_alias(span, fact.aliases, rel.case_sensitive):
        return "correct", span
    if _REFUSAL_PAT.search(span):
        return "refusal", span
    literal = route in CODE
    if literal and span == "":
        # An empty string literal ("") is a well-formed empty answer; an integer slot left empty is malformed code.
        return ("garbage" if rel.kind == "int" else "fluent_incorrect"), span
    # A closed literal is well-formed, so a one-letter answer ("G" for gold) is a wrong answer, not garbage;
    # for numeric answers a wrong number ("80" for gold) is a wrong answer, not garbage.
    numeric = rel.kind == "int" or fact.answer_type in ("year", "number")
    return ("garbage" if is_garbage(span, 1 if literal else 2, numeric) else "fluent_incorrect"), span
