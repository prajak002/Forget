"""Build the 'exclusive retention: Indian football' corpus from Wikipedia plain-text extracts.

retain : Indian football articles   -> data/indian_football/retain.jsonl
forget : general + hard-negative articles (world football, cricket, India in general, science, ...)
                                     -> data/indian_football/forget.jsonl
QA     : hand-written questions whose gold answer is checked to appear in the retain (or forget)
         corpus; unverifiable questions are dropped -> data/indian_football/qa.json

Each passage is ~120 words, tagged with its source article.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "data" / "indian_football"
UA = "exclusive-unlearning-research/0.1 (academic experiment)"

RETAIN = [
    "India national football team", "Indian Super League", "I-League", "All India Football Federation",
    "Sunil Chhetri", "Mohun Bagan Super Giant", "East Bengal FC", "Mohammedan SC (Kolkata)", "Bengaluru FC",
    "Kerala Blasters FC", "Mumbai City FC", "FC Goa", "Chennaiyin FC", "Durand Cup", "Santosh Trophy",
    "IFA Shield", "Federation Cup (India)", "Football in India", "Bhaichung Bhutia", "Syed Abdul Rahim",
    "Salt Lake Stadium", "India women's national football team", "Indian Women's League", "Super Cup (India)",
    "Odisha FC", "Hyderabad FC", "Jamshedpur FC", "NorthEast United FC", "Punjab FC", "Gurpreet Singh Sandhu",
    "Sandesh Jhingan", "2017 FIFA U-17 World Cup", "Football at the 1951 Asian Games",
    "Football at the 1962 Asian Games", "Chuni Goswami", "P. K. Banerjee", "Peter Thangaraj", "Igor Štimac",
    "SAFF Championship", "Calcutta Football League", "Dempo SC", "Churchill Brothers FC", "Salgaocar FC",
    "Kolkata Derby", "History of Indian football", "India national under-17 football team",
    "Fatorda Stadium", "Jawaharlal Nehru Stadium, Kochi", "Inter Kashi FC", "Gokulam Kerala FC",
]
FORGET = [
    "France", "Paris", "World War II", "Albert Einstein", "Photosynthesis", "Cricket", "Sachin Tendulkar",
    "India", "Taj Mahal", "Mahatma Gandhi", "Bollywood", "Premier League", "FIFA World Cup", "Lionel Messi",
    "Cristiano Ronaldo", "Real Madrid CF", "Manchester United F.C.", "Olympic Games", "Basketball", "Tennis",
    "Moon", "Sun", "Solar System", "DNA", "Computer", "Internet", "United States", "China", "Japan",
    "William Shakespeare", "Mona Lisa", "Leonardo da Vinci", "Isaac Newton", "Gravity", "Water",
    "Climate change", "Ancient Rome", "Egypt", "Great Pyramid of Giza", "Amazon rainforest", "Mount Everest",
    "Pacific Ocean", "Wolfgang Amadeus Mozart", "The Beatles", "Hollywood", "Economics", "Democracy",
    "Chess", "Coffee", "Tea", "Elephant", "Tiger", "Jupiter", "Neil Armstrong", "Great Wall of China",
]

# (question, [accepted answers]) -- kept only if an accepted answer appears in the right corpus.
QA_RETAIN = [
    ("Who is the all-time top goalscorer for the India national football team?", ["Chhetri"]),
    ("What is the name of the governing body of football in India?", ["All India Football Federation", "AIFF"]),
    ("In which year was the All India Football Federation founded?", ["1937"]),
    ("In which year was the Durand Cup first held?", ["1888"]),
    ("In which city is the Salt Lake Stadium located?", ["Kolkata", "Calcutta"]),
    ("Which club won the IFA Shield in 1911, becoming the first Indian team to win it?", ["Mohun Bagan"]),
    ("The Kolkata Derby is played between Mohun Bagan and which club?", ["East Bengal"]),
    ("In which year did the first Indian Super League season take place?", ["2014"]),
    ("Which country hosted the 2017 FIFA U-17 World Cup?", ["India"]),
    ("What is the name of the Indian inter-state football championship for state teams?", ["Santosh Trophy"]),
    ("Which Indian footballer is nicknamed the 'Sikkimese Sniper'?", ["Bhutia"]),
    ("Who coached India to the football gold medal at the 1951 and 1962 Asian Games?", ["Rahim"]),
    ("In which city are Kerala Blasters FC based?", ["Kochi"]),
    ("In which city is Chennaiyin FC based?", ["Chennai"]),
    ("Which Croatian manager was appointed head coach of India in 2019?", ["Stimac", "Štimac"]),
    ("Which regional football tournament for South Asian national teams has India won most often?", ["SAFF"]),
    ("In which city were the 1951 Asian Games held, where India won the football gold medal?", ["New Delhi", "Delhi"]),
    ("Which city is the home of the club Bengaluru FC?", ["Bengaluru", "Bangalore"]),
    ("In which state is the club FC Goa based?", ["Goa"]),
    ("What is the name of the top-tier women's football league in India?", ["Indian Women's League", "IWL"]),
    ("Which goalkeeper has been India's first-choice keeper and played for Bengaluru FC?", ["Gurpreet"]),
    ("Which Indian centre-back left Kerala Blasters in 2020 after a six-year association with the club?", ["Jhingan"]),
    ("In which year did India qualify for the FIFA World Cup but not participate?", ["1950"]),
    ("Which Kolkata club, founded in 1891, is one of the oldest football clubs in India?", ["Mohammedan"]),
]
QA_FORGET = [
    ("What is the capital of France?", ["Paris"]),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare"]),
    ("Which planet is the largest in the Solar System?", ["Jupiter"]),
    ("Who developed the theory of relativity?", ["Einstein"]),
    ("What is the capital of Japan?", ["Tokyo"]),
    ("Who painted the Mona Lisa?", ["Leonardo", "da Vinci"]),
    ("What is the highest mountain on Earth?", ["Everest"]),
    ("What is the largest ocean on Earth?", ["Pacific"]),
    ("Which country won the 2022 FIFA World Cup?", ["Argentina"]),
    ("Which country does Lionel Messi play for internationally?", ["Argentina"]),
    ("Which country does Cristiano Ronaldo represent internationally?", ["Portugal"]),
    ("Which sport is Sachin Tendulkar famous for?", ["cricket"]),
    ("In which city is the Taj Mahal located?", ["Agra"]),
    ("Who led India's non-violent independence movement and is called the Father of the Nation?", ["Gandhi"]),
    ("What is the capital of India?", ["New Delhi", "Delhi"]),
    ("Who was the first person to walk on the Moon?", ["Armstrong"]),
    ("What does DNA stand for?", ["deoxyribonucleic"]),
    ("Which city were The Beatles from?", ["Liverpool"]),
    ("In which country is the Great Pyramid of Giza?", ["Egypt"]),
    ("In which country is the Great Wall located?", ["China"]),
    ("Who formulated the laws of motion and universal gravitation?", ["Newton"]),
    ("What process do plants use to convert sunlight into chemical energy?", ["photosynthesis"]),
    ("Which club has won the most English Premier League titles?", ["Manchester United"]),
    ("Which Spanish club plays its home games at the Santiago Bernabéu?", ["Real Madrid"]),
]


RAW = OUT / "raw"


def fetch(title: str) -> str:
    """Fetch with a local cache and exponential backoff on HTTP 429."""
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / (re.sub(r"[^A-Za-z0-9]+", "_", title) + ".txt")
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text()
    delay = 2.0
    for attempt in range(6):
        try:
            txt = _fetch(title)
            cache.write_text(txt)
            time.sleep(1.0)
            return txt
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


def _fetch(title: str) -> str:
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                "titles": title, "format": "json"})
    req = urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        pages = json.load(r)["query"]["pages"]
    return next(iter(pages.values())).get("extract", "") or ""


def passages(text: str, words: int = 120, max_per_article: int = 14) -> list[str]:
    text = re.split(r"\n== (See also|References|External links|Notes|Further reading|Bibliography) ==", text)[0]
    text = re.sub(r"\n=+ [^=]+ =+\n", "\n", text)
    toks = text.split()
    out = [" ".join(toks[i:i + words]) for i in range(0, len(toks), words)]
    return [p for p in out if len(p.split()) >= 60][:max_per_article]


def build(titles, path):
    corpus, n = [], 0
    with open(path, "w") as f:
        for t in titles:
            try:
                txt = fetch(t)
            except Exception as e:  # report and continue; a missing article only shrinks the corpus
                print(f"  ! {t}: {e}")
                continue
            ps = passages(txt)
            if not ps:
                print(f"  ! {t}: empty")
            for p in ps:
                f.write(json.dumps({"article": t, "text": p}) + "\n")
                n += 1
            corpus.append(txt)
    print(f"{path.name}: {n} passages from {len(corpus)} articles")
    return "\n".join(corpus)


def verified(qa, corpus):
    """Keep a question only if some sentence contains an accepted answer AND a content word of the question.
    Prints the supporting sentence for manual review."""
    sents = re.split(r"(?<=[.!?])\s+", corpus)
    stop = set("what which who whom where when in on of the a an is was were to for and by did does has have from with as at its it this that how most first".split())
    keep, drop = [], []
    for q, ans in qa:
        qwords = [w for w in re.findall(r"[A-Za-z0-9']+", q.lower()) if w not in stop and len(w) > 3]
        support = next((s for s in sents for a in ans if a.lower() in s.lower() and sum(w in s.lower() for w in qwords) >= 2), None)
        if support:
            keep.append({"question": q, "answers": ans, "support": support[:300]})
            print(f"  KEEP {q}\n       -> {support[:200]}")
        else:
            drop.append(q)
            print(f"  drop {q}")
    return keep


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    r_corpus = build(RETAIN, OUT / "retain.jsonl")
    f_corpus = build(FORGET, OUT / "forget.jsonl")
    qa = {"retain": verified(QA_RETAIN, r_corpus), "forget": verified(QA_FORGET, f_corpus)}
    (OUT / "qa.json").write_text(json.dumps(qa, indent=2))
    print(f"qa: {len(qa['retain'])} retain, {len(qa['forget'])} forget questions")
