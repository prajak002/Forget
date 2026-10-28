"""Build data/rwku/rwku_probe.jsonl: RWKU forget probes (levels 1 and 2) in recall-probe form.

  python build_rwku_probe.py                    # download from the Hugging Face Hub
  python build_rwku_probe.py --raw-dir DIR      # use local forget_level1.json / forget_level2.json

RWKU: Jin et al., 2024, https://huggingface.co/datasets/jinzhuoran/RWKU (CC BY 4.0). Conversion rules are in
exclusive_unlearning/rwku.py. The output is committed, so running the probe does not need this script.
"""
import argparse
import collections
import json
import urllib.request
from pathlib import Path

from exclusive_unlearning.config import ROOT
from exclusive_unlearning.recall_probe import RELATIONS, name_derivable
from exclusive_unlearning.rwku import load_rwku, rwku_items

URL = "https://huggingface.co/datasets/jinzhuoran/RWKU/resolve/main/All/{}.json"
OUT = ROOT / "data" / "rwku" / "rwku_probe.jsonl"


def fetch(name: str, raw_dir: Path | None) -> list[dict]:
    if raw_dir:
        return json.loads((raw_dir / f"{name}.json").read_text())
    with urllib.request.urlopen(URL.format(name), timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    l1, l2 = fetch("forget_level1", a.raw_dir), fetch("forget_level2", a.raw_dir)
    items = rwku_items(l1, l2)
    for it in items:  # the one-shot examples must not be probed
        assert it["subject"] != RELATIONS[it["relation"]].example[0]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    facts = load_rwku(a.out)
    c = collections.Counter((f.relation, f.answer_type) for f in facts)
    spell = sum(name_derivable(f) for f in facts)
    print(f"read {len(l1)} level-1 + {len(l2)} level-2 probes -> {len(items)} items, "
          f"{len({f.entity for f in facts})} people, {spell} spellable from the question")
    for k, v in sorted(c.items()):
        print(f"  {k[0]:10} {k[1]:6} {v}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
