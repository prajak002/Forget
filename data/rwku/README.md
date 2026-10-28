# RWKU items for the cross-route recall probe

`rwku_probe.jsonl` holds the forget probes of **RWKU** (Jin et al., 2024, *RWKU: Benchmarking Real-World
Knowledge Unlearning for Large Language Models*, NeurIPS Datasets and Benchmarks), converted for
`run_recall_probe.py --facts rwku`. Source: <https://huggingface.co/datasets/jinzhuoran/RWKU>, files
`All/forget_level1.json` (cloze) and `All/forget_level2.json` (questions). License: CC BY 4.0.

Changes from the source (rules in `exclusive_unlearning/rwku.py`, script `build_rwku_probe.py`):

- kept one-blank cloze sentences and plain questions; dropped the 2 multiple-choice items, cloze sentences
  without exactly one `___`, answers longer than 5 words, and duplicates;
- added accepted answer variants (without a leading article or trailing parenthetical, the part before a colon,
  numbers without thousands separators);
- labelled each answer `year`, `number` or `text`, which groups the specificity and chance controls.

One line per item: `relation` (`rwku_qa` or `rwku_cloze`), `subject` (the question or cloze sentence),
`aliases`, `answer` (original), `entity` (the person), `answer_type`, `level`.
