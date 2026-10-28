# Forget Everything but Code

### Curvature-Guarded Exclusive Unlearning and an Execution-Verified Test of What Was Really Forgotten

**Prajak Sen** · October 2026 · [Paper (PDF)](paper.pdf) · [Model on Hugging Face](https://huggingface.co/Pras13/Qwen2.5-Coder-0.5B-CG-OSS)


## Abstract

Can a language model be taught to forget almost everything it knows while keeping one skill intact? We take Qwen2.5-Coder-0.5B and ask it to keep writing working programs while losing its general world knowledge. Our method, **CG-OSS** (Curvature-Guarded Orthogonal Subspace Scrubbing), pushes the model toward complete uncertainty on general text while steering its updates away from the directions that coding depends on. The unlearned model gives no informative answer to 100% of held-out general prompts and fails every direct factual question we ask, while keeping, on average over three training runs, **61% of the original HumanEval pass rate**, ahead of the same objective without curvature guidance (53%). We then introduce an **execution-verified, cross-route recall probe**: instead of only asking questions, we request the same fact through code and verify it by running the code in a sandbox. It shows that question-and-answer checks alone can overstate what a model has forgotten.

## The story in four steps

1. **The question.** A code assistant does not need to know the capital of France, yet every code model does. Could that knowledge be removed after training, leaving only the craft?
2. **The first lesson.** In a controlled pilot, popular forgetting objectives (gradient ascent, NPO) made the model *confidently wrong*, not uncertain. A max-entropy objective, which aims for a uniform prediction, produced real uncertainty.
3. **The method.** We added curvature guidance: measure, with the Fisher information, which directions coding relies on more than general text, and keep the forgetting updates out of them. Then we moved to a real model on a single 16 GB laptop.
4. **The sharper question.** When the unlearned model failed every direct question, we asked whether the knowledge was gone or the door was merely closed, and built a test that knocks on a different door: code.

## Key results

**Forgetting general knowledge while keeping code** (Qwen2.5-Coder-0.5B, HumanEval all 164 problems, mean of 3 training seeds)

| Method | No answer on general prompts | HumanEval Pass@1 | Share of original coding kept |
|---|---|---|---|
| Original model | 3% | 0.268 | 100% |
| NPO (1 seed) | 100% | 0.043 | 16% |
| Max-entropy | 100% | 0.142 | 53% |
| **CG-OSS (ours)** | **100%** | **0.165** | **61%** |

* Next-token entropy on general text reaches 11.89 nats, within 0.3% of a perfectly uniform prediction (11.93).
* Max-entropy is exactly CG-OSS without the projection, so the comparison isolates curvature guidance. CG-OSS is ahead in 5 of 6 seed and benchmark comparisons (HumanEval-40: 0.30 vs 0.24 mean, ahead on every seed).
* Coding scores vary between training runs; the best run (seed 0) keeps 86% of the original pass rate (0.232 vs 0.268).

**Curvature finds the right weights.** Zeroing the 5% of MLP weights with the highest forget-to-retain Fisher ratio raises general-text loss by **2.60** nats and code loss by only **0.14**, a 19 to 1 ratio. Random weights give 0.30 and 0.32.

**The code route.** After unlearning, direct questions retrieve **0 of 96** facts the original model knew. Asked through code, and verified by execution, **13 to 15%** of the facts whose answer cannot be spelled from the question come back, in **every one of six** unlearned models (3 seeds x 2 methods, p <= 0.004 each). For example, from CG-OSS seed 0: Tokyo, Moscow, New Delhi, Shakespeare, Fe, Na, Ag, K, Cu, W, and Spanish for Mexico.

```
Question: What is the capital of Japan?        ->  (noise)
assert capital("Japan") == "                    ->  Tokyo"
```

## How this compares with prior work

| Work | What is kept | Alternative route tested | Verified by |
|---|---|---|---|
| TOFU, MUSE | everything else | none | text overlap |
| Data transformations (Joshi et al., 2024) | everything else | MCQ, cloze, analogy | text match |
| Prompt attacks (Jang et al., 2025) | everything else | adversarial prompts | MCQ accuracy |
| Exclusive Unlearning (Sasaki et al., 2026) | medicine, math | jailbreaks | judged answers |
| ATU, K-Bench (2026) | everything else | tools, agent channels | answer leakage |
| **This work** | **code** | **code generation** | **sandbox execution** |

## Method at a glance

* **Objective.** `L = w_f * KL(uniform || p(. | general prompt)) + w_r * CE(code)`. The forgetting term is zero exactly when the model is perfectly uncertain.
* **Curvature guidance.** For each LoRA layer, solve `C_retain v = lambda (C_forget + eps I) v` on Kronecker-factored Fisher factors (true Fisher, sampled labels), keep directions with `lambda > 1`, and project both the forgetting gradient and the post-Adam update out of them.
* **Weight removal.** Zero the 5% of MLP weights per matrix with the highest `F_forget / F_retain`.
* **Recall probe.** 100 facts, 5 relations, 7 routes (3 direct questions, dictionary, assert, function return, comment). One label per response (correct, fluent but incorrect, garbage, refusal), with controls for defaults, chance, matched in-context examples and spellable answers. The verifier has its own tests.
* **Larger probe and an external benchmark** (results pending). `--facts large`: 895 facts in 13 relations (adds atomic numbers, event years, inventors, rivers, SI units, sports, product makers, company founders). `--facts rwku`: 5,223 questions and cloze sentences about 200 people from RWKU (CC BY 4.0), in the same 7 routes. The analysis reports recovery per relation and per route with Wilson 95% intervals, the recovery expected by chance, Holm-corrected tests, and pooling over seeds. How to run it on a free GPU: [docs/recall_probe_colab.md](docs/recall_probe_colab.md).

## Reproduce

Everything runs on one Apple M1 with 16 GB.

```bash
uv sync && uv run pytest                                   # includes the verifier audit

# train and evaluate (rows append to results/qwen2.5-coder-0.5b.csv)
uv run python run_qwen.py --model Qwen/Qwen2.5-Coder-0.5B --prune-alpha 0 --prune-frac 0.05 \
    --methods cg_oss max_entropy npo --seeds 0

uv run python run_humaneval.py                             # all 164 HumanEval problems
uv run python run_recall_probe.py --facts full             # 100 facts x 7 routes
uv run python analyze_recall_probe.py --tag full           # tables, controls, significance
uv run python run_recall_probe.py --facts large --models base cg_oss=Pras13/Qwen2.5-Coder-0.5B-CG-OSS   # 895 facts
uv run python run_recall_probe.py --facts rwku  --models base cg_oss=Pras13/Qwen2.5-Coder-0.5B-CG-OSS   # RWKU
uv run python analyze_recall_probe.py --tag large          # per-relation and per-route intervals
tectonic paper.tex                                         # the paper
```

## Repository map

```
exclusive_unlearning/   library: config, data, losses, K-FAC Fisher and projection, trainer, pruning,
                        evaluators, sandboxed code execution, recall probe and verifier
run_qwen.py             train + evaluate methods on Qwen (per seed)
run_humaneval.py        full HumanEval with pass / fail / timeout per problem
run_recall_probe.py     cross-route recall probe (pilot / full / large / rwku) on saved or Hub adapters
analyze_recall_probe.py probe analysis with all controls, per-relation and per-route intervals
build_rwku_probe.py     converts RWKU into data/rwku/rwku_probe.jsonl
docs/                   how to run the large probe and RWKU on Colab or Kaggle
prune_sweep.py          Fisher-selected vs random weight removal
make_qwen_video.py      results video from real outputs
results/                every CSV, raw generation and analysis behind the numbers above
tests/                  unit tests, including the recall-probe verifier audit
paper.tex               the paper (build with tectonic)
```

## Citation

```bibtex
@misc{sen2026forget,
  title  = {Forget Everything but Code: Curvature-Guarded Exclusive Unlearning and an
            Execution-Verified Test of What Was Really Forgotten},
  author = {Sen, Prajak},
  year   = {2026},
  url    = {https://github.com/prajak002/Forget}
}
```

## License

The adapter builds on Qwen2.5-Coder-0.5B, which is released under Apache 2.0.
