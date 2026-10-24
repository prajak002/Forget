# Literature check: recovering unlearned facts through other routes (24 Oct 2026)

Produced by a research subagent from web searches. **Citations are not yet verified:** [F] = abstract
fetched, [S] = search snippet only, [M] = from memory. Open every paper before citing it.

## Closest prior work
- [S] *Towards Robust Evaluation of Unlearning in LLMs via Data Transformations*, EMNLP Findings 2024, arXiv 2411.15477.
  Rewrites TOFU facts as MCQ, cloze, analogy and odd-one-out; models that "forgot" in Q&A answer in other formats.
  Closest format-shift precedent.
- [S] *Existing LLM Unlearning Evaluations Are Inconclusive*, 2025, arXiv 2506.00688. Recommends a cross-format
  leakage matrix.
- [S] Sun et al., *Unlearning vs. Obfuscation: Are We Truly Removing Knowledge?*, EMNLP 2025 (2025.emnlp-main.577).
- [S] Jang et al., *Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods*, arXiv 2506.10236.
- [F] Gupta et al., *Can LLMs Truly Forget? Revealing Unlearning Gaps Through Adversarial Evaluation*, arXiv 2608.21606.
  Multilingual reformulations leaked only ~3%, so recovery through another route is not guaranteed.
- [S] *Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility*, ICML 2026, arXiv 2606.03291.

## Agentic / tool routes
- [F] Chen et al., *Forgotten in Weights, Recovered by Tools: Agentic Tool Unlearning for LLM Agents*, arXiv 2608.21544.
- [F] Yu et al., *K-Bench: A Benchmark for LLM Unlearning in Agentic Deployments*, arXiv 2609.12808.
  Neither appears to use code generation or sandbox execution as the route.

## Code-domain unlearning (code is the forgotten content, not the route)
- [F] Patil & Sandoval, *Suppression is not Deletion: Adversarial Probes Recover Unlearned Knowledge in Code LLMs*,
  ICML 2026 workshop (CodeUnlearn-Bench).
- [S] *Mitigating Sensitive Information Leakage in LLMs4Code through Machine Unlearning*, arXiv 2502.05739.
- [S] PROD, *Large Language Model Unlearning for Source Code*, AAAI 2026, arXiv 2506.17125.

## Keep-one-domain ("exclusive") unlearning
- [F] Sasaki, Nakayama, Miyao, Oseki, Isonuma, *Exclusive Unlearning*, arXiv 2604.06154. Forgets everything
  except retained domains (medicine, math). **Must cite and differentiate**; check whether they evaluate code.
- [S] *Forget Narrowly, Retain Broadly*, arXiv 2607.09236.

## Benchmarks
- [S] OpenUnlearning (arXiv 2506.12618), RWKU (2406.10890), MUSE (2407.06460); [M] TOFU (2401.06121), WMDP (2403.03218).
  None found that verifies by executing code.

## Verdict (provisional)
No paper found that uses code completion (dict literal, assert, return value) as the route for extracting
general world facts forgotten in Q&A, or that verifies the extraction by execution. This rests on a handful of
searches, so it is not proof of absence. Re-search before any novelty claim, and check *The Unlearning Mirage*
(arXiv 2603.11266), which the subagent could not assess.
