# Related Work: Machine Unlearning for Exclusive Task Retention

**Project:** *Exclusive Task Retention: Stripping General World Knowledge from Foundation Models While Preserving Specialized Code Execution Capabilities*

**Purpose of this document.** Survey the unlearning literature that our method (Curvature-Guarded Orthogonal Subspace Scrubbing, CG-OSS) builds on, state each family's mathematical limitation precisely, and identify our point of departure. Citation keys match `references.bib`.

---

## 1. Problem Framing

Let $\theta_0$ be a pretrained foundation model, $\mathcal{D}_f$ a *forget* distribution (general world knowledge: open-domain QA, chit-chat, factual recall) and $\mathcal{D}_r$ a *retain* distribution (code synthesis). Classical unlearning asks for $\theta^\ast$ such that the model behaves as if it were trained on $\mathcal{D} \setminus \mathcal{D}_f$. Our setting is inverted and more extreme: $\mathcal{D}_f$ is *the majority of pretraining*, and $\mathcal{D}_r$ is a narrow, well-specified capability. We call this **exclusive task retention**: the objective is not privacy-grade indistinguishability, but a model whose *only* surviving competence is the retained task.

This differs from standard unlearning benchmarks (TOFU, WMDP, MUSE) in three ways:

1. **Forget/retain volume ratio is inverted.** The forget set dominates; the retain set is small. Methods calibrated on "forget 1%, keep 99%" fail under "forget 95%, keep 5%".
2. **Entanglement is structural, not incidental.** Code generation depends on natural-language comprehension (docstrings, identifiers, problem statements). The forget and retain manifolds share subspaces by construction.
3. **The success criterion is behavioral, not statistical.** We measure refusal / high-entropy output on general queries and Pass@1 on code, not membership-inference AUC.

---

## 2. Exact and Sharded Unlearning

**SISA (Sharded, Isolated, Sliced, Aggregated)** [bourtoule2021sisa] partitions training data into $S$ shards, trains a constituent model per shard, and retrains only the affected shard on a deletion request. Unlearning is *exact* by construction.

*Limitation.* The cost is $O(|\mathcal{D}|/S)$ per request and requires access to the original training pipeline. For a foundation model whose pretraining corpus is the forget set, SISA degenerates to full retraining from scratch, which is exactly what we want to avoid. Moreover, SISA's ensemble aggregation degrades sample efficiency on the retain task, since each constituent sees only $1/S$ of the code data.

**Certified removal** [guo2020certified] offers $(\epsilon,\delta)$-style guarantees via Newton-step corrections on convex models. The convexity assumption is violated by transformers; the Hessian is intractable at the $10^9$-parameter scale.

---

## 3. Gradient-Based Approximate Unlearning

### 3.1 Gradient Ascent (GA)

The simplest approximate method [jang2023knowledge, yao2023llm_unlearning] maximises the forget-set loss:

$$\theta \leftarrow \theta + \eta \nabla_\theta \mathcal{L}_{\text{CE}}(\theta; \mathcal{D}_f).$$

*Limitation.* The objective is **unbounded**. As $p_\theta(y \mid x) \to 0$ on $\mathcal{D}_f$, $-\log p \to \infty$ and gradient norms explode. Empirically GA collapses the model within a few hundred steps (degenerate repetition, NaN losses). Because the gradient on $\mathcal{D}_f$ is not orthogonal to the retain gradient, collateral damage on $\mathcal{D}_r$ is severe. GA + retain-loss (GA+KL, GA+GD) [maini2024tofu] mitigates but does not solve this: the two gradients still fight in shared parameter directions.

### 3.2 Negative Preference Optimization (NPO)

NPO [zhang2024npo] reframes forgetting as DPO with only the *rejected* response:

$$\mathcal{L}_{\text{NPO}}(\theta) = -\frac{2}{\beta}\,\mathbb{E}_{(x,y)\sim\mathcal{D}_f}\Big[\log\sigma\Big(-\beta \log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\Big)\Big].$$

The $\log\sigma(\cdot)$ wrapper bounds the loss, and the gradient weight $\propto \sigma(\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}})$ vanishes adaptively as the forget probability drops. NPO is provably "slower than GA by an exponential factor" in divergence rate.

*Limitation.* NPO's gradient is still **GA's gradient with a scalar reweighting**: $\nabla \mathcal{L}_{\text{NPO}} = w(x,y)\,\nabla\log\pi_\theta(y\mid x)$. The *direction* in parameter space is unchanged, so entanglement with $\mathcal{D}_r$ is unchanged; NPO only decelerates the damage. The reference model $\pi_{\text{ref}}$ also anchors the forget distribution to its original form, so NPO pushes probability mass *away from* specific tokens rather than *toward* a well-defined target (uniform). This yields "shadow" knowledge: the model still ranks the true answer above random alternatives.

### 3.3 Target-driven objectives (Max-Entropy / Uniform-KL)

Rather than minimising likelihood, several works push the forget-set output distribution toward a target: uniform over the vocabulary (max-entropy) [chundawat2023badteacher, li2024wmdp_rmu], a refusal string ("I don't know") [eldan2023whos, maini2024tofu], or the distribution of a random label. The max-entropy objective is

$$\mathcal{L}_{\text{ME}}(\theta) = \mathbb{E}_{x\sim\mathcal{D}_f}\Big[ \mathrm{KL}\big(\mathcal{U}(V)\,\|\,\pi_\theta(\cdot\mid x)\big)\Big] = \mathbb{E}\Big[\log|V| - \tfrac{1}{|V|}\textstyle\sum_v \log \pi_\theta(v\mid x)\Big],$$

which is bounded below by zero and has a fixed point (uniform), so it does not diverge like GA. RMU [li2024wmdp_rmu] does the analogous thing in *activation* space (steer hidden states toward a random direction).

*Limitation.* These objectives fix the *target* but not the *path*. The gradient still lives in the full parameter space, and in our inverted regime (forget set >> retain set) the retain-loss term is overwhelmed. Nothing prevents the uniform-KL gradient from passing through subspaces that encode syntax, indentation and API-name recall.

---

## 4. Curvature-Aware and Subspace Methods

### 4.1 Fisher-based forgetting and pruning

Golatkar et al. [golatkar2020eternal, golatkar2021mixed] introduce Fisher forgetting: add Gaussian noise with covariance $\propto F^{-1}$ along directions the retain set does not constrain. **Selective Synaptic Dampening (SSD)** [foster2024ssd] scales parameters by the ratio $F_{\mathcal{D}_f}/F_{\mathcal{D}_r}$, dampening weights important to the forget set but not the retain set. **Fisher-pruning / WAGLE** [jia2024wagle] uses weight-attribution to select unlearning-relevant parameters.

*Limitations.* (i) These methods operate on the *diagonal* Fisher $\mathrm{diag}(F)$, which ignores cross-parameter correlation; a parameter can be individually unimportant to code yet jointly critical. (ii) They are *one-shot* (noise injection or dampening) rather than iterative, so they cannot handle a forget set as large as general pretraining. (iii) They ask "which weights encode the forget set", whereas we need "which *directions* are safe to move in."

### 4.2 Gradient projection in continual learning

The continual-learning literature solved the dual problem: learn new tasks *without* destroying old ones. **Orthogonal Gradient Descent (OGD)** [farajtabar2020ogd] projects new-task gradients onto the orthogonal complement of stored old-task gradients. **Gradient Projection Memory (GPM)** [saha2021gpm] computes a basis of the input-representation space for each layer from old-task activations and projects the new-task gradient onto its null space. **Adam-NSCL** [wang2021adamnscl] uses the null space of the uncentred covariance of layer inputs. **O-LoRA** [wang2023olora] enforces orthogonality between LoRA subspaces of successive tasks.

*Limitation for unlearning.* These methods are designed for *additive* learning and have been applied to unlearning only in small-scale settings [chen2023boundary]. Under max-entropy unlearning of a dominant corpus the projection residual is large and accumulates over many steps, so an *exact* null-space constraint either over-restricts (nothing is unlearned) or is relaxed heuristically. None uses a *curvature-weighted* projection: GPM/NSCL treat all retained directions as equally important.

### 4.3 Task-arithmetic and model-editing views

Task vectors [ilharco2023taskvectors] show unlearning can be done by *negating* a fine-tuning delta, which fails when the forgotten capability was never isolated in a delta (our case: it is pretraining itself). ROME/MEMIT [meng2022rome] edit specific facts via rank-one updates and do not scale to a whole knowledge distribution.

---

## 5. Sequential Unlearning and Re-entanglement

Recent work shows unlearning is often *superficial*. **Relearning attacks** [hu2024jogging, lynch2024eight] show that a handful of fine-tuning steps on a small related dataset restores "forgotten" knowledge. **Latent/shadow recovery** [patil2024sensitive, lo2024relearning] demonstrates that forgotten facts remain linearly decodable from intermediate activations. **Sequential unlearning** [shi2024muse, gao2024practical] shows that repeated forget requests re-entangle previously scrubbed knowledge with newly learned material, so utility degrades super-linearly with the number of requests.

The mechanistic explanation [lee2024mechanistic] is that gradient-based unlearning primarily modifies *late* layers and the unembedding (suppressing output), while the *representation* of the forgotten knowledge in earlier layers survives. Any method that evaluates only output-level forgetting will over-report success.

*Implication.* Our evaluation must include (a) a **relearning probe** (few-shot fine-tuning on held-out general text and measure recovery rate), and (b) a **linear-probe recovery** test on hidden states.

---

## 6. Benchmarks and Metrics

| Benchmark | Forget set | Retain set | Metric |
|---|---|---|---|
| TOFU [maini2024tofu] | Fictitious author QA | Real-author QA | Forget quality (KS-test), model utility |
| WMDP [li2024wmdp_rmu] | Bio/cyber hazard MCQ | MMLU | Accuracy drop vs. MMLU retention |
| MUSE [shi2024muse] | Harry Potter / BBC news | Related corpora | Verbatim mem., knowledge mem., privacy leak, utility |
| **Ours** | UltraFeedback prompts (general instruction data) | MBPP / HumanEval | Refusal rate, token entropy, Pass@1, relearning resistance |

Standard utility metrics (MMLU, perplexity on general text) are *inappropriate* for us: we want them to collapse. The retain metric must be functional (unit-test Pass@1), not likelihood-based, because likelihood of code is itself correlated with general-language likelihood.

---

## 7. Point of Departure: CG-OSS

Combining Sections 3.3, 4.1 and 4.2, our method makes three moves, each addressing a specific limitation above:

1. **Bounded target objective (vs. GA/NPO, §3.1–3.2).** We use the max-entropy KL objective $\mathcal{L}_{\text{ME}}$ on the forget set, so the forget loss has a fixed point and does not diverge. Unlike NPO we push toward a *target* (uniform), eliminating the rank-ordering shadow that reference-anchored losses leave behind.

2. **Curvature-guarded null-space projection (vs. GPM/NSCL, §4.2; vs. SSD, §4.1).** Before unlearning we compute a (low-rank, per-layer) empirical Fisher $F_r$ on the *retain* (code) set. At every unlearning step the forget gradient $g_f$ is projected:
   $$\tilde g_f = \big(I - U_k U_k^\top\big)\, g_f, \qquad U_k = \text{top-}k \text{ eigenvectors of } F_r,$$
   where $k$ is chosen so that $U_k$ captures a fraction $\tau$ of the retain-set Fisher trace. This is the direction-level analogue of SSD's weight-level test and uses curvature, not raw gradient memory, to decide which subspace is protected. In the low-rank LoRA setting the projection is applied in the adapter's input space, giving an $O(d\,k)$ cost per layer.

3. **Re-entanglement audit (vs. §5).** We report relearning resistance and linear-probe recovery, not just output refusal, and show that curvature-guarded updates leave less recoverable shadow than unprojected max-entropy or NPO because the forget signal is confined to subspaces the retain task does not use.

**Hypotheses to test (Phases 3–5).**
- H1: GA diverges; NPO and ME both forget, but ME reaches higher output entropy at the same retain Pass@1.
- H2: Adding the curvature-guarded projection to ME dominates the forget/retain Pareto front of all unprojected baselines.
- H3: Projected models show lower relearning recovery after $n$ fine-tuning steps on held-out general data.

---

## References (keys for `references.bib`)

- bourtoule2021sisa — Bourtoule et al., *Machine Unlearning*, IEEE S&P 2021.
- guo2020certified — Guo et al., *Certified Data Removal from Machine Learning Models*, ICML 2020.
- jang2023knowledge — Jang et al., *Knowledge Unlearning for Mitigating Privacy Risks in Language Models*, ACL 2023.
- yao2023llm_unlearning — Yao, Xu, Liu, *Large Language Model Unlearning*, 2023.
- maini2024tofu — Maini et al., *TOFU: A Task of Fictitious Unlearning for LLMs*, COLM 2024.
- zhang2024npo — Zhang et al., *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning*, COLM 2024.
- chundawat2023badteacher — Chundawat et al., *Can Bad Teaching Induce Forgetting?*, AAAI 2023.
- li2024wmdp_rmu — Li et al., *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning*, ICML 2024.
- eldan2023whos — Eldan & Russinovich, *Who's Harry Potter? Approximate Unlearning in LLMs*, 2023.
- golatkar2020eternal — Golatkar, Achille, Soatto, *Eternal Sunshine of the Spotless Net*, CVPR 2020.
- golatkar2021mixed — Golatkar et al., *Mixed-Privacy Forgetting in Deep Networks*, CVPR 2021.
- foster2024ssd — Foster, Schoepf, Brintrup, *Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening*, AAAI 2024.
- jia2024wagle — Jia et al., *WAGLE: Strategic Weight Attribution for Effective and Modular Unlearning in LLMs*, NeurIPS 2024.
- farajtabar2020ogd — Farajtabar et al., *Orthogonal Gradient Descent for Continual Learning*, AISTATS 2020.
- saha2021gpm — Saha, Garg, Roy, *Gradient Projection Memory for Continual Learning*, ICLR 2021.
- wang2021adamnscl — Wang et al., *Training Networks in Null Space of Feature Covariance for Continual Learning*, CVPR 2021.
- wang2023olora — Wang et al., *Orthogonal Subspace Learning for Language Model Continual Learning*, EMNLP Findings 2023.
- chen2023boundary — Chen et al., *Boundary Unlearning*, CVPR 2023.
- ilharco2023taskvectors — Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023.
- meng2022rome — Meng et al., *Locating and Editing Factual Associations in GPT*, NeurIPS 2022.
- hu2024jogging — Hu et al., *Jogging the Memory of Unlearned Models Through Targeted Relearning Attacks*, 2024.
- lynch2024eight — Lynch et al., *Eight Methods to Evaluate Robust Unlearning in LLMs*, 2024.
- patil2024sensitive — Patil, Hase, Bansal, *Can Sensitive Information Be Deleted From LLMs?*, ICLR 2024.
- lo2024relearning — Lo et al., *Large Language Models Relearn Removed Concepts*, ACL Findings 2024.
- shi2024muse — Shi et al., *MUSE: Machine Unlearning Six-Way Evaluation for Language Models*, 2024.
- gao2024practical — Gao et al., *Practical Unlearning for Large Language Models*, 2024.
- lee2024mechanistic — Lee et al., *A Mechanistic Understanding of Alignment Algorithms / Unlearning*, 2024.
- hu2022lora — Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
- cui2023ultrafeedback — Cui et al., *UltraFeedback*, 2023.
- austin2021mbpp — Austin et al., *Program Synthesis with Large Language Models*, 2021.
- chen2021humaneval — Chen et al., *Evaluating Large Language Models Trained on Code*, 2021.
- qwen2024coder — Hui et al., *Qwen2.5-Coder Technical Report*, 2024.
