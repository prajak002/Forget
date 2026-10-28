# Running the large recall probe and RWKU on Colab or Kaggle

This runs the expanded cross-route probe on the base model and the one published adapter,
**CG-OSS seed 0** (`Pras13/Qwen2.5-Coder-0.5B-CG-OSS`). Nothing is trained; the run only generates text and
scores it.

| Probe | Facts | Prompts per model | What it answers |
|---|---|---|---|
| `full` (sanity check) | 100, 5 relations | 700 | Does this setup reproduce the published result? |
| `large` | 895, 13 relations | 6,265 | Does code-route recovery hold on 500+ facts, per relation? |
| `rwku` | 5,223 RWKU items, 200 people | 36,561 | Does it hold on facts we did not write? |

Running time on a GPU has not been measured yet. Each run prints its progress and the time left.

## 1. Get the code onto the machine

The new code must be on GitHub (commit and push your branch), or uploaded as a zip.

**Colab** (Runtime → Change runtime type → T4 GPU). To keep results if the session drops, work inside Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
!git clone -b <your-branch> https://github.com/prajak002/Forget.git
%cd Forget
```

**Kaggle** (Settings → Accelerator → GPU T4; Settings → Internet → on):

```python
%cd /kaggle/working
!git clone -b <your-branch> https://github.com/prajak002/Forget.git
%cd Forget
```

## 2. Install

Colab and Kaggle already ship torch with CUDA, so do not install torch again.

```python
!pip install -q -U "transformers>=4.56" peft accelerate pandas scipy
!python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())"
```

The last line must end in `True`. Optional: `!pip install -q pytest && python -m pytest -q tests/test_recall_probe.py tests/test_recall_probe_large.py tests/test_analyze_recall_probe.py`
(58 tests; `tests/test_core.py` needs the training stack and can be skipped).

## 3. Sanity check: reproduce the published 100-fact result

```python
!python run_recall_probe.py --facts full --tag full_gpu --models base cg_oss=Pras13/Qwen2.5-Coder-0.5B-CG-OSS
!python analyze_recall_probe.py --tag full_gpu
```

The published run (Apple M1, bfloat16, one prompt at a time) gave, for `cg_oss` on the `non_derivable` facts:
**72 facts, 11 recovered, C_rec 0.153, McNemar p ≈ 0.001**, and the base model knew 96 of 100 facts.

On a GPU the script uses float32 and batches of 32, which can change a few greedy outputs. Expect numbers close to
those above, not identical. If they differ a lot, re-run this step with `--batch-size 1` (one prompt at a time) to
rule out padding effects, and write down which setting you used.

## 4. The large probe (895 facts, 13 relations)

```python
!python run_recall_probe.py --facts large --models base cg_oss=Pras13/Qwen2.5-Coder-0.5B-CG-OSS
!python analyze_recall_probe.py --tag large
```

## 5. RWKU (external benchmark)

```python
!python run_recall_probe.py --facts rwku --models base cg_oss=Pras13/Qwen2.5-Coder-0.5B-CG-OSS
!python analyze_recall_probe.py --tag rwku
```

For a quicker first look add `--limit 1500 --tag rwku1500`: a fixed random subset of 1,500 items, with the same
item ids as the full set.

## 6. If the session disconnects

Results are saved after each model. Run the same command again with `--resume` added and finished models are
skipped. On Colab this only works if you cloned into Drive (step 1). On Kaggle, download the `results/` files
before the session ends.

## 7. Bring the results back

```python
!zip -j probe_results.zip results/recall_probe_full_gpu_* results/recall_probe_large_* results/recall_probe_rwku*
```

Download `probe_results.zip`, unzip it into the repo's `results/` folder, and commit. Each run produces:

| File | Contents |
|---|---|
| `recall_probe_<tag>_responses.jsonl` | every prompt, raw output and category (lets the analysis be re-run anywhere) |
| `recall_probe_<tag>_meta.json` | device, dtype, batch size, library versions |
| `recall_probe_<tag>_analysis.md` | all tables (read this first) |
| `recall_probe_<tag>_by_relation.csv` | C_rec per relation with Wilson 95% interval, chance expectation, McNemar p, Holm p |
| `recall_probe_<tag>_by_route.csv` | recovery per relation and per code route (dict, assert, func, comment) with intervals |
| `recall_probe_<tag>_recovery.csv`, `_summary.csv` | the overall tables used in the paper |

## 8. What to read in `recall_probe_<tag>_analysis.md`

1. **Facts per relation**: how many facts were written, how many the base model knows through both direct
   and code routes, and how many of those are not spellable from the question. The `ALL` row of the last column
   is the number the issue's "500+ facts" refers to.
2. **Recovery by relation, `non_derivable` facts**: the main result. Each relation gets C_rec with a 95% interval,
   the recoveries expected by chance, and a p-value corrected for testing many relations.
3. **Recovery by relation and route**: which code form (dict, assert, function) carries the recovery.
4. **Every code-route recovery**: the actual recovered answers, to spot-check by eye.

The analysis can be re-run on your own computer from the `responses.jsonl` files alone (no GPU needed):
`uv run python analyze_recall_probe.py --tag large`.

## Later: when the other adapters are available

Run the new models under a separate tag, then analyse both tags together. The analysis adds a table pooled over
seeds (names `method@seed` are grouped by method):

```python
!python run_recall_probe.py --facts large --tag large_seeds --models \
    cg_oss@1=<repo-or-dir> cg_oss@2=<repo-or-dir> \
    max_entropy=<repo-or-dir> max_entropy@1=<repo-or-dir> max_entropy@2=<repo-or-dir> npo=<repo-or-dir>
!python analyze_recall_probe.py --tag large large_seeds
```

The same works for `--facts rwku`. Adapters saved locally under `checkpoints/<method>_seed<k>/final` can be
named without `=...` (for example `cg_oss@1`).
