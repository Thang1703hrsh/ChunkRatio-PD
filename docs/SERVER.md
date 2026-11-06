# Running CRPD on a multi-GPU server (2–4 × B200)

This guide covers `script/server/run_pair.py`, which runs CRPD outside Modal for the three NARVAL Table 1 pairs:

| `--pair` | T0 (SFT teacher) | T+ (DPO teacher) | rho (SFT student) | NARVAL comparison row |
|---|---|---|---|---|
| `qwen3_1p7b_to_0p6b` | `vukien2301/qwen3-1.7b-deita-sft-student` | `vukien2301/qwen3-1.7b-ultrafeedback-dpo-teacher` | `vukien2301/qwen3-0.6b-deita-sft-student` | `vukien2301/qwen3-0.6b-riskkd` |
| `qwen3_8b_to_1p7b` | `vukien2301/qwen3-8b-deita-sft-teacher` | `vukien2301/qwen3-8b-ultrafeedback-dpo-teacher` | `vukien2301/qwen3-1.7b-deita-sft-student` | `vukien2301/qwen3-1.7b-tailriskKD` |
| `llama31_8b_to_llama32_1b` | `vukien2301/llama-3.1-8b-deita-sft-teacher-epoch1` | `vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1` | `vukien2301/llama-3.2-1b-deita-sft-student` | `vukien2301/llama3.2-1b-tailriskKD` |

All checkpoints are the NARVAL authors' public repos listed in `riskKD_hf_checkpoints.xlsx`, pinned to exact
revisions in `chunkratio/riskkd.py`. In each pair the three models share one tokenizer, vocabulary and chat
template. The end-of-turn token (`<|im_end|>` for Qwen, `<|eot_id|>` for Llama) is the EOS of every role.

## 1. Set up the environments

The commands below are for Linux, run from the repository root.

```bash
# Training environment. B200 needs a CUDA 12.8 build of PyTorch.
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-crpd.txt huggingface_hub safetensors numpy

# Benchmark environment. Keep it separate: lm-eval needs transformers 4.56, training pins 4.52.4.
python -m venv ../venv-eval && ../venv-eval/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    && ../venv-eval/bin/pip install -r requirements-eval.txt

# Hugging Face login. Only the Llama pair needs it: its weight check downloads the gated
# meta-llama/Llama-3.1-8B and -Instruct, so accept their licence on the Hub first.
huggingface-cli login
```

Before any GPU work, check the plan:

```bash
python script/server/run_pair.py --pair qwen3_8b_to_1p7b --scale x10 --gpus 0,1,2,3 --gammas 0.1,0.05 --dry-run
```

On Windows, use `script/server/run_pair.ps1` with the same arguments.

## 2. Recommended order

α and γ were tuned only for the Qwen3 0.6B pair (α = 0.03, γ = 0.05 is the pinned best run). The 8B teachers
produce different gaps. Start each pair with a cheap x10 sweep, choose on the select split, check held-out
behaviour, then run the full data.

```bash
# 1) x10 sweep: one collection, each gamma trains on its own GPU at the same time
script/server/run_pair.sh --pair qwen3_8b_to_1p7b --scale x10 --gpus 0,1,2,3 --gammas 0.1,0.05,0.03,0.02

# 2) Pick the gamma with the best select RM among runs with held-out EOS >= 90%. Read the last log lines, or
#    output/server/<pair>/x10/runs/<name>/seed0/{diagnostics.json,heldout.json}.

# 3) Full data with that gamma, plus Winogrande and TruthfulQA against rho and the NARVAL student
script/server/run_pair.sh --pair qwen3_8b_to_1p7b --scale full --gpus 0,1,2,3 --gammas 0.05 \
    --stages prepare,verify,collect,train,heldout,benchmarks --eval-python ../venv-eval/bin/python
```

To test the same-source pair rule (DECISIONS 92), add `--pair-sources within`.

## 3. What each stage does

| Stage | GPUs | Notes |
|---|---|---|
| `prepare` | none | Pins revisions, downloads UltraFeedback, tokenizes it with the pair's template and selects the prompt subset. Also downloads the trio and the 8B selection RM into the HF cache |
| `verify` | none | Weight-space check that T0 is the nearest checkpoint to T+. It is recorded as an assumption with its evidence, not as a verification. Loads two 8B checkpoints at a time (about 32 GB RAM) |
| `collect` | all | One worker per GPU (`collection_shard`). Workers share one cache, and the union of their work is the single-process collection (tested) |
| `train` | one per γ | Each γ trains on its own GPU from the shared collection. RM checkpoint selection runs on the same 64 select prompts. Full scale first counts the pairs and sets `steps = passes × pairs / batch` (`--passes 1`, 3 selections) |
| `heldout` | one per run | The 64 held-out test prompts: RM vs rho, EOS, length, repetition, length-controlled RM. Writes `heldout.json` |
| `benchmarks` | all | Winogrande 5-shot and TruthfulQA-MC2 0-shot in hf fp32, with paired tests against rho and NARVAL. Writes `benchmarks/summary.md` |

Every stage can be resumed. Shards, targets and the policy checkpoint are saved as the run goes, so after an
interruption you rerun the same command.

## 4. Resources and expected time

These are **estimates**, scaled from the A100-80GB runs measured on Modal. The assumption that a B200 is about
2× an A100 for this workload has not been measured, so run x10 first and read
`diagnostics.json → cost.stage_seconds`.

| Pair, full data (about 55k prompts, 1 pass) | Collection (4 GPUs) | Training (1 GPU per γ) | Total wall time |
|---|---:|---:|---:|
| Qwen3-1.7B → 0.6B | about 1.5 h | about 3 h | about 5–6 h |
| Llama-3.1-8B → 3.2-1B | about 5 h | about 5 h | about 11–14 h |
| Qwen3-8B → 1.7B | about 5.5 h | about 8 h | about 15–18 h |

An x10 run (640 prompts) takes roughly 15–45 minutes depending on the pair.

### Memory plan (automatic)

`run_pair.py` reads each GPU's memory with `nvidia-smi` (override with `--gpu-memory-gb`) and fills it. The plan
is printed at start, and it changes no result beyond FP32 rounding.

| Per GPU (≈180 GB B200) | 0.6B pair | Qwen 8B → 1.7B | Llama 8B → 1B | A100-80GB (measured) |
|---|---:|---:|---:|---:|
| Generation tokens per batch | 450,560 | 212,992 | 245,760 | 163,840 |
| Scoring tokens per batch / states | 45,056 / 128 | 24,576 / 80 | 28,672 / 80 | 16,384 / 48 |
| Training processes per GPU | **2** | 1 | **2** | 1 |

- **Collection:** the batch budgets grow with the memory left after the three FP32 models. Two FP32 8B
  teachers need about 98 GB per GPU, so the 8B pairs refuse to run on 80 GB GPUs.
- **Training:** the policy microbatch stays at the measured 8,192 tokens, because 12,288 was slightly slower on
  an A100. The spare memory runs **two γ per GPU** where they fit, since GPU utilisation while training was about
  33%. Override with `--trainings-per-gpu`.
- **Host RAM:** each training process drops the two teachers once the targets exist (`release_teachers`), so it
  no longer keeps 64 GB of FP32 8B teachers in RAM. Plan for about 40 GB per training process, mostly the
  selection RM.
- **Out-of-memory fallback:** the plan for B200 is computed, not measured. If a collection worker runs out of
  memory, its budgets are halved and it resumes from its shards. If a training process runs out with two per GPU,
  it is rerun alone on a GPU.

`--policy-tokens` and `--generation-tokens` override the plan.

## 5. Output layout

```
output/server/<pair>/
  data/            revisions.json, raw UltraFeedback, prepared and subset prompts (per row count)
  teacher_reference_evidence.json
  <scale>/configs/ resolved configs (one per gamma, one per collection worker)
  <scale>/cache/   collection shards, targets, policy checkpoints (shared by every gamma)
  <scale>/runs/<variant>-alpha..-gamma..-<pair_sources>/seed0/
                   student/ (HF format), diagnostics.json, heldout.json, manifest.json
  <scale>/logs/    one log per worker and per run
  <scale>/benchmarks/summary.md
```
