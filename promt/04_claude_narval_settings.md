# Task for Claude Code: apply the verified NARVAL settings on top of prompt 03

You are running unattended. Never ask questions or stop for confirmation. Finish every step,
then stop.

Prompt 03 (`prompts/03_claude_narval_protocol.md`, the first version) has already run. This
task is incremental: keep everything 03 built, verify it, and change only what is listed
below. Read CLAUDE.md, PLAN.md, DECISIONS.md, REPORT.md, REVIEW.md and
`git log --oneline -40` first, and find the `narval:` commits from 03.

## Rules
- Same rules as before: active interpreter only, one commit per change
  (`narval-04: <what>`), never `git push`, never rewrite history, never delete outside
  the repo.
- Do NOT change the method mathematics, loss, routing, targets or variant semantics.
  The ADPA/TVKD protocol (spec §11) must keep working unchanged.
- Settings marked VERIFIED come from the public NARVAL code
  (github.com/PhiVanDat123/NARVAL: recipes/configs/narval.yaml, scripts/run_narval.py,
  alignment/data.py, alignment/model_utils.py). Settings marked ASSUMED are declared
  starting values: store them in config with `"source": "assumed"`, list them in REPORT.md
  under "Open inputs (NARVAL)", and never present them as matching the paper.
- Anything still unknown stays `null` and fails closed in `validate-config`.

## Step 0: audit what 03 delivered
Check each item and record the result (done / partial / missing, with file and test name)
in a new REPORT.md section "Prompt 03 audit". Finish anything partial or missing before
Step 1, following the 03 text:
1. `tbpo-baseline` tag verified (or `tbpo-original` added) and recorded.
2. Batched KV-cached sampler with per-row generators; batched == unbatched test; resume
   test; throughput micro-benchmark logged.
3. Pre-registration in DECISIONS.md: collection/head shared across seeds (seed removed
   from their cache keys, with test), `correction_replicates = 1`, `states_per_prompt`
   with its causal selection rule and tests.
4. `diagnose-displacement` stage with the synthetic-tree convergence test.
5. `config/crpd/narval/` with three protocol files; UltraFeedback preparation;
   Qwen3 EOS/template preflight; tiny Qwen3-style model in tests and smoke.
6. Open LLM / MT-Bench / HH-RLHF plan templates; `report --layout narval`.
7. Mutation check extended and all mutations killed.

## Step 1: record the verified NARVAL facts (DECISIONS.md, new section "NARVAL code")
- Pipeline: SFT student; SFT teacher (= NARVAL `teacher_ref`); DPO teacher trained from
  that SFT teacher. NARVAL then trains the student on (prompt, chosen, rejected) pairs.
- NARVAL's own training settings (VERIFIED): beta 0.1, mu 0.97, lr 5e-7, 5 epochs, cosine,
  warmup_ratio 0.1, per-device batch 1, gradient accumulation 16, bf16, max_length 2048,
  max_prompt_length 1800, truncation keep_end, seed 42, DeepSpeed ZeRO-3 with CPU offload.
  These describe NARVAL; they are not CRPD settings.
- NARVAL's implicit reward is beta * sum w_t log(pi_theta / pi_teacher_dpo): its reference
  is the DPO teacher, not the student SFT. CRPD keeps rho = student SFT. Add this to the
  paper-disclosure list.
- Chat template (VERIFIED): alignment-handbook. A tokenizer without a chat template (Llama
  base checkpoints) gets DEFAULT_CHAT_TEMPLATE, Zephyr style, where every turn ends with
  tokenizer.eos_token. Qwen3 tokenizers keep their own template.
- Data conflict: the paper says UltraFeedback Binarized; the repository README says the
  DPO teacher uses DPO-Mix-7K; the config points to a private local path. Default to the
  paper. Not in the repository: SFT/DPO scripts and hyperparameters, evaluation scripts,
  lm-eval version, few-shot counts, MT-Bench judge.

## Step 2: change the NARVAL protocol configs
Apply these changes to the three files in `config/crpd/narval/`, and extend validation
and tests where a rule is new:

1. **Dataset field.** `preference_dataset` is a config field with allowed values
   `ultrafeedback_binarized` (default, `HuggingFaceH4/ultrafeedback_binarized`,
   `train_prefs` / `test_prefs`) and `dpo_mix_7k`. Teacher DPO, the student-DPO reference
   row, `crpd_anchor` and the prompt pool all read this one field. Test that changing it
   changes all four and nothing else.
2. **Lengths.** `max_length = 2048` and `max_prompt_length = 1800` (VERIFIED) for data
   preparation, teacher DPO and the student-DPO row. CRPD collection keeps `H = 512`.
   Prompts longer than 1800 tokens are dropped by rule before generation and their ids
   recorded. Disclosure line: NARVAL trains on responses up to 2048 minus prompt tokens,
   while CRPD collects up to 512.
3. **Recipes no longer null.** SFT recipe ASSUMED = ADPA/TVKD (Deita-10K-V0, 3 epochs,
   lr 2e-5, global batch 128). Teacher DPO and student-DPO recipe ASSUMED = ADPA/TVKD
   (beta 0.05, lr 5e-7, global batch 128, 2 epochs). `validate-config` accepts ASSUMED
   values but prints them as a warning list on every run.
4. **Checkpoints.** Qwen: `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-0.6B-Base`,
   `Qwen/Qwen3-8B-Base` (ASSUMED Base). Llama: `meta-llama/Llama-3.1-8B`,
   `meta-llama/Llama-3.2-1B` (ASSUMED base). Revisions stay `null` until pinned.
5. **Templates.**
   - Llama: set `chat_template` explicitly to the alignment-handbook
     DEFAULT_CHAT_TEMPLATE string (copy it verbatim into the config, with its source),
     so every role uses it and the assistant turn ends with `<|end_of_text|>`.
   - Qwen3: the tokenizer's own template with thinking disabled; `<|im_end|>` is the EOS
     for every role and in every saved generation_config.
6. **Qwen pilot provenance.** If 03 added a validate-config rule requiring the 1.7B
   teacher to come from the 8B -> 1.7B run, remove that requirement. Keep only a note: in
   NARVAL Table 1 the 1.7B teacher rows equal the 1.7B student SFT/DPO rows of the
   8B -> 1.7B setting. The pilot trains Qwen3-1.7B SFT then DPO directly; no 8B model.
7. **Pilot scale.** New field `max_prompts` (int or null): a fixed subset chosen by hashing
   prompt ids with the split seed, before any generation. Pilot values: `max_prompts =
   2000`, `states_per_prompt = 4`, `correction_replicates = 1`, one seed. Test that the
   subset depends only on ids and seed.
8. **Precision.** Qwen pilot: every role FP32. Llama and Qwen-8B configs: FP32 teachers
   on a separate device, as in DECISIONS item 20.
9. **CRPD settings** stay as DECISIONS item 9 (alpha .3, lr 1e-6, effective batch 128)
   and are tagged ASSUMED. The report prints NARVAL's budget (5 epochs over the train
   pairs) next to the CRPD step count and total trained tokens.

## Step 3: reported rows
Create `config/crpd/narval/reported.jsonl` with these rows from NARVAL Table 1, exactly as
printed. Columns: HellaSwag, ARC, MMLU, TruthfulQA, Winogrande, GSM8K, Avg. Each row gets
`source = "NARVAL Table 1"`, `reported = true`, and its pair id.

```text
pair: qwen3_1p7b_to_0p6b
DPO teacher  68.44 53.84 64.69 52.54 64.01 65.25 61.46
SFT teacher  67.47 52.90 64.90 49.52 64.40 64.37 60.59
SFT          53.08 37.63 50.59 45.92 59.51 47.84 49.09
DPO          53.74 39.08 50.80 46.92 59.43 52.46 50.41
WPO          52.81 37.80 50.33 47.24 59.51 53.36 50.17
Ra-DPO       52.42 36.24 48.99 45.92 59.43 52.42 49.24
DCKD         53.12 37.84 49.76 46.92 59.27 53.18 50.01
ADPA         54.10 39.33 50.98 46.11 59.27 53.22 50.50
TVKD         53.42 38.48 50.27 49.00 59.98 54.44 50.93
NARVAL       53.85 39.68 50.61 48.19 60.14 54.97 51.24

pair: llama31_8b_to_llama32_1b
DPO teacher  83.05 52.30 63.39 57.37 78.22 52.77 64.52
SFT teacher  81.85 51.20 63.05 53.59 78.45 47.84 62.66
SFT          65.84 39.24 31.08 41.13 62.90  7.28 41.25
DPO          66.94 39.25 31.21 44.57 62.12  7.88 42.00
WPO          67.44 40.70 31.21 42.45 62.27  6.22 41.72
Ra-DPO       66.14 39.40 31.02 41.61 62.83  7.71 41.45
DCKD         65.52 38.29 33.21 48.60 62.27  6.90 42.47
ADPA         65.87 38.57 32.85 51.46 62.27  6.88 42.98
TVKD         67.64 39.12 33.00 50.53 62.59  6.91 43.30
NARVAL       68.20 40.19 34.81 51.35 62.92  7.76 44.21
```

Add a test that recomputes each Avg from the six task scores and fails if any row differs
from the printed Avg by more than 0.01 (this catches transcription errors). The report
renders these rows separated from measured rows and never uses them in significance tests.

## Step 4: reference-row gate
Add `python -m chunkratio check-reference-rows --measured ... --reported ... --pair ...`.
It compares our measured SFT, DPO, SFT teacher and DPO teacher rows with the reported rows
and fails when any Avg differs by more than 0.5 points, printing the per-task differences.
`report --layout narval` refuses to put CRPD rows next to reported rows until this gate
has passed for that pair, unless `--allow-ungated` is given, which labels the table.

## Step 5: verification
- New mutations, all must be killed: dataset field ignored by one consumer; prompt length
  limit not applied; Llama template not set explicitly; `max_prompts` subset reading a
  score; reported-row Avg check disabled; reference-row gate bypassed.
- Smoke `--protocol narval-tiny` covers the dataset field, `max_prompts`, the reported
  rows and the gate (with a synthetic measured file).
- Run `python -m pytest -q`, both smoke scripts and the mutation check until all pass.

## Step 6: report
Update REPORT.md, PLAN.md and DECISIONS.md. Write the exact Linux CUDA command sequence
for the Qwen3-1.7B -> Qwen3-0.6B pilot:
1. prepare Deita-10K-V0 and UltraFeedback Binarized;
2. SFT Qwen3-1.7B-Base and Qwen3-0.6B-Base;
3. DPO the 1.7B SFT (teacher) and the 0.6B SFT (student-DPO reference row);
4. Open LLM eval of the four reference rows, then `check-reference-rows`
   (stop and report if it fails);
5. `diagnose-displacement` on 100 prompts, R = 8;
6. `crpd_main`, `crpd_raw`, `crpd_anchor` with one seed and 2000 prompts.
List every `null` and ASSUMED input. Final commit: `narval-04: settings complete`.
