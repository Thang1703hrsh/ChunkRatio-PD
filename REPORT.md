# ChunkRatio-PD v1.1 implementation report

Started: 26 October 2026; completed: 27 October 2026 (Asia/Saigon).
Specification: revision 2. All reported model experiments
below are **synthetic CPU verification**, not language-model benchmarks.

## Delivery and implementation

Implemented the offline CRPD pipeline under `chunkratio/`: pure tensor ratios,
suffix labels, exact routing, hierarchical pair weights and soft BCE; independent
FP64 finite-tree algebra; canonical preflight, shared causal scoring and detached
reference features; raw categorical prefix/sibling/correction collection;
atomic versioned resumable shards; scalar mu-head fitting and head_val selection;
frozen endpoint targets; hierarchical student training, microbatch accumulation,
checkpoint selection/resume and student export. Seven variants have explicit
configs. Audit is diagnostic and takes zero student optimization steps.

The staged CLI is `python -m chunkratio`. `train.py --crpd` dispatches to it before
legacy Unix/FSDP imports. The legacy invocation and TBPO-Q/A source are preserved.
This is a separate offline training loop, not an extension of the inherited
chosen/rejected batch loop or a Hydra CRPD loss registry entry. See DECISIONS.md.

Added revision-2 preparation configs and single-process SFT/teacher-DPO launchers,
canonical local data preparation (including HelpSteer helpfulness tie removal),
both preference tracks / three-seed run generation, held-out FsfairX RM evaluation,
external benchmark command plans, protocol-checked tables, prompt-paired bootstrap
intervals, training-seed variation, candidate coverage, token accounting, conditional
cost estimates and Windows/Bash launchers. External benchmark plans are templates
requiring actual pinned installations/configuration, not verified full adapters.

`AGENTS.md` and `CLAUDE.md` are identical. `reference/HASHES.txt` records the actual
supplied documents/checks, and their Git blob hashes match working-file hashes.
The absent `tbpo-baseline` tag was created at starting HEAD before shared edits.
Local step commits were made throughout. No push, history rewrite, large-model
download, or full-scale training was performed.

## Environment and commands actually executed

Active interpreter throughout: **Python 3.15.1, Windows**. Stable PyPI had no
compatible torch wheel; the official CPU nightly index supplied
**torch 2.16.0.dev20261026+cpu**. Pytest is 9.1.1 and PyYAML is 6.0.3.
`requirements-crpd-cpu-tested.txt` freezes the successful CPU environment.
The original `env/train_env.yml` instead pins Python 3.11, torch 2.6.0+cu124,
transformers 4.52.4, datasets 3.5.0, accelerate 1.7.0, peft 0.14.0 and Hydra 1.3.2.
Codex only inspected those research dependencies. The review then installed
numpy 2.5.4 and transformers 4.52.4 (tokenizers 0.21.4, safetensors 0.8.0) into
the same interpreter. With them it ran the non-synthetic code path on tiny,
randomly initialised local Llama checkpoints (`tests/test_hf_runtime.py`).
datasets, accelerate, peft and Hydra are still not installed.

Install commands used in the active environment:

```text
python -m pip install pytest pyyaml
python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

The review installed NumPy, which removed the torch NumPy initialization warning.
The nightly wheel is a reproducibility limitation; check stable versions in a
compatible research environment before scaling.

Final verification after the review (27 October 2026), with summary lines.
REVIEW.md lists the review fixes, so the counts differ from Codex's run
(32 tests):

```text
python reference/ChunkRatio_Math_Checks.py output/review/reference-checks.json
ALL CHECKS PASSED

python -m pytest -q
53 passed in 27.12s

script/smoke_test.ps1 -Output output/smoke-review-final        (Windows PowerShell 5.1)
SMOKE PASSED: 7 variants, all stages, export/reload and standalone generation

bash --noprofile --norc script/smoke_test.sh --output output/smoke-review-bash
SMOKE PASSED: 7 variants, all stages, export/reload and standalone generation

python -m chunkratio learned-tree --output output/review/learned-tree.json
LEARNED TREE COMPLETED: synthetic finite-data experiment   (numbers identical to the table below)
```

Codex's original runs, kept for the record:

```text
python reference/ChunkRatio_Math_Checks.py
ALL CHECKS PASSED

python -m chunkratio toy-check --output output/reference-checks.json
ALL CHECKS PASSED

python -m pytest -q
32 passed in 8.05s

python -m chunkratio smoke --output output/smoke-final
SMOKE PASSED: 7 variants, all stages, export/reload and standalone generation

python -m chunkratio learned-tree --output output/learned-tree.json
LEARNED TREE COMPLETED: synthetic finite-data experiment

powershell -NoProfile -ExecutionPolicy Bypass -File script/smoke_test.ps1 -Output output/smoke-powershell
SMOKE PASSED: 7 variants, all stages, export/reload and standalone generation

bash --noprofile --norc script/smoke_test.sh --output output/smoke-bash
SMOKE PASSED: 7 variants, all stages, export/reload and standalone generation
```

Codex reported that the PowerShell launcher first hit the machine's disabled-script
policy. In the review session the process policy was already Bypass, so the
launcher ran directly. The review also fixed a launcher bug: redirecting stderr
(`2>&1`) used to abort the run. No persistent policy was changed. Bash runs used
Git Bash on Windows. Native Linux and CUDA have not been run.

Earlier test runs skipped tensor modules before installing torch; those are
superseded by the complete runs above (32 tests by Codex, 53 after review), with no skipped modules. Both a
completed checkpoint and an interrupted/extended partial head/policy run are
compared against uninterrupted execution. `python -m compileall -q chunkratio
tests script/run_crpd_protocol.py` and `git diff --check` also passed.

Portable summary: `reports/crpd_validation_summary.json`. Full local logs and
artifacts: `output/verification/`, `output/reference-checks.json`,
`output/smoke-final/`, `output/learned-tree.json` (intentionally gitignored).

## Verification table from specification section 12

| Required observation | Implementation | Executed test / evidence |
|---|---|---|
| Both residuals, Q identity, KL identities, bounds, regimes | `chunkratio/tree.py` | `tests/test_tree_reference.py::test_fp64_tree_identities_regimes_and_routing`; independent torch expectations run through exhaustive reference checks, compare all JSON fields <=1e-10 |
| Variable EOS stopping, action mass, partition invariance | `tree.py` | `test_variable_eos_action_mass`; partition invariance also in the preceding exhaustive identity test |
| Causal shift, prompt/PAD/post-EOS mask, sampled EOS once | `scoring.py` | `tests/test_scoring.py::test_shift_mask_prompt_padding_eos` |
| Padding/batching including EOS=PAD | `scoring.py`, `tiny.py` | `test_batching_features_causality_and_frozen_gradients` |
| Prefix feature causality | `scoring.py::state_features`, `trajectory_features` | Same test compares prefix-only hidden states with extended sequences; one-pass trajectory features equal per-prefix features (TinyLM and HF Llama, `tests/test_hf_runtime.py`) |
| Suffix indexing / endpoint excludes past chunk | `math.py::mu_targets`, `collection.py` | `tests/test_math.py::test_suffix_endpoint_exclusion`; `tests/test_collection.py::test_correction_rollouts_labels_and_features` recomputes labels, features and budgets independently for rho (main) and T+ (mu_teacher) rollouts |
| Global H, chunk boundary nonterminal | `sampling.py`, `collection.py` | `tests/test_pipeline.py::test_global_horizon_raw_sampling`; `tests/test_collection.py::test_candidate_budget_is_global_and_chunk_cutoff_nonterminal` (H-n < L) |
| Untruncated categorical sampling | `sampling.py::generate` | Same test: 3000 controlled categorical draws match declared probabilities |
| Pair logit .24, p .55971365, loss log2, derivative -.05971365 | `math.py` | `tests/test_math.py::test_numeric_fixture`; nonzero margin (alpha on the student side) in `test_nonzero_margin_alpha_on_both_sides` |
| Frozen target t_s(c)=delta_T+mu_hat, terminal mu_hat=0, endpoint budget | `math.py::target_score`, `targets.py` | `tests/test_targets.py::test_target_score_function`, `test_build_targets_sign_correction_terminal_and_budget` |
| Pair swap/common offset/soft targets | `math.py` | `test_symmetry_offset_and_frozen_targets`; label bound/entropy floor in `test_label_error_and_entropy_floor` |
| Policy changes only student; head changes only head; frozen roles under no_grad | `training.py`, `value_head.py`, `scoring.py`, `sampling.py` | `tests/test_pipeline.py::test_collection_resume_and_gradient_routing`; `tests/test_head.py::test_head_gradient_and_resume`; `tests/test_scoring.py::test_frozen_roles_always_run_under_no_grad` (records autograd mode inside every forward) |
| Hierarchical/full/accumulated reductions, fixed H regression | `math.py`, `targets.py`, `training.py`, `value_head.py` | `tests/test_math.py::test_weights_and_accumulated_reductions`; `tests/test_targets.py::test_pair_weights_are_hierarchical`, `test_sampler_matches_hierarchical_weights`; `tests/test_head.py::test_feature_shards_and_minibatch_head`; the policy microbatch loop normalizes by the total sampled batch |
| Null update / shared reference bypass | `math.py::routing`, `targets.py`, `training.py` | `test_exact_routes`; `tests/test_pipeline.py::test_no_alignment_update_and_exact_zero` |
| Metadata invalidation, immutable atomic shards, no duplicates | `artifacts.py`, `collection.py` | `tests/test_artifacts.py::test_resume_integrity_and_incompatible_metadata`, `test_stage_identity`; actual collection repeat in pipeline test |
| Optimizer/scheduler/RNG/sampler partial resume | `training.py`, `value_head.py` | Partial one-step/one-epoch execution extended to full run in pipeline/head tests; histories and model fingerprints match |
| TBPO-Q/A unchanged | inherited sources + opt-in dispatch | `tests/test_baselines.py` exact baseline-source checks; `tests/test_tbpo_runtime.py::test_tbpo_q_a_tiny_forward_equivalence`: original/current tiny forwards and one inherited SBA/RMSprop/AdamW update agree bit-for-bit |

The baseline runtime fixture extracts original functions from the immutable tag
and current functions without optional logger/FSDP imports. It verifies tiny
forwards and an optimization step. It does **not** establish a full legacy
Hydra/FSDP training job on real data.

Additional executed checks cover duplicate split leakage/tokenizer mismatch,
source hashes, strict config/variant validation, portable CLI dispatch, HelpSteer
ties, reporting rejection of synthetic or incomparable outcomes, paired bootstrap
intervals, token accounting, protocol constants, docstrings and annotations.

## Synthetic learned finite-data experiment

Finite reference trajectories: 16 per state for training and a separate 16 for
head validation. Both regimes use a fitted scalar head and an actual trainable
tabular autoregressive policy optimized with soft pair BCE for 200 steps.
Every state record includes exact phi, exact mu_rho and fitted mu_hat.

| Regime | head_val SSE/H | Max sibling mean error | KL(target learned correction \| pi*) | KL(fitted policy \| pi*) | KL(fitted \| target) | Excess pair BCE |
|---|---:|---:|---:|---:|---:|---:|
| Small displacement / large mismatch | 0.00176903 | 0.0394639 | 9.59561e-5 | 9.59685e-5 | 1.96789e-12 | 6.67244e-14 |
| Large displacement / small mismatch | 0.374711 | 0.594565 | 0.118012 | 0.118011 | 1.87103e-10 | 9.36942e-11 |

These numbers separate policy-fitting error from target error. They do not show
universal improvement or language-model performance. Independent validation
trajectories share enumerable state support; this is not unseen-prompt
generalization. The restricted token-0/1 graph disclosure reports its unidentified
reference EOS mass at the root; the full-support policy is the fitted experiment.

Smoke main has 15 prefix states, 54 training pairs and 31 correction trajectories.
Main/raw/teacher-mean/anchor/audit have 54 pairs, suffix has 51, token has 79.
The six training variants each take 3 steps; audit takes 0. Every export is
reloaded, its scores match, and it generates using just the student.

## Reproduction commands for a real main table

The following is the exact command sequence supported by this implementation,
**after resolving the inputs listed in Not done**. No real main table exists
yet. Preparation configs deliberately contain null model IDs/revisions until
verified inputs are supplied; validation rejects unresolved configs.

1. Obtain versioned local exports of Deita-10K-V0, DPO-MIX-7K and HelpSteer2.
   Create canonical data using a pinned shared tokenizer and assign duplicate
   groups to splits before collection. `prepare-data` accepts exported chat pairs
   or raw helpfulness comparison records; Deita requires its SFT answer exported
   as chosen_ids. Preserve source versions, filters and split manifests.

```text
python -m chunkratio prepare-data --input data/dpo_mix_export.jsonl --tokenizer MODEL --revision COMMIT --output data/dpo_mix_7k.jsonl
python -m chunkratio prepare-data --input data/helpsteer_export.jsonl --tokenizer MODEL --revision COMMIT --output data/helpsteer2.jsonl
```

2. Resolve the three preparation configs, then reproduce/verify the teacher pair
   and student SFT. These commands are provided, not run at scale:

```text
python -m chunkratio train-sft --config config/crpd/sft_teacher.json
python -m chunkratio train-sft --config config/crpd/sft_student.json
python -m chunkratio train-teacher-dpo --config config/crpd/dpo_teacher.json
```

3. Fill actual checkpoint paths/revisions, teacher-reference provenance and pinned
   reward-model revision in `config/crpd/protocol.json`. Use the same predeclared
   select/RM generation rule for every method. Generate/run both tracks and
   three training seeds; no such command was executed during this task:

```text
python -m chunkratio validate-config --config config/crpd/protocol.json
python script/run_crpd_protocol.py --config config/crpd/protocol.json --dpo-prompts data/dpo_mix_7k.jsonl --helpsteer-prompts data/helpsteer2.jsonl --variants crpd_main crpd_raw --execute
```

4. For each exported student, use its generated input_config.json to run held-out
   RM generation/scoring. Example for the first primary run:

```text
python -m chunkratio rm-evaluate --config output/protocol/DPO-MIX-7K/crpd_main/0/input_config.json --model output/protocol/DPO-MIX-7K/crpd_main/0/student --output output/eval/main_dpo_seed0/rm.jsonl
python -m chunkratio benchmark-plan --model output/protocol/DPO-MIX-7K/crpd_main/0/student --model-id crpd-main-dpo-seed0 --judge gpt-4o-mini --adpa-outputs output/adpa/dpo_seed0/alpaca_outputs.json --output output/eval/main_dpo_seed0/plan.json
```

Supply frozen official generation/annotator configs, record judge identities,
pin the plan's external dependency versions, install their official packages,
and generate benchmark output files before invoking:

```text
python -m chunkratio benchmark-run --plan output/eval/main_dpo_seed0/plan.json --output output/eval/main_dpo_seed0/execution.json
```

Repeat for all tracks/seeds/methods. The default Open LLM plan declares zero-shot
ARC-Challenge, HellaSwag, TruthfulQA-MC2, MMLU, Winogrande and GSM8K; report this
choice and replace with a preregistered alternate only consistently across all
methods. MT-Bench uses inherited single-answer scoring. AE-vs-ADPA requires the
same primary judge; AE2 and Arena-Hard retain their official configurations.

5. Retrain the actual external baselines, prepare the teacher DPO/SFT and student
   SFT/DPO reference rows, and normalize official outcomes into measured JSONL
   rows with `method`, `dataset`, `seed`, `synthetic=false`, `split=test`,
   `metrics`, per-metric `protocols` fingerprints and `cost.gpu_hours`.
   Required metric names are in `chunkratio/evaluation.py::METRICS`.

```text
python -m chunkratio report --input output/eval/measured_results.jsonl --output output/eval/main_table.md
```

The strict report rejects missing reference/baseline rows, fewer than three
main/raw seeds, incomplete metrics, mixed protocols and synthetic/selection
data. `--allow-incomplete` explicitly labels a partial report. `paired_interval`
computes prompt-paired intervals separately from the training-seed SD table.
Automatic normalization of every upstream results schema and external baseline
training commands remain Not done, so this is not yet an unattended benchmark
reproduction pipeline.

## Compute and GPU-hour estimate

Collection artifacts count generated/scored response tokens by model; diagnostic
reports also count actual forward/prefill input tokens, policy gradient forward
tokens, selection generations and RM scoring tokens. Wall time and peak CUDA
allocated bytes are recorded. CPU smoke uses no GPU (peak CUDA allocation 0).
Full experimental GPU-hours, end-to-end tuning and preparation costs are not
measured; do not interpret invocation wall time as total GPU-hours.

Conditional collection planning command:

```text
python -m chunkratio estimate-cost --states 1000 --trajectories 2000 --average-suffix 256 --generation-rate 100 --scoring-rate 1000
```

At those **hypothetical aggregate rates**, 640,000 generated tokens plus
1,408,000 scored tokens give about **2.169 GPU-hours of collection**. This is
arithmetic, not a hardware forecast; it excludes prefix prefill, policy training,
head fitting, preparation, selection and tuning. Without measured hardware
rates the estimator returns null, rather than fabricating a complete run estimate.

## Not done, deviations and remaining risks

- **No real experiments or main benchmark table.** Exact model revisions,
  teacher-DPO provenance, datasets/splits, judged generations, credentials and
  target GPU hardware were not supplied. Full LLaMA training was explicitly
  prohibited for this task. Real RM selection/evaluation and the preparation
  launchers are implemented but unvalidated on large checkpoints.
- **External baselines not implemented/retrained.** Official ADPA, TVKD, DPKD,
  DCKD and SimPO code is not present; do not claim a CRPD ablation substitutes
  for them. Faithful preference Delta-MOPD is omitted. TBPO continuity needs
  primary-protocol data adaptation and real runs beyond the tiny equivalence check.
- **External evaluation is not turnkey.** Benchmark plans require actual pinned
  official generation files, judge configs and Arena-Hard checkout/settings.
  Arena-Hard generation and normalization/import of all external results schemas
  are not automated. No official AE2/Arena-Hard/MT-Bench/OLL scores are claimed.
- **Training integration uses an opt-in dedicated loop.** CRPD does not run inside
  the legacy BasicTrainer/Hydra/FSDP loop. This keeps inherited loss/batch behavior
  intact and enables Windows CPU operation. Distributed training, LoRA and sharded
  model loading for CRPD are not implemented. The research template uses full
  FP32 models. Since the review, teachers can set their own `device`/`dtype`
  (for example both FP32 teachers on a second GPU), but multi-GPU placement has
  not been run on CUDA.
- **Scaling/compute auditing needs work.** Since the review, correction features
  are hash-verified binary tensor shards, loaded lazily for minibatch head
  fitting, and are computed in one causal pass per trajectory. Correction volume
  is still large: every prefix state plus half the nonterminal candidate
  endpoints, times `correction_replicates=4`, at about 4 bytes x hidden size per
  visited state. There is one shard file per trajectory. Collection sampling has
  no KV cache and no batching: each generated token reruns the whole context at
  batch size 1. This is correct but probably the dominant GPU cost. Jobs are
  single writer; a restart replays incomplete jobs from their saved semantic
  seeds. Full cumulative GPU/energy accounting and matched-compute tuning
  budgets require real profiling.
- **Audit coverage is finite.** Replicated exponential estimates include bootstrap
  intervals and a rare-tail warning; they are not an oracle. The main head has
  head_val error reporting and candidate coverage; an independent replicated mean
  audit of all teacher-proposed endpoints at research scale is not yet delivered.
- **Platform/runtime scope.** Native Linux/CUDA and the full legacy dependency
  stack have not been run. The Windows/Git-Bash CPU tests that passed use a nightly
  PyTorch build. The HF code path has only run on tiny local Llama checkpoints
  with a monkeypatched local reward model. A stable compatible training
  environment still needs verification. The preparation launcher is single-process
  FP32 full fine-tuning: an 8B SFT/DPO run does not fit one GPU this way and needs
  an external trainer (inherited FSDP stack, alignment-handbook or similar) whose
  DPO reference is recorded the same way.
- **Review.** REVIEW.md lists the defects found in review, the mutation check
  and the readiness verdict.
- **License provenance.** The inherited README declares Apache-2.0, and vendored
  FastChat retains its Apache-2.0 LICENSE. No root LICENSE was present in the
  supplied checkout; this task does not create or infer one for inherited code.

The mathematical method and core verification are delivered. The remaining items
above prevent claiming that the entire research protocol or every external
benchmark adapter is complete. PLAN.md checkmarks track delivered workflow work;
these explicit unresolved requirements are retained for the next reviewer.

## Baseline tag verification (NARVAL task, step 0)

`tbpo-baseline` is a lightweight tag on `01abcff43e4d61903168a0863a3131f1ff3a74f3`
("first commit", 26 Oct 2026 23:20 +0700, Thang1703hrsh). Its parent is
`32d21ec0856dbfb2462f2228c4b82c6acb5c2ceb`. `git show --stat tbpo-baseline`:

```text
 README.md | 3 ---
 1 file changed, 3 deletions(-)
```

The tag is the last of seven user upload commits and precedes Codex's first commit
`21c84f1` ("crpd step 0"). The TBPO sources (`train.py`, `trainers.py`, `loss/`, `config/`,
`preference_datasets.py`) and `reference/ChunkRatio_Math_Checks.{py,json}`,
`reference/ChunkRatio_PD_Method_1_1.md` all arrive in the user's upload `09b1d90`. From
`09b1d90` to the tag only `README.md` changes. The tagged tree has no `chunkratio/`,
`prompts/`, `CLAUDE.md`, `AGENTS.md`, plan or decision files. The tag is therefore correct,
and no `tbpo-original` tag was created.

## NARVAL-comparable protocol (prompt 03)

A second, config-selected protocol (`"protocol_name": "narval"`) sits beside the unchanged
ADPA/TVKD protocol. The method mathematics, loss, routing, targets and variant semantics did
not change. What was added (one `narval:` commit each):

- Batched, KV-cached collection sampler (`sampling.generate_batch`). Prefixes are left-padded,
  each row has its own `torch.Generator` named by the job's semantic id, and sampling uses
  temperature 1 over raw logits with no truncation, penalty or forced EOS. Batched and
  unbatched runs give identical tokens (`tests/test_sampling.py`, TinyLM, tiny Llama and tiny
  Qwen3). The micro-benchmark is logged to `output/benchmarks/sampler_throughput.json`, not
  asserted. On a tiny CPU Llama (16 rows x 24 tokens) the batched path was about 8.7x faster;
  that is not a GPU estimate.
- Pre-registration (DECISIONS items 22-25): collection shared by training seeds
  (`collection_seed`), `correction_replicates = 1`, `states_per_prompt` (pilot value 4), and
  per-row streams.
- `python -m chunkratio diagnose-displacement`: d_s moments, empirical Jensen gap with a
  bootstrap interval and a rare-tail flag, and sibling gap spread omega_s. Labelled a
  diagnostic. Its exact-tree convergence test is `tests/test_diagnostics.py`.
- `config/crpd/narval/{qwen3_8b_to_1p7b,qwen3_1p7b_to_0p6b,llama31_8b_to_llama32_1b}.json`
  with fail-closed validation and the 1.7B -> 0.6B provenance link.
- `prepare-data` for UltraFeedback Binarized (`--format ultrafeedback`, `--assign-split test`,
  `--chat-template-kwargs`). `head_val` and `select` are held out of train_prefs by prompt id,
  and the assignment hash is written to `<output>.splits.json`.
- Preflight: padded output dims must be identical for all four models, and a declared
  `eos_token` must be every tokenizer's EOS. Tiny Qwen3-style checkpoints are built in code
  (`chunkratio/tiny.py::build_tiny_qwen3`, no download).
- `narval-benchmark-plan` (templates only; no API calls) and `report --layout narval`.
- `python -m chunkratio smoke --protocol narval-tiny`, which both smoke launchers now run.

The NARVAL code itself was not inspected in this task. Everything that depends on it is
`null` below. The Table 1/2 column order in `report --layout narval` follows the metric list
in the task text and must be checked against the paper before publication.

## Open inputs (NARVAL)

Every item is `null` in all three protocol files and makes `validate-config` fail with the full
list. Fill them only from the NARVAL repository, the paper or verified checkpoints.

| Group | Fields | Notes |
|---|---|---|
| Checkpoints | `models.{teacher_plus,teacher_base,student_ref}.{path,revision}`; `narval.reference_rows.{teacher_dpo,teacher_sft,student_sft,student_dpo}.{path,revision}` | For qwen3_1p7b_to_0p6b, T0/T+ must equal the `student_sft`/`student_dpo` rows of qwen3_8b_to_1p7b (checked) |
| Base checkpoints | `narval.base_checkpoints.{teacher,student}` | Base vs post-trained Qwen3/Llama ids |
| SFT recipe | `narval.sft.{dataset_revision,epochs,learning_rate,batch_size,max_length}` | Deita-10K-V0 for teacher and student |
| Teacher DPO | `narval.teacher_dpo.{dataset_revision,beta,learning_rate,epochs,max_length}` | UltraFeedback train_prefs. If max_length differs from H=512, revisit H (spec §5.2) |
| Preference data | `narval.preference_data.{revision,split_manifest_sha256}` | Copy `assignment_sha256` from the `.splits.json` written by prepare-data |
| Selection | `selection.reward_model_revision` | CRPD's own checkpoint-selection rule (FsfairX on `select`), the same for every method |
| Open LLM | `open_llm.lm_eval_version`; per task `num_fewshot`, `metric`; `task` for ARC and TruthfulQA | ARC (challenge/easy) and TruthfulQA (mc1/mc2) variants are not verified |
| MT-Bench | `mt_bench.{judge,judge_version}` | 8 categories plus average |
| HH-RLHF | `hh_rlhf.{dataset_revision,subset_ids_file,subset_sha256,judge,judge_version,reference_outputs,reference_model}` | The task text names GPT-4o-mini as reference and GPT-5.4-mini as judge; not verified here |
| EOS (Llama pair only) | `eos_token` | Must be the template's end-of-turn token for all four models |

Declared (ours, not NARVAL's): H=512, L=32, K=4, alpha=0.3, lr=1e-6, 1000 steps, batch 128,
`selection_interval=100`, `states_per_prompt=4`, `correction_replicates=1`, head settings,
`anchor_lambda=0.1`, `anchor_beta=0.05`, held-out fractions 0.05/0.05, `generation_batch_size=16`.

## Pilot: Qwen3-1.7B -> Qwen3-0.6B (commands)

Run on Linux CUDA from the repository root after every open input above is filled. Nothing
here has been run. The checkpoint, data and revision placeholders (`<...>`) are yours to fill.

```text
# 0. Local copies of UltraFeedback (requires the `datasets` package)
python -c "from datasets import load_dataset as l; l('HuggingFaceH4/ultrafeedback_binarized',revision='<REV>',split='train_prefs').to_json('data/narval/raw/uf_train_prefs.jsonl')"
python -c "from datasets import load_dataset as l; l('HuggingFaceH4/ultrafeedback_binarized',revision='<REV>',split='test_prefs').to_json('data/narval/raw/uf_test_prefs.jsonl')"

# 1. Canonical prompts with the student-SFT tokenizer, thinking disabled
python -m chunkratio prepare-data --input data/narval/raw/uf_train_prefs.jsonl --tokenizer <STUDENT_SFT_0P6B> --revision <REV> --output data/narval/qwen3_1p7b_to_0p6b/ultrafeedback_train.jsonl --format ultrafeedback --chat-template-kwargs '{"enable_thinking": false}'
python -m chunkratio prepare-data --input data/narval/raw/uf_test_prefs.jsonl --tokenizer <STUDENT_SFT_0P6B> --revision <REV> --output data/narval/qwen3_1p7b_to_0p6b/ultrafeedback_test.jsonl --format ultrafeedback --assign-split test --chat-template-kwargs '{"enable_thinking": false}'

# 2. Fill both Qwen3 protocol files, then check provenance and open inputs
python -m chunkratio validate-config --config config/crpd/narval/qwen3_8b_to_1p7b.json
python -m chunkratio validate-config --config config/crpd/narval/qwen3_1p7b_to_0p6b.json

# 3. Verify the SFT/DPO reference rows against NARVAL Table 1 before any CRPD run
python -m chunkratio narval-benchmark-plan --config config/crpd/narval/qwen3_1p7b_to_0p6b.json --model <STUDENT_SFT_0P6B> --model-id qwen3-0p6b-sft --output output/narval/rows/student_sft/plan.json
python -m chunkratio benchmark-run --plan output/narval/rows/student_sft/plan.json --output output/narval/rows/student_sft/execution.json
#    (repeat for teacher_sft, teacher_dpo and student_dpo; normalise to measured.jsonl and put the paper's numbers in reported.jsonl)
python -m chunkratio report --layout narval --input output/narval/rows/measured.jsonl --reported output/narval/rows/reported.jsonl --output output/narval/rows/table1_check.md

# 4. Displacement diagnostic on 100 train prompts (before training any student)
python -m chunkratio diagnose-displacement --config config/crpd/narval/qwen3_1p7b_to_0p6b.json --prompts 100 --replicates 16 --output output/narval/qwen3_1p7b_to_0p6b/diagnostic

# 5. One seed of each variant (copies of the protocol file with "variant" and "output" changed)
python -m chunkratio evaluate --config config/crpd/narval/local/qwen3_1p7b_to_0p6b_crpd_main.json
python -m chunkratio evaluate --config config/crpd/narval/local/qwen3_1p7b_to_0p6b_crpd_raw.json
python -m chunkratio evaluate --config config/crpd/narval/local/qwen3_1p7b_to_0p6b_crpd_anchor.json
```

The protocol files set one `cache_root` per pair, so the three variants reuse a single prefix
pool and candidate set, and `crpd_main` and `crpd_anchor` also reuse the correction rollouts and
the head (`crpd_raw` needs none). Stage keys still separate everything a variant changes. On a
budget, keep `generation_batch_size` as high as GPU memory allows: it does not change results.
Place both FP32 teachers on a second GPU with `models.<teacher>.device`.

## Verification after prompt 03 (27 October 2026)

```text
python -m pytest -q
72 passed in 113.99s

script/smoke_test.ps1 -Output output/smoke-narval-ps3          (Windows PowerShell 5.1)
SMOKE PASSED: 7 variants, all stages, export/reload and standalone generation
NARVAL-TINY SMOKE PASSED: prepare-data, diagnose-displacement, 3 variants, report --layout narval

bash --noprofile --norc script/smoke_test.sh --output output/smoke-narval-bash
SMOKE PASSED: 7 variants, all stages, export/reload and standalone generation
NARVAL-TINY SMOKE PASSED: prepare-data, diagnose-displacement, 3 variants, report --layout narval

python script/mutation_check.py
34 of 34 mutations killed (29 earlier + shared collection broken, state selection reading an
outcome, per-row generator removed, EOS from PAD, EOS check skipped)
```

The first PowerShell run failed with `PermissionError` in `os.replace` on `manifest.json`: the
repository is in OneDrive, which briefly locks freshly written files. Atomic writes now retry
that replace with bounded backoff (`artifacts.replace_with_retry`). Native Linux and CUDA are
still not run; no NARVAL checkpoint, dataset or judge was used.

## Modal pilot launcher: Qwen3-1.7B -> Qwen3-0.6B (not yet run)

`script/modal/qwen3_pilot.py` runs the whole pilot on Modal (profile `quangltt280305`, app and
volume `crpd-qwen3-pilot`). It trains the missing checkpoints itself, with assumed recipes listed
in `chunkratio/pilot.py::PILOT_ASSUMPTIONS` and copied into every config. These values are not
verified against NARVAL, and the results are a pilot, not a reproduction.

| Stage | Hardware | What it does |
|---|---|---|
| prepare | CPU x4, 16 GB | Pins Hub revisions, downloads Deita-10K-V0 and UltraFeedback, writes canonical data (thinking disabled, EOS `<|im_end|>`), pilot subset 2000/200/64 prompts |
| sft_teacher | A100-80GB | T0 = SFT of Qwen3-1.7B-Base on Deita (3 epochs, 2e-5, batch 128, max_length 2048, TF32) |
| sft_student | L40S (parallel with sft_teacher) | rho = SFT of Qwen3-0.6B-Base, same data and recipe |
| dpo_teacher | A100-80GB | T+ = DPO of T0 on all UltraFeedback train_prefs (beta 0.05, 5e-7, 2 epochs); its frozen reference is T0 |
| diagnose | L40S | `diagnose-displacement`, 100 prompts, R = 16 |
| train_variants | A100-80GB, 64 GB RAM | `crpd_main`, `crpd_raw`, `crpd_anchor`, seed 0, one shared `cache_root`; FsfairX selection every 250 steps on 64 prompts |
| run_pilot / finish | 0.25 CPU | Orchestrators; phase 2 is spawned separately so each phase has 24 h |

`validate-config` runs in pilot mode. Every training input must be filled; evaluation fields may stay
null, and CRPD refuses to start unless T+'s recorded DPO reference fingerprint equals T0's.
Deviation: the teacher pair is trained from Qwen3-1.7B-Base directly, not taken from the
8B -> 1.7B protocol as NARVAL Table 1 does.

No GPU time has been measured. The 1.7B DPO over about 60k pairs (microbatch 1) is expected to be
the longest stage, and every duration and cost is unknown until a run reports it. Possible early
failures, all in the cheap CPU `prepare` stage: a Hub download that needs a token, or a Deita
field layout different from ShareGPT `conversations`.
