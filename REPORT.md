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
