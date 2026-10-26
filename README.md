# ChunkRatio-PD: Preference Distillation with Sibling Chunk Ratios and Reference Correction

**Method abbreviation:** CRPD  
**Method version:** 1.1  
**Implementation and experiment specification:** revision 2

CRPD transfers preferences from an aligned teacher and its reference model to a smaller student. It compares **sibling chunks that share exactly the same prefix**, combines teacher policy ratios with a learned reference correction, and trains the student with a soft pairwise preference loss.

The implementation is intended for a fork of the TokenRatio/TBPO codebase. The original TBPO-Q and TBPO-A implementations remain separate continuity baselines.

> **Implementation status:** This README specifies the CRPD v1.1 workflow and the revision-2 experimental protocol. The CRPD command names and added module layout below are integration targets, not verified executable entry points. Inherited TBPO setup instructions are identified separately. No CRPD benchmark scores or hardware-performance claims are reported here.

## Contents

- [Overview](#overview)
- [Method](#method)
- [Repository organization](#repository-organization)
- [Installation](#installation)
- [Training protocol](#training-protocol)
- [Training workflow](#training-workflow)
- [Variants and baselines](#variants-and-baselines)
- [Evaluation](#evaluation)
- [Verification and reproducibility](#verification-and-reproducibility)
- [Citation and acknowledgements](#citation-and-acknowledgements)
- [License](#license)

## Overview

CRPD uses four model roles:

| Role | Symbol | Description |
|---|---|---|
| Aligned teacher | $T^+$ | Teacher obtained by DPO from the verified teacher reference |
| Teacher reference | $T^0$ | Exact SFT checkpoint used as the teacher's DPO reference |
| Student reference | $\rho$ | Frozen student SFT checkpoint |
| Trainable student | $\pi_\theta$ | Initialized from $\rho$ and optimized by CRPD |

The primary student-distillation run consumes **prompts only** from the preference datasets. Its soft preference labels come from the teacher pair. Human preference pairs are used to train the DPO teacher and label-based baselines; a separately reported `crpd_anchor` variant also uses them during student training.

The method has three components:

1. **Sibling chunk ratios.** Compare two candidate chunks beginning at the same token prefix. Teacher chunk scores are summed token log-probability differences between $T^+$ and $T^0$.
2. **Reference correction.** Generate continuations from frozen $\rho$, score them with both teachers, and fit one scalar head to the expected future teacher log-ratio. Freeze this head before student training.
3. **Soft preference distillation.** Match the teacher-derived preference probability using differences of student/reference chunk log-ratios.

The deployed student generates normally. It does not require either teacher or the correction head.

CRPD is distinct from any inherited **ChunkRPO** experiment. CRPD uses shared-prefix candidates, sums rather than length-normalized chunk scores, a learned continuation head, and a soft logistic loss. A discourse chunker, cross-response trimmed baseline, or one-sided SBA loss is not the CRPD method. Such legacy code may remain isolated, but must not be relabeled as CRPD.

## Method

Let $s$ denote a prompt and generated prefix, and let $c,d$ be two distinct sibling chunks. Define

$$
\delta_T(c\mid s)=\log\frac{T^+(c\mid s)}{T^0(c\mid s)},\qquad
a_\theta(c\mid s)=\log\frac{\pi_\theta(c\mid s)}{\rho(c\mid s)}.
$$

Every chunk log-probability is a **sum over valid chunk tokens**. Prompt tokens and padding are excluded; a sampled EOS is included once.

### Reference correction

The sequence-level transfer target is

$$
\pi^*(y\mid x)\propto\rho(y\mid x)\frac{T^+(y\mid x)}{T^0(y\mid x)}.
$$

Its exact continuation correction is a log-exponential expectation. CRPD v1.1 approximates it with an ordinary mean:

$$
\phi(s)=\log\mathbb E_{z\sim\rho(\cdot\mid s)}
\exp\left(\log\frac{T^+(z\mid s)}{T^0(z\mid s)}\right),\qquad
\mu_\rho(s)=\mathbb E_{z\sim\rho(\cdot\mid s)}
\log\frac{T^+(z\mid s)}{T^0(z\mid s)}.
$$

A single scalar head $\widehat\mu_\psi$ predicts $\mu_\rho$ from detached frozen-reference features. The head is not an unbiased estimator of $\phi$, and the approximation need not improve transfer for every model pair. See the method document for the Jensen-gap analysis and failure regimes.

### Policy loss

For each candidate, form

$$
t_s(c)=\delta_T(c\mid s)+\widehat\mu_\psi(sc).
$$

For a sibling pair, compute

$$
p_{cd}=\sigma\left(\alpha[t_s(c)-t_s(d)]\right),\qquad
m_{cd}=\alpha[a_\theta(c\mid s)-a_\theta(d\mid s)],
$$

$$
\boxed{\mathcal L_{\mathrm{CRPD}}
=\mathbb E_{s,c,d}\left[\operatorname{softplus}(m_{cd})-p_{cd}m_{cd}\right].}
$$

Use the same positive $\alpha$ for target and student logits. Keep soft labels; candidate source does not determine which candidate is preferred. Implement the loss with binary cross-entropy with logits and the prescribed sampling weights. Do not normalize over the sampled candidate list.

The primary run has no additional SFT or DPO anchor, explicit KL penalty, variance head, reward clipping, length normalization, label smoothing, online target refresh, or optimizer weight decay. The fixed teacher/reference models and the fitted correction head receive no policy-training gradients.

### Exact cases

Before collecting correction data, verify effective-model identities:

- If $T^+=T^0$, retain the initialized student without an optimizer step.
- Otherwise, if $\rho=T^0$, bypass the head and use an exact zero residual while training the policy.
- Otherwise, use the reference-rollout mean correction. This is the intended route for the primary small-student protocol.

Model-family names or approximate score agreement are insufficient to trigger these shortcuts.

## Repository organization

The revision-2 specification is the authority for implementation and experiments; the method document is the authority for mathematics. Keep these sources under `reference/` with SHA-256 hashes.

The following organization separates inherited TBPO infrastructure from the CRPD additions. Existing paths are taken from the supplied TBPO README; inspect the actual fork before relying on them. New CRPD paths are intended integration locations.

```text
.
├── train.py                       # Inherited TBPO training entry point
├── trainers.py                    # Inherited trainers; preserve baseline behavior
├── preference_datasets.py         # Extend loading to expose prompts-only inputs
├── baseline_head.py               # Inherited TBPO head, not the CRPD μ-head
├── merge.py                       # Inherited adapter export utility
├── loss/                          # Preserve TBPO-Q/A and other baseline losses
├── config/                        # Inherited configs; add explicit CRPD configs
├── chunkratio/                    # New CRPD implementation
│   ├── sampling.py                # Prefixes, sibling chunks, reference rollouts
│   ├── scoring.py                 # Token scores, masks, causal reference features
│   ├── collection.py              # Versioned datasets and scoring caches
│   ├── value_head.py              # Scalar μ-head and trajectory-weighted fitting
│   ├── targets.py                 # Exact routing and frozen preference targets
│   ├── loss.py                    # Soft sibling-pair BCE
│   ├── training.py                # Hierarchical sampling, training and resume
│   └── evaluation.py              # Diagnostics and measured cost reports
├── reference/
│   ├── ChunkRatio_PD_Method_1_1.md
│   ├── ChunkRatio_PD_v1_1_Coding_Spec_rev2.md
│   ├── ChunkRatio_PD_Revision_Checks.py
│   ├── ChunkRatio_PD_Revision_Checks.json
│   └── HASHES.txt
├── tests/                         # Mathematical, scoring and pipeline checks
├── script/                        # Inherited launch/evaluation helpers
├── winrate_eval/                  # Inherited evaluation infrastructure
├── mtbench/FastChat/              # Inherited MT-Bench infrastructure
├── env/                           # Environment definitions
└── CLAUDE.md                      # Verified commands and actual integration map
```

Inspect existing repository instructions before integration. Record verified commands and real file mappings in `CLAUDE.md`. Reuse model loading, tokenization, and evaluation utilities when they satisfy the specification. Keep TBPO-Q/A behavior unchanged and verify it on tiny fixtures before and after integration.

## Installation

The supplied TBPO README uses separate Conda environments for training and general-language evaluation. The inherited setup is:

```bash
# Inherited TBPO training environment
conda env create -f env/train_env.yml
conda activate TokenBPO-train

# Inherited general-language evaluation environment
conda env create -f env/eval_env.yml
conda activate TokenBPO-eval
```

Inspect the environment files and available hardware before installation. The inherited documentation mentions CUDA 12.4 and `lm-eval==0.4.9.2`; these are upstream settings, not independently verified CRPD requirements. Freeze the actual dependency versions used by every comparison.

MT-Bench, AlpacaEval, and Arena-Hard may require separate evaluation dependencies. Configure model access and judge credentials through environment variables or the relevant client configuration. Do not commit credentials.

The upstream claims about tested GPUs, LoRA memory requirements, or full-finetuning quality do not establish CRPD hardware performance. Record measured resource usage for CRPD runs.

## Training protocol

The primary protocol follows the **ADPA/TVKD-aligned setup specified in revision 2**. Run separate DPO-MIX-7K and HelpSteer2 tracks, with the corresponding teacher-training and dataset provenance recorded for each track. Literature numbers obtained with different teachers, splits, judges, or decoding settings are not directly interchangeable with these runs.

### Model preparation

| Component | Revision-2 setting |
|---|---|
| Primary teacher family | LLaMA-3.1-8B |
| Primary student family | LLaMA-3.2-1B |
| Second pair | Mistral-7B with H2O-Danube2-1.8B or Danube3-500M, only after tokenizer/template preflight passes |
| Fallback second setting | LLaMA-3.1-8B teacher with LLaMA-3.2-3B student |
| SFT dataset | Deita-10K-V0 |
| SFT recipe | 3 epochs, learning rate `2e-5`, effective batch size 128 |
| Teacher reference $T^0$ | Resulting teacher SFT checkpoint |
| Aligned teacher $T^+$ | DPO from that exact $T^0$; `beta=0.05`, learning rate `5e-7`, effective batch size 128, 2 epochs |
| Student reference $\rho$ | Student trained with the same SFT dataset and recipe |
| Student initialization | Exact copy of $\rho$ |

Resolve full checkpoint IDs, revisions, preprocessing, and training settings in the run manifest. Verify that $T^0$ really is the DPO reference used to obtain $T^+$. A similarly named public base checkpoint is not a substitute.

Confirm identical token mappings, special tokens, EOS conventions, output vocabulary, and canonical chat template across all four roles. The proposed family names do not waive this check. Unsupported cross-tokenizer pairs must fail preflight.

### Data and selection

| Item | Setting |
|---|---|
| Preference datasets | DPO-MIX-7K; HelpSteer2 using helpfulness preferences with ties removed |
| Primary CRPD inputs | Prompts only; teacher pair supplies soft labels |
| Splits | `train`, `head_val`, `select`, and held-out evaluation/test data |
| Head checkpoint selection | Trajectory-weighted regression loss on `head_val` |
| Policy checkpoint selection | Average FsfairX-LLaMA3-RM-v0.1 reward on `select`, with the same declared rule for every method |

Assign duplicate prompt groups to splits before generation. Keep all derived prefixes, candidates, and trajectories in the originating split. Head fitting and policy training may share `train` prompts; disclose the reuse. Reserve separate held-out preference-set prompts for reporting RM scores, and do not select checkpoints on benchmark test prompts.

HelpSteer2 preference filtering prepares the common dataset; “prompts only” means that the primary CRPD student loss does not consume the human pair labels. It does not mean that teacher alignment or dataset preparation is independent of preference annotations.

### CRPD collection settings

| Parameter | Initial setting |
|---|---|
| Training-data response horizon `H` | 512 generated tokens after the original prompt |
| Chunk length `L` | 32 tokens, with EOS/global-horizon stopping |
| Candidates `K` | 4: two from $\rho$, two from $T^+$ |
| Prefix schedule | Every `L` response tokens, including the initial prompt state |
| Candidate pairing | All unordered pairs after exact-token deduplication |
| Endpoint seed fraction | 0.5, as specified in rev2 |
| Correction continuation source | Frozen $\rho$ |
| Head | `Linear(d + 1, 256) -> GELU -> Linear(256, 1)`, no dropout |
| Additional head input | Remaining response-budget fraction |
| Policy weight decay | 0 |

Use the teacher-DPO response horizon instead if it differs, and document the choice. A total context limit is not the same as a generated-response horizon: record prompt/context limits separately and do not copy a total `max_length` into `H` without checking its meaning.

The positive preference scale `alpha`, CRPD learning rate, optimizer, effective batch size, training steps, rollout budget, and parameterization must be explicit in the resolved configuration. Do not silently inherit the original TokenRatio hyperparameters. In particular, the teacher's DPO `beta=0.05` and CRPD's `alpha` have different roles. These settings are not claims of validated CRPD optima.

### Collection versus evaluation decoding

| Use | Decoding contract |
|---|---|
| Training-data collection | Temperature 1, plain categorical sampling, no top-k/top-p truncation, repetition penalty, minimum-length suppression, forced EOS, stop strings, or beam search |
| Benchmark evaluation | Benchmark-prescribed decoding, fixed across methods within each benchmark; independent of the training-data horizon `H` |

For a prefix containing `n` response tokens, the remaining budget is `H-n`. A chunk boundary does not reset this budget and is not itself terminal. EOS and the global response horizon are terminal; their continuation value is exactly zero. Do not append an artificial EOS at the horizon.

## Training workflow

### 1. Prepare and verify checkpoints

Reproduce or obtain the teacher SFT/DPO pair and student SFT checkpoint under the declared protocol. Verify identities, tokenization, stopping conventions, and split assignments. Apply exact-case routing before correction collection.

### 2. Collect prefixes and sibling chunks

Build the fixed prefix pool from $\rho$ trajectories. Generate the candidate mixture at each selected nonterminal state, deduplicate exact token sequences, and skip states with fewer than two distinct candidates. Cache candidate tokens, provenance, endpoint metadata, and teacher/reference log-probabilities.

Sample policy-training examples hierarchically: prompt uniformly, then an eligible state uniformly within the prompt, then a pair uniformly within the state. An unweighted mean over a flattened pair table does not implement this sampling measure.

### 3. Collect correction trajectories

Select seed states before observing their returns. Include prefix states and nonterminal endpoints from both candidate sources. Generate full suffixes from frozen $\rho$ and score each suffix with $T^+$ and $T^0$:

$$
r_t=\log T^+(y_t\mid s_t)-\log T^0(y_t\mid s_t),\qquad
Y_t=\sum_{j=t}^{N-1}r_j.
$$

Compute reverse cumulative sums. Attach $Y_t$ to the state **before** token $y_t$. The endpoint target excludes the teacher score of the chunk already emitted. Terminal targets are zero.

### 4. Fit and freeze the correction head

Extract detached frozen-$\rho$ features, including the remaining-budget input. Fit squared error over all selected nonterminal states, with a sum per trajectory divided by fixed `H`, then an average over trajectories. Do not normalize by each realized suffix length.

Select the head on `head_val`, store its hash, and freeze it. The head predicts a mean future log-ratio; do not substitute the TBPO partition-baseline head or update targets online during policy training.

### 5. Build targets and train the student

Cache each candidate's `target_score = logp_plus - logp_base + endpoint_mu`. Construct soft pair labels and recompute only student chunk scores during policy optimization. Keep frozen models unchanged, preserve gradients through student log-probabilities, and use the same `alpha` on both sides of the loss.

Soft-label BCE has a nonzero entropy floor. Report excess BCE, logit error, and saturation rather than expecting the training loss to reach zero.

### 6. Select, export, and evaluate

Select policy checkpoints using the shared reward-model rule on `select`, then export the student. For adapter training, merge or export with the exact initialization and verify pre-/post-export score agreement. Ensure that the final model generates without teachers or the correction head.

### Target command interface

The following is the **illustrative stage interface from the specification**. Bind it to verified entry points during integration before treating these commands as executable:

```text
crpd validate-config
crpd toy-check
crpd collect-prefixes
crpd collect-candidates
crpd collect-correction
crpd fit-head
crpd build-targets
crpd train-policy
crpd evaluate
```

Each stage should consume a resolved configuration and a compatible run manifest. The original `loss=A_tbpo`, `loss=Q_tbpo`, or `loss=chunkrpo` launchers do not run CRPD. Add dedicated CRPD configuration and batch/loss integration; do not merely rename a launcher.

## Variants and baselines

### CRPD variants

| Variant | Definition | Role |
|---|---|---|
| `crpd_main` | Reference-rollout mean correction | Primary v1.1 method |
| `crpd_raw` | Zero endpoint correction | Test the value of correction |
| `crpd_mu_teacher` | $\mathbb E_{T^+}\log(\rho/T^0)$ with separate teacher rollouts and labels | v1.0-direction ablation |
| `crpd_token` | `L=1` | Token-granularity comparison |
| `crpd_suffix` | Complete suffix candidates; zero terminal correction | Test whether chunk localization is useful |
| `crpd_anchor` | CRPD plus $\lambda$ times human-pair DPO loss, with fixed documented $\lambda$ | Secondary comparison allowing human labels in student training |
| `crpd_audit` | Expensive exponential-moment estimates on a small subset, with uncertainty | Diagnostic, not a claimed exact oracle |

Only `crpd_anchor` adds the human-label objective. It does not change the definition of `crpd_main`. Compare label access and total compute explicitly; adding an anchor alone does not make all training conditions identical.

### Reference rows and external baselines

Include teacher DPO ($T^+$), teacher SFT ($T^0$), student SFT ($\rho$), and student DPO reference rows.

Retrain the specified comparisons under the common protocol:

- **Preference optimization:** DPO and SimPO. The student-DPO reference row also serves as the DPO baseline; avoid duplicating an identical run.
- **Distillation:** DPKD, DCKD, ADPA, and TVKD.
- **TokenRatio continuity:** TBPO-Q and TBPO-A with the student SFT checkpoint as their reference.
- **Teacher-delta transfer:** a faithful preference adaptation of Δ-MOPD if feasible; otherwise disclose its omission.

Use verified implementations or documented faithful reimplementations. Preserve TBPO-Q/A losses and baseline behavior. A separate UltraFeedback experiment is optional and requires its own verified teacher pair; it is not the primary protocol.

## Evaluation

### Metrics

| Evaluation | Required reporting |
|---|---|
| FsfairX RM score | Mean on held-out prompts from each preference dataset, separate from checkpoint-selection prompts |
| MT-Bench | Score with a predeclared fixed judge and identical decoding/settings across methods |
| AlpacaEval win rate against ADPA | Compare against the retrained ADPA outputs using the same declared judge as the corresponding primary judge-based comparison |
| AlpacaEval 2 | Official configuration; report both length-controlled win rate and raw win rate |
| Arena-Hard | Official configuration and win rate |
| Open LLM tasks | ARC, HellaSwag, TruthfulQA, MMLU, Winogrande, and GSM8K; report per-task values and the declared average |
| Cost | Generated/scored tokens by model, GPU-hours, wall time, and peak memory |

Freeze judge identities, prompt templates, benchmark versions, and decoding configurations before evaluating methods. The rev2 protocol notes different judges in the original ADPA and TVKD studies; do not mix their published scores into a common directly comparable table. Rerun the comparison under the chosen judge.

Use one declared common judge for the primary MT-Bench/AE-against-ADPA comparisons. Preserve the official AlpacaEval 2 and Arena-Hard configurations for their separately labeled results; if their prescribed judges differ, disclose those benchmark-specific identities. A custom judge run must not be presented as the official benchmark configuration. Never change a judge between methods within one reported comparison.

Pin `lm-eval-harness`, exact task variants, few-shot settings, and aggregation. Do not silently substitute an unspecified `QA` task for TruthfulQA. Preserve benchmark-standard MT-Bench scoring rather than assuming that an inherited pairwise script produces the required score.

### Existing evaluation infrastructure

The supplied TBPO README identifies these integration points:

| Path | Intended reuse |
|---|---|
| `script/eval/general/` | Open LLM task launchers |
| `winrate_eval/` | Judge/model/client configuration and win-rate generation |
| `mtbench/FastChat/` | MT-Bench answer generation and judging |
| `diversity_metrics/` | Optional diversity diagnostics |

Inspect the scripts and resolved settings before reuse. The inherited `tonyshelby/processed_data` download is not automatically the CRPD training or evaluation split. AlpacaEval 2 LC and Arena-Hard may require additional verified adapters. Diversity measures are supplementary and do not replace the rev2 primary metrics.

### Reporting

Report at least three training seeds for `crpd_main`, `crpd_raw`, and the strongest baselines, with prompt-level paired confidence intervals. Keep training-seed variation separate from prompt-sampling uncertainty. Agreement with the training teacher is not an independent quality metric.

Main table schema:

| Method | DPO-MIX-7K: RM / MT / AE-vs-ADPA / AE2-LC / OLL | HelpSteer2: RM / MT / AE-vs-ADPA / AE2-LC / OLL | GPU-hours |
|---|---|---|---|

Provide Arena-Hard, raw AlpacaEval 2 WR, and per-task Open LLM scores in companion tables. Record the actual checkpoints and decoding configurations with every result. Do not populate tables with synthetic-check values or unexecuted estimates.

Report both matched-candidate and matched-total-compute comparisons. The latter includes correction collection, scoring by both teachers, head fitting, student training, and tuning. Report teacher/SFT preparation costs separately and state how shared preparation is attributed. Reference-model rollouts reduce reliance on teacher autoregressive generation but do not eliminate teacher scoring costs.

## Verification and reproducibility

Before large-model training, complete:

1. **Exact tree checks:** residual representations, Q identity, both KL identities, approximation regimes, and an added variable-EOS fixture.
2. **Scoring checks:** causal shift, positional padding masks, EOS-as-PAD, batched/unbatched agreement, and frozen-prefix feature causality.
3. **Training checks:** suffix indexing, global horizon, raw sampling distribution, pair-loss sign/symmetry, gradient routing, and correct hierarchical/trajectory weighting.
4. **Operational checks:** exact-case routing, cache invalidation, restart/resume without duplicated records, and unchanged TBPO-Q/A baseline behavior.
5. **Learned finite-data experiment:** learn the head from finite reference rollouts and fit an actual policy; distinguish approximation error, predictor error, and policy-fitting error.
6. **Tiny causal-LM smoke test:** exercise the full correction path with distinct frozen models, then export/reload the student and generate without the teacher pair or head.

The supplied exact-tree script enumerates expectations; its success does not establish learned-head performance, LLM quality, or practical speedup. Label synthetic experiments accordingly.

Persist model/adapters and tokenizer identities, source hashes, code revision, dependency versions, split assignments, stopping conventions, independent seed streams, head hash, target-table provenance, and measured costs. Use atomic writes and compatible manifests for resume. Changing the head, `alpha`, chunk rule, or model must invalidate the appropriate downstream cache entries.

## Citation and acknowledgements

Use the exact paper title when referring to this work:

> **ChunkRatio-PD: Preference Distillation with Sibling Chunk Ratios and Reference Correction** (CRPD).

CRPD author names, publication venue, and paper URL should be added when finalized. No publication metadata is fabricated here.

This project builds on TokenRatio/TBPO. Cite that work separately when using its code or baselines:

```bibtex
@misc{nguyen2026tokenratio,
  title={TokenRatio: Principled Token-Level Preference Optimization via Ratio Matching},
  author={Truong Nguyen and Tien-Phat Nguyen and Linh Ngo Van and Duy Minh Ho Nguyen and Khoa D. Doan and Trung Le},
  year={2026},
  eprint={2605.12288},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2605.12288}
}
```

Relevant methods and protocol sources:

- [TokenRatio](https://arxiv.org/abs/2605.12288).
- [ADPA: Advantage-Guided Distillation for Preference Alignment in Small Language Models](https://arxiv.org/abs/2502.17927) and its [implementation](https://github.com/SLIT-AI/ADPA).
- [TVKD: Preference Distillation via Value Based Reinforcement Learning](https://arxiv.org/abs/2509.16965).
- [PAD: Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models](https://arxiv.org/abs/2502.14272).

Retain attribution for the inherited TBPO code and the vendored [FastChat](https://github.com/lm-sys/FastChat) evaluation components. Paper-specific settings are defined by `reference/ChunkRatio_PD_v1_1_Coding_Spec_rev2.md`; verify external baseline configurations against their original implementations.

## License

The supplied upstream TBPO README declares Apache-2.0 licensing. Consult and retain the repository's actual `LICENSE` and applicable third-party notices when preparing this fork for release. The README revision does not change licensing or establish a new license for added code.
#   C h u n k R a t i o - P D  
 