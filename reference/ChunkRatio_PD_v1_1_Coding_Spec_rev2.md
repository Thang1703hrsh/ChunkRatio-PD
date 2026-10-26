# ChunkRatio (ChunkRatio-PD) v1.1: Implementation and Experiment Specification

**Revision:** 2 (26 October 2026). Replaces `ChunkRatio_PD_v1_1_Coding_Spec.md`.
**Method version:** 1.1 (unchanged).
**Audience:** a coding assistant (Claude Code) working in a fork of the TokenRatio/TBPO codebase.
**Status:** requirements only. Nothing here is an implemented result or evidence of effectiveness.

## 0. What changed in this revision

| Area | Revision 1 | Revision 2 |
|---|---|---|
| Codebase | Unspecified repository | Fork of the TBPO codebase; mapping and integration rules in Section 3 |
| Experimental protocol | Models, data and benchmarks left `null` | Primary protocol fixed to the ADPA/TVKD setup so that main tables are directly comparable (Section 11) |
| Response horizon | `H = 256` | `H = 512` for training data; evaluation decoding is independent of `H` |
| Decoding | One sampling convention | Raw sampling for data collection only; benchmark-standard decoding for evaluation |
| Metrics | Generic | Original ADPA/TVKD metrics plus standard AlpacaEval 2 LC and Arena-Hard, one fixed judge |
| Table rows | Methods only | Adds teacher DPO, teacher SFT, student SFT and student DPO reference rows |
| Label source | Implicit | States that the primary method uses prompts only; a human-label-anchored variant is a declared secondary run |
| Checkpoint selection | Policy-selection prompts | Same reward-model selection rule as ADPA/TVKD for every method, plus disclosure |
| Paths | Absolute Windows paths | Repository-relative `reference/` directory |
| Agent instructions | `AGENTS.md` | `CLAUDE.md` in the fork |

The scientific method, loss, routing and invariants are identical to v1.1. Only the implementation and experiment protocol are made concrete.

## 1. Assignment and sources of truth

Implement ChunkRatio-PD v1.1 faithfully in a fork of the TBPO codebase, with small auditable modules, reproducible artifacts and real tests. Do not invent another method.

Copy these files into `reference/` in the fork and record their SHA-256 hashes in `reference/HASHES.txt`:

- `ChunkRatio_PD_Method_1_1.md`: scientific definition, theory and limitations.
- `ChunkRatio_PD_Revision_Checks.py`: exact finite-tree reference implementation.
- `ChunkRatio_PD_Revision_Checks.json`: numerical fixtures from that script.
- This file.

Precedence when documents disagree: (1) the method document for mathematics, (2) this file for implementation and protocol, (3) existing TBPO code conventions for everything else.

If implementation reveals a mathematical contradiction, report the exact equation and a minimal counterexample before changing anything. Missing model access, hardware or datasets must not block CPU math tests or the tiny-model smoke test. They do block any claim that a real experiment has run.

## 2. Method invariants

| Item | Required behavior |
|---|---|
| Teacher pair | Frozen aligned teacher `teacher_plus` ($T^+$) and its verified alignment reference `teacher_base` ($T^0$), i.e. the exact checkpoint from which $T^+$ was DPO-trained |
| Student reference | Frozen `student_ref` ($\rho$), identical to the student's initialization |
| Student | `student` ($\pi_\theta$), initialized from $\rho$ |
| Vocabulary | Identical token IDs, special tokens and chat template across all four models |
| Supervision unit | Two distinct chunks that start at exactly the same token prefix |
| Main correction | $\mu_\rho(s)=\mathbb E_{z\sim\rho(\cdot\mid s)}\log[T^+(z\mid s)/T^0(z\mid s)]$ |
| Correction rollouts | Sampled from frozen $\rho$ (not $T^+$, not the moving student) |
| Correction scoring | Both teachers score sampled tokens by teacher forcing |
| Predictor | One scalar head on detached frozen-$\rho$ features, fit by squared error, then frozen |
| Policy objective | Soft-label binary cross-entropy on student/reference chunk log-ratio differences |
| Transfer exponent | Exactly one. No reward exponent, discount, normalization or clipping |
| Inference | Exported student generates normally; no teacher or head at deployment |

The primary run has no SFT anchor, extra KL penalty, variance head, estimator switching, reward clipping, label smoothing, entropy bonus, TD bootstrapping, online target refresh or weight decay. Parameter-efficient training is allowed if the frozen reference is provably the exact initialization; record the choice.

## 3. Working in the TBPO codebase

The TBPO repository is cited by the TokenRatio paper as `github.com/dinhtruongng/TBPO`. Its internal structure and training framework were not inspected when this file was written. The first task is to inspect it.

### 3.1 First actions for Claude Code

1. Read the repository README, training entry points, configs and any existing agent instructions.
2. Write `CLAUDE.md` at the fork root with: the build/test commands you verified, the mapping table below filled with real paths, and the rule "do not modify TBPO baseline behavior; add ChunkRatio code under `chunkratio/`".
3. Run the original TBPO-Q and TBPO-A training on a tiny config to confirm the baselines still work before adding code.
4. Report the framework (TRL, alignment-handbook, custom, …), versions and license.

### 3.2 Reuse versus add

| Need | Expected source in TBPO | Rule |
|---|---|---|
| Model loading, chat template, tokenization | Reuse | Must produce identical `prompt_ids` for all four models |
| Data loading for preference datasets | Reuse | Extend to emit prompts only for the primary run |
| Per-token log-prob computation | Reuse only if it passes Section 6 tests | Must return **per-token** values; chunk scores are sums, never means |
| TBPO-Q/A losses and state head | Keep untouched | Needed as continuity baselines |
| Evaluation scripts (Open LLM, MT-Bench) | Reuse | Same settings for every method |
| Sibling generation, correction rollouts, scoring cache, μ-head, target table, pair loss, hierarchical sampler | Add under `chunkratio/` | Follow Sections 5 to 9 |

A generic DPO trainer, a generic LM loss or a default generation preset is not an implementation of this method. If TBPO's trainer is used for the policy stage, subclass it and replace only the loss and the batch construction; verify with the fixtures in Section 12.

## 4. Mathematical contract

All logarithms are natural. All chunk scores are sums over valid chunk tokens.

For a shared prefix $s$ and candidate chunk $c$:

$$
\delta_T(c\mid s)=\log T^+(c\mid s)-\log T^0(c\mid s),\qquad
a_\theta(c\mid s)=\log\pi_\theta(c\mid s)-\log\rho(c\mid s).
$$

Exact target, for reference only:

$$
\pi^*(y\mid x)\propto\rho(y\mid x)\frac{T^+(y\mid x)}{T^0(y\mid x)},\qquad
\phi(s)=\log\mathbb E_{\rho(\cdot\mid s)}e^{d_s(z)},\quad d_s(z)=\log\frac{T^+(z\mid s)}{T^0(z\mid s)}.
$$

Implemented target with a frozen head $\widehat\mu_\psi\approx\mathbb E_\rho d_s$:

$$
t_s(c)=\delta_T(c\mid s)+\widehat\mu_\psi(sc),\qquad
u_{cd}=\alpha[t_s(c)-t_s(d)],\quad p_{cd}=\sigma(u_{cd}),
$$

$$
m_{cd}=\alpha[a_\theta(c\mid s)-a_\theta(d\mid s)],\qquad
\ell_{cd}=\operatorname{softplus}(m_{cd})-p_{cd}\,m_{cd}.
$$

Use the same positive `alpha` on both sides. $p_{cd}$ means "c is preferred to d". Candidate source and input order do not determine preference. Never harden labels (`argmax`, `p > 0.5`, chosen/rejected).

Do not softmax over the sampled candidate list. The theoretical policy $q_{\widehat\mu}(c\mid s)\propto\rho(c\mid s)\exp\{\delta_T(c\mid s)+\widehat\mu(sc)\}$ is normalized over the complete chunk space; pairwise training avoids that normalizer, and a sampled set does not identify mass outside itself.

### 4.1 Exact-case routing

Decide before collecting correction data, using verified effective-model identity (weights, adapters, architecture, tokenizer, precision conventions):

1. $T^+=T^0$: return the unchanged student, record `no_alignment_update`, take no optimizer step.
2. Else $\rho=T^0$: route `exact_zero_residual`; skip head training, set every endpoint correction to 0, train the policy.
3. Else: route `reference_mean`; collect $\rho$ rollouts, fit and freeze the head, train the policy.

Never apply an approximate-equality shortcut. Route 2 is required because $\mu_\rho$ can be nonzero while the exact residual is zero. In the primary protocol (Section 11), $\rho\ne T^0$ and the route is `reference_mean`.

## 5. Tokens, horizon and terminal states

### 5.1 Canonical inputs

Serialize each prompt once with the declared chat template; persist `prompt_ids`. All models receive identical IDs. Never decode and retokenize candidates for another model. Preflight must verify vocabulary mapping, added and special tokens, EOS IDs and output dimensions; equal vocabulary size alone is insufficient. Unsupported cross-tokenizer pairs fail preflight.

### 5.2 Global response budget for training data

`H` is the maximum number of generated response tokens after the prompt, including a sampled EOS. Default `H = 512`, chosen because benchmark answers (AlpacaEval, MT-Bench) are often longer than 256 tokens. If the teacher DPO stage used a different maximum length, use that value and record it.

- A prefix with `n` response tokens has remaining budget `H - n`; never reset it at chunk boundaries.
- A candidate stops at the first EOS or after `min(L, H - n)` tokens.
- A correction continuation from endpoint `sc` has budget `H - n - len(c)`.
- Chunk boundaries are not terminal. EOS or the global horizon is terminal and its endpoint value is exactly 0.
- At a horizon cutoff, do not append an EOS. Score only emitted tokens. EOS is counted once. Padding and post-EOS tokens are excluded.
- Prompt length plus `H` must fit every model's context. Drop ineligible prompts by a pre-generation rule; never silently truncate a continuation.

### 5.3 Two decoding regimes

| Regime | Used for | Settings |
|---|---|---|
| **Collection sampling** | Prefix pool, sibling candidates, correction rollouts | Temperature 1, no top-k/top-p, no repetition penalty, no min-length or forced EOS, no stop strings, no beam search. Inspect the resolved generation config; do not trust checkpoint defaults |
| **Evaluation decoding** | All benchmark generations, for every method | The benchmark's standard settings (Section 11.4), identical across methods, independent of `H` |

A minimal categorical sampler over `softmax(raw_logits)` is acceptable for collection. Resumed jobs restore sampler state; never retry or filter samples by outcome.

## 6. Scoring and features

### 6.1 Causal alignment

```python
# ids = prefix_ids + continuation_ids, unpadded; P = len(prefix_ids) >= 1, N = len(continuation_ids)
prediction_logits = logits[:, P - 1 : P + N - 1, :].float()   # logits[:, k] predicts ids[:, k+1]
log_probs = prediction_logits.log_softmax(dim=-1)
token_logps = log_probs.gather(-1, continuation_ids[..., None]).squeeze(-1)
# apply positional validity masks before summing
```

One tested scoring function serves all frozen models and the student. Prompt tokens never enter chunk scores. Never use mean cross-entropy as a log-probability. Teacher-forced rescoring is authoritative over generation-time scores. Exclude padding by position and attention mask, not by `token_id != pad_id` (EOS may equal PAD). Verify batched/unbatched equality. No sequence packing initially. Log-softmax and sums in FP32 or higher; tree tests in FP64. Non-finite scores are errors with record IDs.

### 6.2 Reference features for the head

The state before token `y[t]` uses $\rho$'s final-layer hidden state at position `P + t - 1`. An endpoint `prefix + c` uses the hidden state of its last valid token. Features always come from frozen $\rho$, are detached, and must equal those from running the prefix alone.

Head: `Linear(d + 1, 256) -> GELU -> Linear(256, 1)`, no dropout. The extra input is the remaining-budget fraction `(H - n) / H`. Terminal values are hard-coded to 0 regardless of head output. Record layer, dtype, budget encoding and architecture.

Frozen models run in `eval()` with `no_grad`. The student uses `eval()` (no dropout) **with** gradients. Features used to fit the head must be ordinary detached tensors, not inference-mode tensors.

## 7. Data construction

### 7.1 Splits

Assign prompts and duplicate groups to splits before generating anything: `train`, `head_val`, `select` (checkpoint selection), `test`. Everything derived from a prompt stays in its split. Head fitting and policy training may share `train` prompts; disclose this. Never select anything on `test`. In the primary protocol, `test` is the benchmark itself (Section 11), and `select` is held out from the training preference sets.

### 7.2 Prefix pool and siblings

- Build a fixed prefix pool from $\rho$ trajectories under collection sampling.
- Select nonterminal states on a causal schedule, default every `L` response tokens including the prompt state.
- At each state generate `K = 4` candidates: 2 from $\rho$, 2 from $T^+$, all under the same chunk rule.
- Deduplicate by exact token sequence, keep provenance, skip states with fewer than 2 distinct candidates, and record skips.
- Use all unordered pairs.

**Hierarchical weighting.** Sample a prompt uniformly, then a state uniformly within it, then a pair uniformly within the state. The equivalent full-data weight is

$$
w_{x,s,c,d}=\frac{1}{N_{\rm prompts}}\cdot\frac{1}{N_{\rm states}(x)}\cdot\frac{1}{\binom{K_s}{2}}.
$$

Do not flatten pairs and take an unweighted mean.

### 7.3 Correction trajectories

Choose seeds before observing returns: training-prefix states plus a fraction of nonterminal candidate endpoints (default 0.5), from both candidate sources. From each seed, generate complete $\rho$ continuations to the terminal condition. Score with both teachers:

$$
r_t=\log T^+(y_t\mid s_t)-\log T^0(y_t\mid s_t),\qquad Y_t=\sum_{j=t}^{N-1}r_j,\quad Y_N=0.
$$

```python
def suffix_returns(increments):          # one valid suffix, EOS included, no padding
    return increments.float().flip(0).cumsum(0).flip(0)
```

`Y_t` belongs to the state before `y_t`. Fixture: increments `[0.3, -0.2, 0.4]` give `[0.5, 0.2, 0.4]`; the endpoint after two tokens has target `0.4`; terminal is `0`. No standardization, clipping, discount or division by length.

### 7.4 Head objective

$$
\mathcal L_{\rm head}=\frac1B\sum_{b=1}^{B}\frac1H\sum_{t\in I_b}\big(f_\psi(h_\rho(s_{b,t}),\,b_{b,t})-Y_{b,t}\big)^2,
$$

over uniformly sampled trajectories and all visited nonterminal states. With gradient accumulation, normalize by total trajectories times `H`. Select the checkpoint on `head_val` with the same weighting, freeze it, and store its hash. Report replicated-rollout audits on fixed endpoints, including teacher-proposed ones.

## 8. Target table and policy training

For each candidate, cache `logp_plus`, `logp_base`, `logp_ref`, terminal status and the frozen endpoint correction. Then compute `target_score = logp_plus - logp_base + endpoint_mu`. Store target scores separately from `alpha`.

```python
import torch
import torch.nn.functional as F

def pair_loss(logp_student_c, logp_student_d, logp_ref_c, logp_ref_d,
              target_score_c, target_score_d, alpha):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    with torch.no_grad():
        target_logit = alpha * (target_score_c.float() - target_score_d.float())
        target_p = torch.sigmoid(target_logit)
    margin = alpha * ((logp_student_c.float() - logp_ref_c.detach().float())
                      - (logp_student_d.float() - logp_ref_d.detach().float()))
    losses = F.binary_cross_entropy_with_logits(margin, target_p, reduction="none")
    return losses, margin, target_logit
```

Apply the hierarchical weights outside this function. No `pos_weight`. Never detach student log-probabilities. Monitor **excess** BCE (loss minus Bernoulli entropy of the target), logit error and saturation rate. Weight decay is 0 in the primary run.

Fixture: $\delta_c=0.8$, $\delta_d=0.2$, $\mu_c=0.1$, $\mu_d=-0.1$, $\alpha=0.3$ give target logit $0.24$ and $p\approx0.55971365$. At initialization, the margin is 0, the loss is $\log 2$, and $\partial\ell/\partial m\approx-0.05971365$. Swapping $c$ and $d$ preserves the loss.

With LoRA, the reference must be provably the exact initialization; audit reference scores before and after training.

## 9. Variants that the code must support

All variants share prompts, splits, candidate tables where valid, and evaluation.

| Name | Change from primary | Purpose |
|---|---|---|
| `crpd_main` | None | Primary method (v1.1) |
| `crpd_raw` | `endpoint_mu = 0` | Does the correction help? |
| `crpd_mu_teacher` | v1.0 direction: $\mathbb E_{T^+}\log[\rho/T^0]$, separate $T^+$ rollouts and labels | Direction ablation |
| `crpd_token` | `L = 1` | Granularity |
| `crpd_suffix` | Candidates run to EOS/horizon; endpoint correction 0 | Is localization worthwhile? |
| `crpd_anchor` | Adds $\lambda\cdot$ DPO loss on the human preference pairs of the training set; λ fixed in config | Fair comparison with label-using baselines; reported separately |
| `crpd_audit` | Expensive exponential-moment estimates of $\phi$ on a small audit subset, with sampling uncertainty | Diagnostic only |
| Optional ablations | Uncertainty-tempered labels, decision-point branching, mature-student prefixes | Preregistered, never in the primary run |

External baselines must use their verified implementations or faithful reimplementations, trained under Section 11. Never relabel a ChunkRatio ablation as ADPA, TVKD, TBPO or another method.

## 10. Artifacts, caching and stages

Use versioned JSONL plus tensor shards. Required records:

- Run manifest: versions, document hashes, code revision, resolved config, model identities, precision, seed streams, split hash, EOS/horizon conventions, route.
- Prompt, state, candidate, correction trajectory, feature row, frozen target, pair.
- Evaluation: per-prompt generations, judge identity, outcomes, uncertainty, measured cost.

Cache keys include token IDs, model/adapter identity, tokenizer/template, EOS/horizon, feature convention and precision. Invalidation rules:

- A new head invalidates endpoint corrections only.
- A new `alpha` invalidates pair probabilities only.
- A new `L` invalidates candidates and pairs.
- A new model, tokenizer or `H` invalidates everything downstream.

Writes are atomic. Resume restores optimizer, scheduler, RNG and sampler state without duplicate records. Use separate RNG streams for splitting, each generation source, correction collection, head initialization and policy sampling.

Stages (CLI names are illustrative):

```text
crpd validate-config     crpd toy-check          crpd collect-prefixes
crpd collect-candidates  crpd collect-correction crpd fit-head
crpd build-targets       crpd train-policy       crpd evaluate
```

## 11. Experimental protocol

### 11.1 Primary protocol: ADPA/TVKD setup

This setup is used by ADPA (ICLR 2025) and TVKD (NeurIPS 2025). It satisfies the v1.1 invariants: $T^0$ is the true DPO reference, the tokenizer is shared, and the student is much smaller than the teacher.

| Item | Setting |
|---|---|
| Main pair | Teacher LLaMA-3.1-8B, student LLaMA-3.2-1B (shared Llama 3 tokenizer; confirm in preflight) |
| Second pair | Teacher Mistral-7B, student H2O-Danube2-1.8B or Danube3-500M, **only if** tokenizer preflight passes; otherwise use LLaMA-3.2-3B as a second student size |
| SFT data | Deita-10K-V0; 3 epochs, lr 2e-5, batch 128 (as in ADPA/TVKD) |
| $T^0$ | Teacher after SFT |
| $T^+$ | DPO from $T^0$; β = 0.05, lr 5e-7, batch 128, 2 epochs (as in ADPA/TVKD) |
| $\rho$ | Student after the same SFT |
| Preference sets | DPO-MIX-7K; HelpSteer2 (helpfulness, ties removed) |
| Primary run inputs | Prompts only; labels come from the teacher pair |
| Checkpoint selection | FsfairX-LLaMA3-RM-v0.1 average reward on the `select` split, the same rule for every method |

Before training any student, reproduce the teacher pair and the SFT student, and verify that `teacher_base` is the exact DPO reference.

### 11.2 Secondary protocol: TokenRatio continuity

Run TBPO-Q and TBPO-A from the forked codebase on the primary protocol's data, with the student SFT as their reference. Also report Open LLM and MT-Bench as in TokenRatio. A separate UltraFeedback protocol is optional and requires its own verified teacher pair.

### 11.3 Methods in the main table

Reference rows, required:

- Teacher DPO ($T^+$)
- Teacher SFT ($T^0$)
- Student SFT ($\rho$)
- Student DPO on human labels

Baselines, all retrained under 11.1:

- Preference optimization: DPO, SimPO
- Distillation: DPKD, DCKD, ADPA (official code), TVKD
- Continuity: TBPO-Q, TBPO-A
- Teacher-delta: Δ-MOPD adapted to preferences, if a faithful implementation is feasible; otherwise state the omission

Ours: `crpd_main`, plus `crpd_raw`, `crpd_token`, `crpd_suffix` and `crpd_anchor` in the ablation table.

### 11.4 Evaluation

| Metric | Purpose | Setting |
|---|---|---|
| FsfairX RM score | Same as TVKD | Average on held-out prompts of each preference set |
| MT-Bench | Same as ADPA/TVKD | One fixed judge for all methods; state which (ADPA used GPT-4-0125-preview, TVKD gpt-4o-mini) |
| AlpacaEval WR against ADPA | Same as ADPA/TVKD | Same judge |
| AlpacaEval 2 LC and WR | Standard, length-controlled | Official configuration |
| Arena-Hard WR | Standard | Official configuration |
| Open LLM average (ARC, HellaSwag, TruthfulQA, MMLU, Winogrande, GSM8K) | Capability retention; continuity with TokenRatio and ADPA | lm-eval-harness, fixed version |
| Cost | Required for every row | Generated and scored tokens by model, GPU-hours, wall time, peak memory |

Report at least 3 training seeds for `crpd_main`, `crpd_raw` and the strongest baselines, with prompt-level paired confidence intervals. Agreement with the training teacher is not an independent quality metric.

### 11.5 Main table layout

| Method | DPO-MIX-7K: RM / MT / AE-vs-ADPA / AE2-LC / OLL | HelpSteer2: RM / MT / AE-vs-ADPA / AE2-LC / OLL | GPU-h |
|---|---|---|---|

Report Arena-Hard and per-task Open LLM scores in a second table. Report both matched-candidate and matched-total-compute comparisons; matched compute includes correction collection, head fitting and tuning.

### 11.6 Disclosures required in the paper

- The primary ChunkRatio run uses prompts only, while label-using baselines see human pairs; `crpd_anchor` gives the like-for-like comparison.
- Total compute of every method.
- The judge models, decoding settings and lm-eval-harness version.
- Training-data horizon `H`, chunk length `L`, `K` and `alpha`.

## 12. Verification before scaling

| Test | Required observation |
|---|---|
| Tree reference | Reproduce the residual identities, the Q identity, both KL identities and the regime table from `reference/ChunkRatio_PD_Revision_Checks.py` in FP64 |
| Variable stopping | Tree with EOS before horizon: action probabilities sum to 1; chunk factorization invariant |
| Shift and mask | Hand-computed fixture matches; prompt and padding excluded; EOS counted once |
| Padding and batching | Same scores and features batched and unbatched, including EOS = PAD |
| Prefix causality | Features unchanged when an unseen suffix changes |
| Suffix indexing | `[0.3, -0.2, 0.4]` fixture and endpoint exclusion |
| Global horizon | Prefix with `n` tokens has at most `H - n` future tokens; chunk cutoff is not terminal |
| Raw sampling | Controlled categorical model is sampled at its declared probabilities |
| Pair loss | Numeric fixture, swap symmetry, common-offset invariance, gradient sign |
| Gradient routing | Policy step changes only student parameters; head step changes only head parameters |
| Reductions | Full-batch and accumulated objectives match the declared weights |
| Routing | Null displacement gives no update; shared reference bypasses the head |
| Cache and resume | Incompatible metadata rejected; resume creates no duplicates |
| TBPO baselines unchanged | TBPO-Q/A outputs on a tiny config are bit-for-bit or tolerance-equal before and after the fork changes |

**Learned finite-data experiment.** Sample finite $\rho$ rollouts on a tree, learn $\widehat\mu$, train a tabular or tiny policy with the real pair loss. Report exact $\phi$, exact $\mu_\rho$, learned $\widehat\mu$, sibling errors, target KL, fitted KL and excess pair loss. Use both regimes (small displacement with large base gap, and the reverse). On a restricted candidate graph, report the unidentified outside mass.

**Tiny causal-LM smoke test.** Synthetic tiny models with a shared tokenizer: a base, a perturbed aligned teacher, and a distinct reference; the student is an exact copy of the reference. Run every stage, export, reload, and confirm the exported student reproduces its scores and generates without teachers or head. Label it synthetic.

## 13. Delivery order

1. Inspect the TBPO fork, write `CLAUDE.md`, confirm baselines run, add config validation and pure math helpers.
2. Port the tree checks and add the variable-EOS fixture; provide a CPU test command.
3. Scoring, collection sampling and causal features, with Section 12 tests.
4. Collection artifacts, head fitting, frozen targets.
5. Pair loss, hierarchical sampler, checkpoint and resume, learned finite-data experiment.
6. Tiny-model smoke test with export and reload.
7. Reproduce the 11.1 teacher pair and SFT student; then the evaluation adapters, cost reports and all variants and baselines.

Delivery is complete when another researcher can reproduce stages 1 to 6 from the README, all Section 12 checks pass, every invariant in Section 2 is traceable to code, and stage 7 has runnable configs. A passing smoke test is not a research result. Report commands actually executed, unresolved inputs and deviations. Do not claim benchmark improvements without measurements.

## 14. Prompt for Claude Code

> You are working in a fork of the TBPO (TokenRatio) repository. Read `reference/ChunkRatio_PD_v1_1_Coding_Spec_rev2.md` and `reference/ChunkRatio_PD_Method_1_1.md` first. Inspect the repository, write `CLAUDE.md` with verified commands and a mapping of the spec to existing code, and confirm the original TBPO-Q/A baselines still run. Then implement the stages in Section 13 in order, placing new code under `chunkratio/` and leaving TBPO baselines unchanged. Preserve the method invariants in Section 2: sequence objective, reference-rollout mean correction, one frozen scalar head, exact-case routing, and the soft pair loss with the same alpha on both sides. Start with CPU math tests and the tiny-model smoke test, and distinguish them from real LLM results. Use the ADPA/TVKD protocol in Section 11 for experiments. Do not add auxiliary losses, variance heads or estimator switching to the primary run. Report executed checks, unresolved inputs and deviations.

## 15. References

- ChunkRatio-PD v1.1 method document (in `reference/`).
- Nguyen et al. TokenRatio: Principled Token-Level Preference Optimization via Ratio Matching. ICML 2026. https://arxiv.org/abs/2605.12288 (code cited as github.com/dinhtruongng/TBPO)
- Gao et al. Advantage-Guided Distillation for Preference Alignment in Small Language Models (ADPA). ICLR 2025. https://arxiv.org/abs/2502.17927 (code: https://github.com/SLIT-AI/ADPA)
- Kwon et al. Preference Distillation via Value Based Reinforcement Learning (TVKD). NeurIPS 2025. https://arxiv.org/abs/2509.16965
- Gu et al. Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models (PAD). Findings of ACL 2025. https://arxiv.org/abs/2502.14272
- Hugging Face Transformers generation configuration: https://huggingface.co/docs/transformers/main_classes/text_generation
- PyTorch `binary_cross_entropy_with_logits`: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
