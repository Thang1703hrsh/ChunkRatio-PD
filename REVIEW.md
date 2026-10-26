# ChunkRatio-PD v1.1 review

Reviewed commit: `ca684c9` (Codex, "crpd: implementation complete"). Review date: 27 October 2026.
Sources checked: `ChunkRatio_PD_v1_1_Coding_Spec_rev2.md`, `reference/ChunkRatio_PD_Method_1_1.md`,
`reference/ChunkRatio_Math_Checks.py` and its JSON. All review fixes are separate `review:` commits
after `ca684c9`. Reasoning for new settings is in DECISIONS.md items 13 to 21.

## Verdict: NOT READY for GPU runs

The mathematics is right and is now properly tested. The code is not ready to spend GPU budget on
the revision-2 protocol, for four reasons:

1. Required inputs do not exist. There is no verified teacher pair (T0 = SFT LLaMA-3.1-8B,
   T+ = its DPO), no SFT student, no data exports, and no pinned FsfairX revision. The in-repo
   preparation launcher (`chunkratio/preparation.py`) is single-process FP32 full fine-tuning with
   AdamW. An 8B model needs about 128 GB for weights plus optimizer state alone. The teacher
   stage therefore needs an external trainer (the inherited FSDP stack or similar) that records its DPO
   reference.
2. Nothing has run on CUDA or Linux. Every run here is Windows CPU on Python 3.15 with a
   nightly torch build. The research pin (Python 3.11, torch 2.6, CUDA 12.4) has not been
   exercised. Exact-resume equality is verified only on CPU. Two-GPU placement for the teachers
   (added in review) has not been run.
3. Collection cost is unmeasured and probably dominant. The sampler (`chunkratio/sampling.py:22`)
   has no KV cache and runs at batch size 1: every generated token reruns the whole context.
   The declared correction budget is every prefix state, plus half of the nonterminal endpoints,
   times `correction_replicates=4`, at H=512. Cache keys include the training seed, so each of
   the three seeds recollects. No throughput number exists to size the run.
4. Baselines and benchmark adapters are missing. ADPA, TVKD, DPKD, DCKD and SimPO are not in
   the repo, and the AE2, Arena-Hard and MT-Bench plans are templates (REPORT, Not done).

What is ready: the method math, routing, targets, loss, measure, masking and frozen-model
handling, all verified below. The full staged pipeline also runs end to end on real Hugging Face
classes (tiny local Llama checkpoints), including exact routes, RM checkpoint selection, export
and held-out RM scoring. The sensible next step, once checkpoints exist, is a small CUDA pilot
(a few hundred prompts, one seed, `crpd_main`) to measure collection throughput and memory
before committing to the protocol.

## Bugs found

Line numbers refer to `ca684c9`. Severity: High blocks a real run or silently corrupts it;
Medium is a spec violation or an unguarded correctness path; Low is hygiene.

| # | Where (`ca684c9`) | Severity | Problem | Fix | Commit |
|---|---|---|---|---|---|
| 1 | `chunkratio/collection.py:139-140` | High | Correction features batched all N visited prefixes, each up to P+N tokens, with every hidden layer kept. At H=512 that is tens of GB for one trajectory: OOM on any GPU. | `scoring.trajectory_features`: one causal pass, gather positions P+t-1 (spec §6.2) | `a779322` |
| 2 | `chunkratio/collection.py:140` | High | Features stored as JSON float lists (about 20 MB per trajectory at d=2048) and all loaded into Python lists by `fit_head`. Infeasible beyond toy scale. | Binary tensor attachments to each shard, hash recorded in the JSON commit and verified on load, loaded lazily | `2963f0f` |
| 3 | `chunkratio/value_head.py:73-83` | Medium | One full-batch optimizer step per epoch, so the declared 20 epochs meant 20 Adam steps for the head. | Optional `head_batch_size` (default full batch, so smoke is unchanged); research configs use 64 | `2963f0f` |
| 4 | `chunkratio/training.py:123,132` | High | RM selection (greedy 1024-token generations on every `select` prompt, FP32 8B RM) and a full resume checkpoint ran after every one of 1000 steps. | `selection_interval` cadence, final step always included, required for real configs (100 in research configs) | `b993789`, `02dcba6` |
| 5 | `chunkratio/data.py:54` | High | If the chat template ended assistant turns with a token other than tokenizer EOS (Llama-3 `<\|eot_id\|>` vs `<\|end_of_text\|>`), data passed silently, and the collection sampler would run every response to H. | `prepare-data` fails closed when the assistant turn lacks tokenizer EOS | `6a603c6` |
| 6 | `chunkratio/pipeline.py:36-40` | Medium | One prompt with prompt+H over context aborted the whole run. Spec §5.2 asks for a pre-generation drop rule. | Drop by rule, record ids in `manifest.json` (`ineligible_prompts`) | `119fefe` |
| 7 | `chunkratio/models.py:33,41` | Medium | All roles shared one device and one dtype: two FP32 8B teachers plus the 1B student and AdamW exceed one 80 GB GPU. | Optional per-teacher `device`/`dtype`; rho, student and head stay on the main device; cross-role increments combined on CPU | `7463002` |
| 8 | `chunkratio/targets.py:65` | Medium | Target formula duplicated inline (the tested `math.target_score` was unused) with no test. Flipping the `delta_T` sign, dropping `mu_hat` or removing terminal zeroing passed every test. | Call `math.target_score`; tests for sign, `mu_hat`, terminal zero and endpoint budget | `39a456d` |
| 9 | tests for `targets.py:82`, `training.py:64` | Medium | Hierarchical pair weights and the prompt/state/pair sampler untested: a flat weighting or a flat sampler passed. | Weight and sampling-frequency tests | `39a456d` |
| 10 | tests for `scoring.py:104,115`, `sampling.py:22` | Medium | Removing `no_grad` from frozen `generate`, `continuation_increments` or `state_features` passed: frozen parameters already lack `requires_grad`, so outputs looked fine. | Probe recording `torch.is_grad_enabled()` inside every forward | `4658e6e` |
| 11 | tests for `collection.py:83,125,133` | Medium | Correction rollout source (rho for main, T+ for `crpd_mu_teacher`) and the global budget when H-n < L untested. | Independent per-token recomputation of labels, features, budgets; H-n < L candidate test | `d0ff373` |
| 12 | tests for `math.py:77` | Medium | Every fixture used margin 0, so dropping `alpha` from the student margin passed. | Nonzero-margin loss and gradient fixture | `640c270` |
| 13 | `chunkratio/targets.py:67` | Low | `crpd_audit` audited the first N candidates in record order, often all rho-proposed. Spec §7.4 requires teacher-proposed endpoints too. | Equal per-source quota; sources recorded in each audit | `24ce94f` |
| 14 | `chunkratio/scoring.py:69`, `collection.py:73-147` | Low | Non-finite scores raised bare `AssertionError`s. Spec §6.1 wants record ids. | Collection jobs re-raise `ValueError` naming the candidate state or correction job | `bc630fb` |
| 15 | `script/*.ps1:2` | Low | With `$ErrorActionPreference='Stop'`, a caller redirecting stderr (`2>&1`, CI logs) turned torch's NumPy warning into a terminating error. | `Continue` around native calls; exit codes decide | `f9c79b9` |
| 16 | `pytest.ini:3` | Low | `output/pytest-temp` and `.pytest_cache` were left with ACLs this user cannot read (previous sandbox), so 5 tests errored at setup. | New basetemp and `cache_dir` under `output/` | `47b2e9f`, `59dbae2` |

Two paths had never run, and the review covered them without code changes. One is the
non-synthetic HF path: model loading, preflight on a real tokenizer API, Llama right-padding
scores, features, all three routes, RM selection, export and `rm-evaluate`. The other is the SFT
and teacher-DPO launchers. No bug was found in either. Commits
`7de9803` and `ab41f3a`; transformers 4.52.4 and numpy were installed into the active interpreter
for this. Documentation commits: `f592a1c` (DECISIONS), `a100da1` (REPORT and validation summary),
`d89a6c3` (mutation runner).

## Formula check against the spec

Current file:line. Each item was read against spec §§4 to 8 and the method equations, then pinned
by a test, a mutation, or both.

| Item | Location | Result |
|---|---|---|
| `p_hat = sigmoid(alpha*(t_c - t_d))`, detached | `math.py:62`, `:79` | Correct; target logit detached; alpha > 0 enforced |
| `m = alpha*(a_c - a_d)`, `a = log pi - log rho`, rho detached | `math.py:77` | Correct; student not detached |
| Loss `softplus(m) - p_hat*m` | `math.py:79` | Correct via `binary_cross_entropy_with_logits`, which equals it exactly |
| `t_s(c) = delta_T + mu_hat(sc)`, terminal 0 | `math.py:53`, `targets.py:58-69` | Correct; terminal zeroed twice (bypass and mask) |
| Endpoint features and budget `(H-n-|c|)/H` | `targets.py:61-63` | Correct |
| Suffix returns: reverse cumsum, terminal 0 | `math.py:31-40`, `collection.py:151` | Correct; Y_t belongs to the state before y_t |
| v1.0 ablation: T+ rollouts, labels `log rho - log T0` | `collection.py:137,147` | Correct |
| Routing order and exact identity | `math.py:43`, `pipeline.py:35,100`, `training.py:75` | Correct; fingerprints of weights, architecture, config and dtype; no tolerance |
| Hierarchical weights | `targets.py:88`, `training.py:64` | Correct; skips and non-train states excluded |
| Head objective: SSE / H per trajectory, mean over trajectories | `math.py:96`, `value_head.py` | Correct |
| Masks: prompt, padding by position, EOS once, nothing after | `scoring.py:63-66` | Correct, including EOS = PAD |
| Logit/label shift | `scoring.py:67`, `:132` | Correct (`logits[k]` predicts `ids[k+1]`) |
| `no_grad` on T+/T0/rho; FP32+ sums | `scoring.py:96,103-125`, `sampling.py:21`, `math.py:15` | Correct |
| Causal chunk boundaries and states | `collection.py:68`, `sampling.py:26` | Correct; chunk cutoff nonterminal, budget never reset |
| Collection vs evaluation decoding | `sampling.py:36`; `selection.decoding`, `evaluation_decoding` | Separate; raw temperature-1 categorical for collection |
| H=512, L=32, K=4 (2 rho + 2 T+), weight decay 0, head architecture | `config/crpd/*.json`, `config.py`, `value_head.py` | Match spec tables |
| Seven variants, reference rows, metrics | `config.py:9`, `evaluation.py:14-17` | Match; the "DPO" baseline is the `student_dpo` reference row |

Engineering choices the spec leaves open (alpha=.3, lr=1e-6, 1000 steps, batch 128,
`correction_replicates=4`) are declared in DECISIONS item 9 and were not changed.

## TBPO-Q/A unchanged

`git diff tbpo-baseline -- train.py trainers.py preference_datasets.py loss/ config/` shows two
things. First, a 7-line block at the top of `train.py` that dispatches to `chunkratio` only when
`sys.argv[1] == "--crpd"`, before the legacy imports. Second, the new files under `config/crpd/`.
`trainers.py`, `preference_datasets.py`, `utils.py`, `baseline_head.py`, `loss/`, `config/config.yaml`,
`config/loss/` and `config/model/` are identical to the tag. `tests/test_baselines.py` checks the
sources byte for byte. `tests/test_tbpo_runtime.py` runs the original and current TBPO-Q/A forward
functions and one optimizer step on a tiny model and gets bit-identical results. A full legacy
Hydra/FSDP job was not run (Unix-only imports, no GPU).

## Mutation check

Runner: `python script/mutation_check.py` (each mutation applied, `pytest -q -x` run, file restored).

| Stage | Mutations | Killed | Survived |
|---|---:|---:|---:|
| Codex tests at `ca684c9` (first 21 mutations) | 21 | 11 | 10 |
| Final tests | 29 | 29 | 0 |

The five required mutations, at `ca684c9` and now:

| Mutation | At `ca684c9` | Final |
|---|---|---|
| Flip the sign of `delta_T` | survived | killed |
| Drop `mu_hat` | survived | killed |
| Move `alpha` outside the sigmoid | killed | killed |
| Include prompt tokens in the mask | killed | killed |
| Remove `no_grad`: `score_batch` / increments / features / `generate` | killed / survived / survived / survived | all killed |

The other survivors at `ca684c9`: terminal `mu_hat` not zeroed, flat pair weights, flat pair
sampler, correction rollouts from T+ in the main variant, and budget reset at chunk boundaries.
`alpha` missing from the student margin was added later and also survived. Mutations killed by
Codex's tests from the start: forward cumsum, logit/label off by one, EOS excluded, post-EOS
included, reference sign in the margin, target not detached, routing order, head divided by
realized length. Added in review and killed: student detached, trajectory feature off by one,
correction budget ignoring n, endpoint budget ignoring the chunk, no-alignment route still
stepping, exact-zero route using the head.

## Final runs

```text
python reference/ChunkRatio_Math_Checks.py output/review/reference-checks.json
ALL CHECKS PASSED                      (all fields match reference/ChunkRatio_Math_Checks.json within 1e-12)

python -m pytest -q
53 passed in 27.12s

script/smoke_test.ps1 -Output output/smoke-review-final          (Windows PowerShell 5.1)
SMOKE PASSED: 7 variants, all stages, export/reload and standalone generation

bash --noprofile --norc script/smoke_test.sh --output output/smoke-review-bash
SMOKE PASSED: 7 variants, all stages, export/reload and standalone generation

python -m chunkratio learned-tree --output output/review/learned-tree.json
LEARNED TREE COMPLETED: synthetic finite-data experiment   (identical numbers to REPORT)
```

Smoke counts are unchanged from Codex's run: main 15 states, 54 pairs, 31 correction trajectories;
token 79 pairs; suffix 51. The audit variant now audits one rho-proposed and one teacher-proposed endpoint.

## Remaining doubts

- Sampler throughput. There is no KV cache and no batching across jobs (point 3 of the
  verdict). Batched cached sampling needs per-row generators to keep resume exact; it was not
  attempted in review.
- Collection is per seed. Cache keys include `seed`, so three training seeds mean three
  collections. That is a defensible reading (seed variance includes data sampling), but the
  spec does not decide it and it triples the dominant cost.
- File and JSON volume at scale. There is one feature file per trajectory, possibly hundreds
  of thousands in one directory. `targets.json` embeds the full state and both candidates in
  every pair, and `digest(targets)` serializes all of it again. The `evaluate` stage scores every
  pair with the student (two forwards per pair). All are correct but slow.
- Precision. The protocol keeps FP32 for every role. The teachers can now go on a second GPU,
  but that path has only run on CPU. bf16 teachers would fit one GPU, but forward noise may be
  comparable to small `log T+ - log T0` values; measure before choosing.
- EOS across the four models. Llama-3 checkpoints must share one EOS that is also the
  template's end-of-turn token. `prepare-data` now checks the data side. The tokenizer and
  generation configs saved by the external SFT/DPO trainer still need checking by hand.
- HF generation defaults. RM selection and `rm-evaluate` pass `do_sample=False`, but other
  fields in a checkpoint's `generation_config` (for example `repetition_penalty`) would still
  merge in. Inspect the resolved config before benchmark runs (spec §5.3).
- Anchor variant. `crpd_anchor` uses one random human pair per step for the DPO term. That
  is an unbiased but noisy estimate; check λ and its variance in the pilot.
- Synthetic selection. Synthetic runs select by negative training-pair BCE. It is labelled as
  a diagnostic and never used for real runs, but it is not a held-out rule.
- Mutation coverage is the 29 listed mutations, not an exhaustive search.
- Reference file names. The spec names `ChunkRatio_PD_Revision_Checks.*`, but the supplied
  files are `ChunkRatio_Math_Checks.*` (DECISIONS item 2). Hashes in `reference/HASHES.txt` match
  the working files.
