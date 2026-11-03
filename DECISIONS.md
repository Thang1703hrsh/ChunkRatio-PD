# Decisions

1. **Precedence**: explicit task > coding spec > method > executable checks >
   inherited conventions, as requested. The spec itself gives mathematics to the
   method; no mathematical conflict has been found.
2. **Reference filenames**: supplied `ChunkRatio_Math_Checks.*` are the actual
   executable sources; spec's Revision_Checks names are aliases from an earlier
   delivery. Preserve supplied files and hash them rather than invent fixtures.
3. **Reference numbers**: Appendix D's historical full ternary-tree table differs
   from the supplied variable-EOS script. Match the executable JSON fixtures;
   retain early EOS and report the discrepancy, not a changed research claim.
4. **Environment**: active interpreter is Python 3.15.1. Initially no torch,
   numpy, transformers or legacy training stack was installed. Stable PyPI has
   no compatible torch wheel; the official CPU nightly index supplies a cp315
   wheel. Install that into the same active interpreter (no interpreter switch).
   Record the exact nightly build and actual executed results in REPORT.md;
   initial dependency-blocked attempts are not successful smoke runs.
5. **Baseline marker**: `tbpo-baseline` was absent. Created it at the starting
   HEAD before shared changes. Source comparisons supplement (do not replace)
   unavailable runtime baseline equivalence checks.
6. **Integration**: opt-in `train.py --crpd ...` dispatch before legacy imports
   avoids the inherited Unix-only resource/FSDP/dependency initialization on
   Windows. CRPD owns its offline batch construction and loop; legacy paths stay
   unchanged. Distributed CRPD is not claimed.
7. **Research inputs**: exact checkpoint revisions, split data, judge versions,
   hardware and throughput are unresolved inputs. Fail closed for missing input;
   never fabricate benchmark scores or a measured GPU-hour estimate.
8. **Eligible measure**: prompts/states with fewer than two distinct candidates
   are recorded as skipped and excluded from policy sampling. Normalize over
   eligible prompts, then their eligible states, then all unordered pairs.
9. **Engineering settings**: the spec fixes H=512, L=32, K=4, the preparation
   recipes and weight decay=0. It does not fix CRPD alpha/lr/steps. Configs use
   alpha=.3, AdamW lr=1e-6, effective batch=128, microbatch=1, 1000 policy steps,
   20 head epochs at lr=.001 and four correction replicates. These are declared
   starting settings, not validated optima or values borrowed from teacher beta.
10. **Sampling resume**: independent semantic job seeds plus saved before/after
    RNG state make a partially written job restart identically. Completed shard
    records are immutable; interrupted uncommitted work is recomputed from its
    original seed, never from an outcome-dependent retry.
11. **Audit variant**: crpd_audit fits the main head and reports independent
    exponential-moment estimates with bootstrap uncertainty. It is diagnostic,
    takes zero policy optimizer steps, and exports the initialized student.
12. **External comparisons**: official ADPA/TVKD/DPKD/DCKD/SimPO implementations
    are not present. Do not substitute CRPD variants. Benchmark CLI plans require
    pinned external installations and official generation/annotator configs;
    generated plans are reviewable templates, not verified benchmark results.

## Review (27 October 2026)

Reviewer decisions. Each fix is in a separate `review:` commit; REVIEW.md lists them.

13. **Single target formula**: `build_targets` now calls `math.target_score`
    instead of repeating the formula inline, so one tested function defines
    `t_s(c) = delta_T + mu_hat` with terminal `mu_hat = 0`. Values are unchanged.
14. **Correction features**: the spec (§6.2) defines the feature of the state
    before `y_t` as the hidden state at position `P+t-1`. One causal forward pass
    over prefix+suffix gives all of them, and the per-prefix batch (all hidden
    layers for N prefixes of length up to P+N) cannot fit in GPU memory at H=512.
    Features are stored as binary tensor attachments to each correction shard.
    The JSON record commits only the content hash, which is verified on every
    load. This replaces JSON float lists (about 20 MB per trajectory at d=2048),
    which ruled out any LLM-scale run.
15. **Head minibatches**: the objective is unchanged (sum over a trajectory /
    H, mean over the trajectories in an optimizer step). Codex's loop took a
    single full-batch step per epoch, so the declared 20 epochs meant 20 Adam
    steps. `head_batch_size` (trajectories per step, shuffled by the
    `head-batches/<epoch>` stream) is optional and defaults to full batch,
    which keeps smoke results identical. Research configs use 64, a declared
    starting setting rather than a tuned value.
16. **Selection cadence**: reward-model selection on `select` prompts and the
    full resume checkpoint now run every `selection_interval` steps and at the
    final step. Synthetic runs default to 1. Real configs must set it, and
    research configs use 100 (10 RM selections in 1000 steps). The selection
    rule itself is unchanged and identical for every method. Resume restarts
    from the last checkpoint with its saved sampler/RNG state, so it stays exact.
17. **Ineligible prompts**: following spec §5.2, prompts with
    len(prompt)+H > the smallest model context are dropped before generation,
    and their ids are recorded in the manifest. Previously one such prompt
    aborted the run. A model without a known context length still fails closed.
18. **EOS convention**: `prepare-data` now rejects chat templates whose
    assistant turn does not contain the tokenizer EOS (for example Llama-3
    `<|eot_id|>` with `eos_token=<|end_of_text|>`). The collection sampler stops
    only at that EOS, so such a template would silently run every response to H.
19. **Audit endpoints**: spec §7.4 requires replicated audits on fixed
    endpoints, including teacher-proposed ones. The audit quota is now split
    equally between proposing sources (first source of a deduplicated
    candidate), in record order and independent of outcomes.
20. **Device and dtype per teacher**: teachers may set their own `device` and
    `dtype`. The student reference, student and head share the top-level device;
    validation enforces this. Defaults are unchanged (all FP32, one device).
    The protocol still declares FP32 because bf16 forward noise can be
    comparable to a small `log T+ - log T0`. Placing the teachers on a second
    GPU is the intended way to fit two FP32 8B teachers.
21. **Not changed**: `correction_replicates=4` (Decision 9) is kept as Codex's
    declared setting, although the spec does not require replicates in the
    training data and it multiplies correction cost by 4. Revisit it after
    measuring pilot throughput, and decide before observing returns.

## Pre-registration (NARVAL pilot)

Fixed on 27 October 2026, before any pilot collection or return was observed.

22. **Shared collection.** The three policy-training seeds share one collection. Splits,
    synthetic models, prefix pool, candidates, endpoint selection, correction rollouts, the
    head and its minibatch order all use `collection_seed` (default 0). The training `seed`
    is absent from their cache keys. Training seeds vary only policy sampling order, pair
    sampling and minibatch RNG; the policy cache key keeps `seed`. Between-seed spread
    therefore measures optimisation noise, not data-sampling noise; report it that way.
23. **No training-data replicates.** `correction_replicates = 1` in the NARVAL protocol
    configs; the spec does not require replicates. Replicated estimates stay in `crpd_audit`
    and in `diagnose-displacement`. The ADPA/TVKD configs keep their declared value 4.
24. **State subsampling.** `states_per_prompt` (int, or null for all). Positions come from
    the causal schedule of the frozen-rho prefix trajectory (every L tokens, nonterminal).
    The prompt state n=0 is always kept; the other k-1 are drawn uniformly without
    replacement with stream `state-select/<prompt_id>` from `collection_seed`. Selection
    happens inside `collect-prefixes`, before any candidate or correction exists. It reads
    only the position list, the prompt id and the seed. Pilot value: 4. Hierarchical
    weights are unchanged (1/N_prompts x 1/N_states x 1/C(K_s,2)) over selected states.
25. **Batched sampling.** One generator per row, named by the job's semantic id
    (`prefix/<prompt>`, `candidate/<state>/<source>/<i>`, `correction/<job>`,
    `exponential-audit/<job>/<r>`). `generation_batch_size` is excluded from cache keys
    because results do not depend on it. Candidate streams changed from one per source to
    one per draw, so caches from earlier revisions are not reused (the code revision is
    in every key anyway).

## NARVAL protocol (prompt 03 implementation)

26. **Second protocol, same method.** `"protocol_name": "narval"` adds validation rules but no
    mathematics: variants `crpd_main`, `crpd_raw`, `crpd_anchor` only; `correction_replicates = 1`;
    Qwen3 pairs fix `eos_token = <|im_end|>` and `enable_thinking = false`. Outside synthetic runs
    every `null` fails closed with the complete list of open inputs.
27. **Padded vocabularies.** Preflight now accepts an output layer larger than the tokenizer
    (Qwen3 pads 151,669 tokens to 151,936 logits) only when all four models have the same padded
    size. Sampling still uses softmax over every raw logit, as spec §5.3 says; trained models put
    negligible mass on padded ids, and the tiny random models do not, which is harmless.
28. **UltraFeedback splits.** The first pair per `prompt_id` (file order) is kept; later duplicates
    are listed in the split manifest. `head_val` and `select` are held out of `train_prefs` by a
    rank rule on sha256([seed, 'uf-split', prompt_id]) with declared fractions 0.05/0.05 (at least
    one prompt each), so small files cannot leave a split empty. `test_prefs` is all `test`.
29. **Synthetic runs with local checkpoints.** A synthetic config that names local model paths now
    loads them with the HF classes (narval-tiny); TinyLM is used only when no path is given.
30. **Shared cache per pair.** `cache_root` lets the three variants reuse every stage whose key
    matches. Keys already include the schedule, candidate rule, correction direction, head
    settings (now including `head_batch_size`), variant (targets) and full config (policy).
31. **Report layout is provisional.** `report --layout narval` uses the metric columns named in the
    task text. The NARVAL paper was not read here, so the exact Table 1/2 column order and metric
    variants (ARC, TruthfulQA) remain open inputs.

## Qwen3 pilot on Modal (user request, 27 October 2026)

32. **Pilot mode.** `"pilot": true` configs must list `pilot_assumptions`; validation then requires
    every training input but lets evaluation fields and reference rows stay null. The NARVAL
    protocol files are unchanged and still fail closed.
33. **Assumed recipes.** With NARVAL's SFT/DPO settings unknown, the pilot uses the launcher's
    ADPA/TVKD recipe (SFT 3 epochs/2e-5/128; DPO beta 0.05/5e-7/2 epochs/128), max_length 2048
    (examples longer than that are dropped, never truncated), Base checkpoints, and `<|im_end|>`
    forced as EOS. TF32 is enabled for SFT/DPO training only; CRPD scoring keeps full FP32.
34. **Pilot subset.** 2000 train, 200 head_val and 64 select prompts, ranked by
    sha256([seed, 'pilot', group_id]) before any generation. Teacher DPO still uses all of train_prefs.
35. **Hardware.** A GPU is attached only where a stage needs one: L40S for the 0.6B SFT and the
    diagnostic, A100-80GB for 1.7B training and for CRPD (which also holds the 8B FsfairX RM in FP32).

## Comparison with the NARVAL reference code and pilot tuning (user request, 27 October 2026)

The user pointed to github.com/PhiVanDat123/NARVAL as working reference code and asked for tuned
CRPD settings without having to follow NARVAL's. That repository holds only the NARVAL
distillation stage (`scripts/run_narval.py`); it has no SFT or teacher-DPO code to compare.

36. **No CRPD defect found by the comparison.** Differences from NARVAL are by design: CRPD scores
    rho rollouts in chunks with frozen FP32 labels, uses no log-ratio clipping and drops rather
    than truncates over-length data (spec §§5.2, 6.1). CRPD's prompt ends with the generation
    prompt and the continuation keeps EOS exactly once, which NARVAL's code also intends.
37. **Probable defect in the reference, not adopted.** In NARVAL's `apply_chat_template` (task dpo)
    `text_chosen` is the whole templated conversation, and `tokenize_row` prepends the prompt again,
    so the prompt appears twice and its second copy is scored. CRPD does not copy this.
38. **SFT turn fitting (pilot).** `prepare-data --fit-max-length N` supervises, for each ShareGPT
    conversation, the latest assistant turn whose conversation fits N tokens; earlier turns stay as
    context and no response is cut. On Deita with the Qwen3 tokenizer at 2048 this keeps 9,708 of
    9,827 conversations (17.6M tokens) instead of 532 (1.0M), which left SFT with about 15 steps.
39. **Mixed precision (pilot).** `autocast_dtype: bfloat16` (SFT/DPO) and `policy_autocast_dtype:
    bfloat16` (student forwards only) follow NARVAL's bf16 training while weights and AdamW stay FP32,
    so 5e-7 updates are not lost to bf16 rounding. `score_tokens` still log-softmaxes in FP32, and
    frozen-model scoring in collection and targets runs without autocast (rho in the policy loss: item 51).
40. **Policy schedule (pilot).** Opt-in `lr_schedule: cosine` with `warmup_ratio: 0.1` at peak lr
    1e-6 (mean about 5.5e-7, near NARVAL's 5e-7 cosine/10% warmup); `policy_microbatch_size: 8`
    changes speed, not the objective. Defaults stay constant lr and FP32, so primary configs are
    unchanged. `alpha` stays 0.3: it rescales targets and margins alike, so its best value needs a
    select-split sweep, which has not been run.
41. **Launcher limits.** SFT/DPO stages run with `retries=0` and 4 h / 16 h timeouts, and SFT/DPO and
    the policy loop print one JSON line per optimizer step so loss can be watched remotely.

## Launchers for the 8B NARVAL pairs (user request, 28 October 2026)

42. **Pair status.** All three NARVAL protocol files existed; only Qwen3-1.7B -> Qwen3-0.6B had a
    runnable launcher (`script/modal/qwen3_pilot.py`, unchanged). `script/modal/narval_pair.py`
    now runs Qwen3-8B -> Qwen3-1.7B and Llama-3.1-8B -> Llama-3.2-1B with the same stages, chosen
    by `CRPD_PAIR`, one Modal app and volume per pair. Nothing has been run for either pair.
43. **Per-pair tokenizer rules.** `chunkratio/pilot.py::PAIRS` fixes, per pair, the Base models,
    one tokenizer shared by every role, the EOS and the chat-template arguments. Llama Base repos
    ship no chat template, so all Llama roles use the Llama-3.2-1B-Instruct tokenizer (same
    vocabulary) with EOS `<|eot_id|>` and `date_string` pinned to "26 Jul 2024"; otherwise the
    template inserts the current date and prompt token ids change from day to day.
    `validate_narval` enforces both for Llama pairs once `eos_token` is set. SFT accepts an
    optional `tokenizer`/`tokenizer_revision`; DPO reuses the tokenizer the SFT checkpoint saved.
44. **Hardware.** In-repo SFT/DPO is single-GPU full FP32 AdamW, about 16 bytes per parameter
    (128 GB for 8B, plus a 32 GB frozen reference for DPO), so 8B SFT/DPO and the CRPD stage
    (two FP32 8B teachers, the 1-1.7B student with optimizer, rho and the FP32 8B RM) use B200
    (192 GB); the diagnostic uses H200; student SFT uses A100-80GB (1.7B) or L40S (1B).
    Generation batch 16 for these pairs. Memory fit and run time are unmeasured.
45. **Shared checkpoint not exploited.** With the same recipe, the Qwen3-8B pair's rho (SFT of
    Qwen3-1.7B-Base) is the Qwen3-1.7B pair's T0, as in NARVAL Table 1. The launchers do not yet
    reuse it across pairs; each pair trains its own copy.

## Published teacher pair for Qwen3-1.7B -> Qwen3-0.6B (user request, 29 October 2026)

46. **Hub teachers.** `CRPD_TEACHERS=hub` uses T0 = ayushshah/Qwen3-1.7B-UltraChat-SFT@698d09a4 and
    T+ = ayushshah/Qwen3-1.7B-Chat@2e46c30d instead of training them. The model cards are not
    evidence: `chunkratio/hub_teachers.py` rebuilds T0 + (alpha/r) B A from the published DPO LoRA
    (ayushshah/Qwen3-1.7B-Chat-LoRA@c1a6e68d) and requires every untouched tensor to be bit-identical,
    T+ to differ from T0, and every LoRA-touched element of T+ to lie within 2 ulp of the stored
    dtype of the rebuilt value. This is a provenance check of a 16-bit merge, not one of the exact
    routing identities, which stay exact. The run stops before any GPU stage if it fails.
47. **rho for hub teachers.** The official Qwen3-0.6B uses a thinking chat template whose prompt
    tokens differ from the teachers' plain ChatML, and it emits <think> blocks the teachers never
    saw. rho is therefore our full SFT of Qwen3-0.6B-Base on the first 10,000 UltraChat-200k
    train_sft conversations (the teachers' SFT corpus), turn-fitted at 2048 tokens, with the
    teachers' tokenizer and template shared by every role.
48. **Reduced hub pilot, fixed before collection.** 1000 train / 100 head_val / 64 select prompts,
    states_per_prompt 2 (the pre-registered 4 halved for cost), 300 steps x 64 pairs, RM selection
    every 150 steps, crpd_main and crpd_raw, diagnostic on 32 prompts x 8 replicates.
49. **Known contamination risk.** T+ was trained on argilla UltraFeedback-cleaned pairs, which share
    the UltraFeedback prompt pool with H4 test_prefs; held-out prompts seen by T+ must be removed
    before any held-out result is reported.
50. **Merge check measured against operands.** The first check on the published pair found every
    untouched tensor identical and 99.85% of LoRA-touched elements exactly equal to bf16(T0 + LoRA),
    but a 255.75-ulp worst case: ulp had been taken of the result, and where T0 and the LoRA update
    nearly cancel, a bf16 sum's rounding (bounded by its operands) is many ulp of a near-zero result.
    The tolerance is now 2 ulp of max(|T0|, |LoRA update|, |result|); unrelated or tampered T+ still fail.

## External code findings (user request, 1 November 2026)

51. **Rho shares the student's precision under autocast (corrects item 39).** With
    `policy_autocast_dtype`, the student ran in bf16 while `a = log pi - log rho` used rho scores
    cached in FP32, so the margin at student = rho was not zero. Rho is now rescored under the same
    autocast in `batch_loss` and in the anchor term, as teacher DPO already does in
    `preparation.py`. Chosen over dropping autocast because FP32 student memory on the A100-80GB is
    unmeasured. Cost: one extra no-grad rho forward per side and microbatch, not yet timed. Targets,
    T+/T0 scoring and collection stay FP32; the default FP32 path still uses cached scores.
52. **Diagnostic prefix cache identity.** `diagnose-displacement` keys its prefix shards by model
    fingerprints, tokenizer, prompt hash and code revision in a `prefixes-<key>` folder, as the
    pipeline does, and records the identities in `displacement.json`. Prefixes in an old
    `<output>/prefixes` folder are not reused.
53. **Forward token count.** `ForwardMeter.input_tokens` counts each forward's new non-padding input.
    Rows already finished in a batch still count, because the model really processes them.
    `forward_input_tokens_this_invocation` in earlier `diagnostics.json` files is inflated for
    KV-cached sampling. Wall times and the collection generated/scored counts are not affected.
54. **Full report completeness.** A report without `--allow-incomplete` needs every present or
    required method on both tracks, checked before the seed count. Reference rows are included
    because the specified main table has both track columns for each method.
55. **Alpha sweep rule, fixed before its results.** `CRPD_ALPHAS` trains one run per alpha on the same
    collection. The winner is the highest best-checkpoint FsfairX score on `select`, compared with
    rho's score there (`selection_baseline`, recorded as step 0 and never a selection candidate).
    Test prompts play no part. A stability run has only 16 select prompts, so a winner there is a
    lead to confirm at larger scale, not a result.
56. **Microbatch 4 for hub runs.** The first stability run ran out of memory on the A100-80GB at policy
    step 3, twice with the same numbers. Its prompts reach 1224 tokens against a median of 87, and
    right padding gives every row of a microbatch the longest width. Hub and stability scales now use
    `policy_microbatch_size: 4`; the objective is the same weighted sum over the sampled batch.
57. **x10 scale (user request, 1 November 2026), fixed before collection.** 640 train / 160 head_val /
    64 select prompts from 20,480 UltraFeedback rows, 2 states per prompt, rho SFT on 10,000 UltraChat
    conversations, 300 steps x 64 pairs, RM selection every 150 steps, alpha 0.3, generation batch 32.
    Select is 4x rather than 10x because selection generates one prompt at a time. `max_prompt_tokens:
    2048` drops longer prompts before generation (none of the 2,048 prepared rows, whose longest has
    1,973 tokens); ids are recorded in the manifest. A rule on prompt length only, never on outcomes.
58. **Alpha 0.1 at x10 scale, fixed before its result (user choice, 1 November 2026).** The stability sweep
    on 16 select prompts favoured alpha 0.1 (best -4.397; 0.3: -4.671; 1.0: -4.434; rho: -5.282). That lead is
    tested on the x10 collection by training alpha 0.1 after alpha 0.3 finishes, with the same seed, steps and
    64 select prompts. Alpha 0.1 is confirmed only if its best select score exceeds alpha 0.3's and it stays
    above rho at every checkpoint. Test prompts play no part, and the launcher code must not change between the runs.

## Correction speed (user request, 3 November 2026)

59. **Batched correction scoring.** `collect_correction` scores each generation group's suffixes with
    `continuation_increments_batch` (T+ and T0) and `trajectory_features_batch` (rho) instead of one forward
    per trajectory and role. Rows are sorted by length and cut so rows x longest row stays within
    `score_tokens_per_batch` (default 8192, read with `cfg.get`, not part of any cache key). Rows are
    left-padded with position ids, so every suffix ends at the last column: teachers return logits for the
    last max-suffix + 1 positions only, and rho returns one position's logits plus its final hidden states.
    Pending correction jobs are ordered by prefix length before grouping. Shard records keep their format.
    Values match the per-trajectory functions to 1e-5 (tests on TinyLM and a tiny HF Llama), not bit for bit.
60. **Sampler rows leave the batch when finished; GPU sampling declined.** `generate_batch` drops a row
    and its KV-cache entries once it samples EOS or reaches its budget, and copies one softmax for the whole
    batch to the CPU per step instead of one per row. Draws still use each row's CPU generator, so the saved
    `rng_before/after` states stay portable and resume stays exact. Sampling on the GPU was not adopted:
    CUDA generator states differ from the CPU streams, so every collected token would change and the stored
    stream states could not be replayed on another machine. Item 53 changes accordingly: finished rows no
    longer count in `ForwardMeter.input_tokens` because they are no longer forwarded.
61. **Revision of uncommitted code (superseded by item 65).** Launch revisions were `git HEAD` plus `-dirty`,
    which does not change when uncommitted code changes, so shards of different code could share a cache key.

## hoangducanh trio and external review (user request, 3 November 2026)

62. **Published NARVAL-matching trio (`CRPD_TEACHERS=hda`, pair `qwen3_1p7b_to_0p6b_hda`).** T0 =
    hoangducanh1865/qwen3-1.7b-deita-sft-student, T+ = hoangducanh1865/qwen3-1.7b-ultrafeedback-dpo-teacher,
    rho = hoangducanh1865/qwen3-0.6b-deita-sft-student, pinned to full commits in `chunkratio.pilot.HDA_CHECKPOINTS`.
    Chosen because on Winogrande 5-shot and TruthfulQA-MC2 0-shot (lm-eval 0.4.9.2, bf16) the 1.7B pair is within
    0.6 points of the NARVAL Table 1 teacher rows and rho within 1.4 points of its SFT row
    (script/modal/benchmark_eval.py). Nothing is trained before CRPD. T+ has no model card: its trainer_state shows
    1 epoch of DPO on 57,673 UltraFeedback pairs, with rewards near 0 at step 1, but does not name the reference.
    Without a published LoRA, `hub_teachers.nearest_reference` checks only that T0 is strictly the nearest of
    {T0, Qwen3-1.7B-Base, Qwen3-1.7B} to T+ in relative L2. That is evidence, not proof, and is recorded as such
    in every config's assumptions; the run stops if it fails.
63. **EOS of the trio.** The three tokenizers declare `<|endoftext|>`, but their shared ChatML template ends each
    assistant turn with `<|im_end|>`, and greedy decoding of rho ends with `<|im_end|>` (checked on two prompts).
    `<|im_end|>` is therefore the EOS of every role. `load_models` now sets a declared `eos_token` on every
    tokenizer, as prepare-data and SFT already did; an `eos_token` outside the vocabulary fails.
64. **Duplicate groups over the whole input.** `prepare_prompts` joins rows that share token ids or a group_id
    transitively (union-find) before deduplication, so prompt X in groups A and B makes B's other prompts part of
    X's group, and any split difference inside a group fails. Rows without a split take one hashed split per
    group (its smallest alias), which equals the previous rule for prompts without duplicates.
65. **Code identity.** `code_revision()` is the commit plus `+src.` and a hash of the chunkratio sources with LF
    line endings. Two different uncommitted versions never share a revision, and therefore never share stage
    cache keys. All earlier caches are keyed without it and are not reused.
66. **Policy resume and the step budget.** With warmup or cosine the learning rate depends on `steps`, so for
    those schedules `steps` is part of the policy checkpoint identity and a run cannot be extended on resume.
    A constant schedule without warmup can still be extended. A checkpoint past the requested steps is refused
    instead of being reported as the requested count.
67. **Launcher completion.** `qwen3_pilot.py` and `narval_pair.py` skip a run only when its marker exists and
    its `resolved_config.json` equals the config about to run. Another config's results in the same output
    folder stop the launcher instead of being skipped.
68. **NARVAL report completeness.** `report_narval` refuses rows without a seed, duplicate (method, pair, seed)
    rows and mixed `protocol` fields. Without `--allow-incomplete` it also needs every Table 1/2 metric per row
    and three seeds for crpd_main and crpd_raw, as `report_tables` already did.
69. **Selection against rho (diagnostic, opt-in fallback).** The RM selector records per-prompt rewards; with
    `selection_baseline` the policy result reports `improved_over_reference` and the paired per-prompt
    difference of the best checkpoint from rho (mean, normal 95% interval, wins/losses). Opt-in
    `selection_fallback_to_reference` exports rho when no checkpoint beats it and reports `selected: reference`.
    It is a different selection rule from the primary protocol and must be named when used.
70. **Sibling correction audit (diagnostic only).** `diagnostics.json` gains `correction_audit`: for the frozen
    pairs, the spread of mu_hat differences against delta_T differences and the measure-weighted share of pairs
    whose preferred side mu_hat flips, by proposing sources; for head_val endpoints of one state, the centred
    correlation and R^2 of mu_hat with single-rollout returns by endpoint source, with prompt-bootstrap 95%
    intervals. The head objective and its selection are unchanged; a head trained or selected for sibling
    differences would be a separate variant.
71. **Scoring compute.** `score_batch` (candidate scoring, policy training, anchor, coverage) left-pads with
    position ids and asks the LM head only for the last longest-chunk + 1 positions (`tail_logprobs`, shared
    with correction scoring), instead of [rows, width, vocab] logits with gradients. Masks are unchanged: prompt
    and padding unscored, first EOS scored once, later tokens not. Pair diagnostics in evaluate run in
    microbatches of `diagnostic_batch_size` (default 32) with the same hierarchical weights, and build_targets
    computes the mu_hat features of a state's siblings in one forward. Values match the previous code to float
    tolerance, not bit for bit. No A100 timing exists yet.
72. **bf16 rho rescoring kept.** Caching rho scores per pair in bf16 was not adopted: under autocast a score
    depends on the microbatch it is computed in, so a cache would break the exact zero margin at student = rho
    that item 51 restored. The rescoring forward is now tail-only (item 71), which removes most of its cost.
