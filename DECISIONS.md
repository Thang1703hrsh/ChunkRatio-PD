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

## Second review of 2775bc1 (user request, 3 November 2026)

73. **hda reference is an assumption, not a verification (corrects item 62).** The weight check shows T+ is
    0.00044 (relative L2) from T0 against 0.0085 from Qwen3-1.7B-Base and 0.054 from Qwen3-1.7B, which supports
    T0 as T+'s DPO reference but does not prove it. The hda launcher now writes `teacher_reference_evidence.json`
    (`supported`, never `verified`), keeps `teacher_dpo_reference_verified: false`, and records
    `teacher_dpo_reference_assumption` (status `assumed`, claim, evidence, reason) in every config, so the
    manifest of every hda run carries it. `validate` accepts an unverified reference only for a pilot with that
    block and non-empty evidence; non-pilot runs still need verification.
74. **Launcher completion also checks code (amends item 67).** `finished()` in both launchers also requires the
    run's `manifest.json` to record the revision this code would record (`artifacts.code_revision(commit)`,
    now shared by pipeline and launchers). Results of other code are not finished: the run repeats, and its
    stage cache keys keep old shards out.
75. **Per-source sibling correlation (corrects item 70).** Centring by state does not centre one endpoint source
    or a bootstrap draw, so `_corr_r2` now takes Pearson correlation around the subset's own means; R^2 compares
    the given predictions with the labels around the subset's label mean. A case with Pearson -1 per source and
    +0.8 pooled is a test.
76. **Fallback metadata (amends item 69).** `selection_score` is always the score of the exported model and
    `best_checkpoint_score` that of the best trained checkpoint; with the fallback enabled the rule is named
    `select_mean_reward_with_reference_fallback`, and an exported rho reports rho's select score.

## Session of 3 November 2026: speed and the compact configuration (user requests)

77. **Speed-only changes (no change to any result).** RM selection generates the select prompts in
    left-padded batches (`selection_generation_batch_size`, 16 for hda runs; 1 is the original rule) and the RM
    is read on the first selection call, so an evaluate stage after a finished policy never loads it. The
    candidate and correction stages write `timing_<stage>.json` (CUDA-synchronized seconds per phase). hda x10
    uses `generation_batch_size` 48 and `policy_microbatch_size` 6. A smoke rerun with this code reproduced every
    `diagnostics.json` field of the earlier smoke except wall time.
78. **Compact configuration (user request; conflicts with the K=4 hard constraint in CLAUDE.md, explicit user
    instructions take precedence).** Opt-in keys, defaults unchanged: `candidates: 2` draws one rho and one T+
    sibling (draw i of a source keeps its stream, so the K=2 siblings are the first draws of K=4);
    `correction_seeds: endpoints` drops the prefix seeds (the head is queried only at candidate endpoints);
    `head_positions: boundary` fits the head only on rollout states on the chunk grid, (n + t) % L == 0, where
    every nonterminal endpoint lies. The launcher's `CRPD_COMPACT=1` sets these with `endpoint_fraction: 1`
    (every nonterminal endpoint is rolled out once), 2 states per prompt and selection every 50 steps (one pair
    per state makes the pool about six times smaller), and tags outputs and cache `-compact`. Compact results
    are a deviation from the protocol and are reported as such.

79. **Isolated fast x10 v1 (explicit user request, 4 November 2026).** Keep the pinned HDA
    1.7B teacher pair and 0.6B rho, H=512/L=32, raw categorical sampling, FP32 frozen
    scoring and matched student/rho autocast. Reuse the compact K=2/endpoint-only R=1/
    boundary-head protocol as an explicitly labelled deviation, with 100 policy steps,
    effective batch 64, alpha .03, lr 1e-6 and select-split RM at steps 0/50/100.
    Cross-state candidate scoring is opt-in (`batch_candidate_scoring`), bounded by
    `score_tokens_per_batch`; generation streams/deduplication are unchanged, while
    real accelerator batch-shape rounding is checked rather than claimed bit-identical.
    A bounded CPU feature cache avoids repeated attachment verification/reads during
    head epochs. Feature-only forwards request one logit position, not the entire prefix.
    Pretrained student export remains mandatory; `save_student_state_dict=false` skips
    the redundant second weights export only in this run. `chunkratio.fast_run` snapshots
    x10 prompts and writes a completion marker bound to config, source and exported-file
    SHA256 hashes only after all training diagnostics succeed. It never treats an early
    manifest as evidence of completion. Main/raw share compatible collection artifacts.
    `script/modal/crpd_fast_x10.py` uses its own app/volume, reads the existing x10 volume
    read-only, performs real-GPU score/initial-margin checks before training, and never
    publishes/pushes. Validation selects checkpoints; fixed Winogrande/TruthfulQA and
    a preselected student-held-out RM sample are evaluated afterward. Teacher exposure
    to those RM test prompts is not verified. This is x10, not full-scale training.

80. **Budgeted unattended smoke suite (4 November 2026, explicit user request).**
    Replace the potentially retried x10 orchestration with `script/modal/crpd_overnight.py`.
    With another smoke app running and no clarification received, leave it untouched,
    wait for it to finish (at most one hour), and reserve USD14.55 of GPU work plus
    USD0.45 for bounded CPU orchestration. This is resource/deadline-based protection,
    not a provider-side workspace billing cap. Each GPU job reserves its full timeout
    plus 180 seconds at USD4.50/A100-hour or USD3.00/L40S-hour before submission, with
    no refunds or logical retries. CPU/RAM have hard limits. GPU preemption may still
    happen; subprocesses retain an absolute deadline and overdue calls are cancelled
    by a non-preemptible CPU controller. Four recipes use 64/16/16 prompt groups and
    30 policy steps: compact main, raw, main with all-position head loss, and main
    with alpha .01/lr 5e-7. A positive select-RM gain may promote one to 256/64/32
    groups and 60 steps; this is exploratory. Only a positive medium validation gain
    triggers fixed test reporting (32 RM prompts, Winogrande and TruthfulQA).
    Reviewed package c784e569e184ca1f includes the separately edited correction batch
    knob; its default is unchanged. This is neither full-data training nor evidence
    that CRPD is superior. Every outcome, including timeouts, remains in the ledger.
81. **Sampler in the collection keys (bug fix).** `gpu_sampling` changes which tokens are drawn but was not in
    the collection cache keys, so switching it on a shared cache would have reused the other sampler's
    rollouts. `pipeline` now adds `sampler: gpu-inverse-cdf-fp64` to the common dependencies when it is set
    (default runs keep their keys; grouping/batch sizes never enter keys). The turbo x10 run used its own
    `cache-turbo` and is unaffected. Regression test `test_sampler_choice_is_part_of_every_collection_key`
    (fails without the fix).
82. **Launcher stage list.** crpd_raw and crpd_suffix no longer run collect-correction and fit-head, which only
    reloaded the three models (about 7–9 minutes in the turbo x10). `CRPD_ONE_PROCESS=1`, implied by
    `CRPD_TURBO=1`, runs validate-config then one `evaluate` call, loading the models once; per-phase
    collection seconds remain in `timing_*.json`.
83. **Shared-prefix scoring (opt-in `shared_prefix_scoring`).** `scoring.sibling_logprobs` runs each prefix once
    with its cache, repeats the cache rows for the prefix's chunks and scores them on it, with score_batch's
    masks. Used for candidate scoring (`fast.score_candidate_siblings`, at most `score_max_states`=48 states
    per forward because sibling FP32 logits otherwise grow with short prefixes) and for the policy pair
    scores of student and rho (c and d on one prefix pass; rho always recomputed the same way, so the margin
    is exactly zero at initialization). Tests: scores and gradients equal score_batch on TinyLM and a KV-cache
    Llama; end-to-end crpd_raw run equal to the default in FP32 and bf16. A100 measurements: T+ scoring of
    1,600 siblings 24.0 s vs 58.4 s (max difference 7.6e-4 nats, 33 GB); 128-pair policy step 2.89 s vs
    4.05 s at token budget 8192 (50 GB). `CRPD_TURBO=1` turns it on. Not adopted: a faster selected-token
    log-prob (about 1% of scoring FLOPs) and batched RM scoring (10-20 s per selection, padding risk).
84. **End-of-turn token unscored (opt-in `score_end_token: false`; user request; deviates from the hard
    constraint "include sampled EOS exactly once", explicit user instructions take precedence).** Diagnosis
    of the turbo crpd_raw x10 student: 0% EOS and 1,024-token responses on the 64 held-out prompts (rho
    98.4%, 247 tokens), 4-gram repetition 34% vs 15%, held-out RM -0.64 [-1.43, 0.11] vs rho. Cause: the hda
    T+ ends 26 of 32 greedy responses with <|endoftext|> (T0 and rho end all with <|im_end|>) and writes
    422 vs 214 tokens; at rho's own response ends T+ gives <|im_end|> log p -20.5 vs T0 -0.75, so pairs with
    one ending side (12.9% of the training weight) preferred the ending side only 8.2% of the time. The
    content signal is sound: on 256 independent UltraFeedback pairs delta_T ranks chosen above rejected 69.5%
    (T+ alone 56.6%, longer-wins 53.1%, correlation with length difference 0.07). With the option off, the
    first EOS and everything after it are unscored for every role (targets, student and rho margins), so
    ending is neutral and the student keeps rho's ending behaviour; the setting enters the candidate cache
    key. Launcher: `CRPD_NO_END_TOKEN=1`. The hda trio therefore violates the identical end-of-turn
    assumption; this is a workaround, not a verification of T+'s DPO reference.

## Session of 5 November 2026: consistency fixes and the gamma variant (user requests)

85. **Unscored end token: consistency fixes (bug fixes, results of finished runs unchanged).** (a)
    `script/modal/pd_diagnostics.py` section A scored the student and rho margins with the end token while the
    targets of a `score_end_token: false` run omit it; it now takes the rule from `resolved_config.json` and
    records it (`pair_margin_end_token_scored`). Sections B and C are unchanged. The `pairs` section of the
    earlier noend diagnostics is therefore not comparable with its targets; its held-out RM, EOS rate, length
    and repetition are unaffected. (b) `candidate_coverage` compared a student mass scored with the end token
    with rho's cached mass scored without it; both are now actual chunk probabilities (end token scored
    once), rho being rescored when the run leaves the end token unscored (`end_token_scored: true`). (c)
    Correction returns, the head and the exponential audit always score the end token, so
    `score_end_token: false` is now rejected for every variant except crpd_raw and crpd_suffix instead of
    silently mixing the two rules. (d) `verify_hda_teachers` reuses earlier evidence only when its recorded
    `sources` equal the current `HDA_CHECKPOINTS`, so a replaced teacher pair is checked again. Tests:
    `test_unsupported_gamma_and_end_token_combinations_fail_closed`,
    `test_candidate_coverage_scores_end_token_for_student_and_rho`.
86. **crpd_raw with a scaled teacher gap (opt-in `teacher_gap_scale` = gamma; user request; a new variant
    setting, not the protocol).** Diagnosis of the hda x10 raw students: the student is asked to reproduce the
    full teacher log-ratio gap (tens of nats per chunk) and drifts far from rho (no end of turn, 4-gram
    repetition 35%). With gamma in (0, 1], raw targets are t_gamma(c) = gamma * delta_T(c), so
    p = sigmoid(alpha * gamma * (delta_T(c) - delta_T(d))) while the student margin m = alpha * (a_c - a_d) is
    unchanged. Lowering gamma differs from lowering alpha: it changes the gap the student must match, not only
    the label temperature. The cached teacher log-probs are kept, so collections are shared; gamma enters the
    target key only when it is not 1 (gamma = 1 keeps every key and is the original target bit for bit), and
    the policy key carries the whole config. Only crpd_raw accepts gamma < 1: main would need
    gamma * (delta_T + mu_hat), and the shortcut rho = T0 => zero correction holds for the current target, not
    for a scaled one in general. Launcher: `CRPD_GAMMAS=1,0.3,0.1` with `CRPD_ALPHAS` and
    `CRPD_VARIANTS=crpd_raw`; runs are named `crpd_raw-alpha0p03-gamma0p3`, gamma = 1 keeps its old name and
    runs first so the others reuse its collection. Selection stays on the select split; held-out prompts only
    confirm. Wording correction: CRPD as implemented is offline distillation on data generated once by the
    frozen reference rho and the teacher T+, not on-policy in the updated student. UltraFeedback preference
    labels come mainly from GPT-4 ratings, so they are called available preference labels, not human labels.
    Tests: `test_teacher_gap_scale_scales_raw_targets_only`, `test_gamma_sweep_shares_collection_and_keys_targets`,
    `test_gamma_sweep_names_and_guards`.
87. **CRPD time optimizations that keep results (user request, 5 November 2026).** Reviewed against the code and
    the gamma runs (gamma 1: 36.8 min with collection, collect-candidates 601 s for 1,513 states; gamma 0.1:
    22.7 min reusing it). (a) Collection, correction, head and target keys carry `cache_revision` (hash of the
    chunkratio sources + Python, torch and transformers versions) instead of the git commit. The gamma
    collection was keyed `d192806-dirty+src.ef4f37bd` while HEAD 1c21d1d has the same sources, so a commit alone
    forced a recollection. Manifests and the policy key keep the full code_revision; any source or library
    change still gives new keys. Existing caches get new key names (they were reachable only from their own
    commit). (b) `collection_splits: ['train']` (crpd_raw and crpd_suffix only; validated): the head_val prompts
    are not collected. Each prompt has its own stream and shard, so train states, candidates, pairs, weights,
    targets and the trained student are identical (test); at x10 this skips 160 of 800 collected prompts. A
    store that also holds head_val shards returns the configured splits only. `candidate_coverage` and
    `counts.states` then cover train states only. (c) Launcher `CRPD_CACHE_TAG` names the cache apart from
    `CRPD_OUTPUT_TAG` (unset: unchanged behaviour). Turbo crpd_raw/crpd_suffix set `collection_splits: ['train']`
    and `save_student_state_dict: false` (the HF export holds the same weights). (d) `cost.stage_seconds` and
    selection `generate_seconds`, `rm_load_seconds`, `rm_score_seconds` record where wall time goes (timing only).
    Not done: lazy loading of the teachers when every collection stage is cached (needs trusted fingerprint
    records; estimated 1-2 min per reuse run, to be measured with (d) first); precomputed rho log-probs (pairs
    are seen about 2.7 times at x10 but about once in a full-data epoch, so little gain at full scale, and the
    exact-zero initial margin depends on scoring rho exactly as the student); smaller K or fewer steps (they
    change the data or the training, not the speed of the same method). Tests:
    `test_train_only_collection_keeps_raw_pairs_and_student`, `test_shared_store_gives_only_configured_splits`,
    `test_collection_splits_validation`, `test_cache_keys_ignore_commit_but_follow_sources`,
    `test_cache_tag_is_independent_of_output_tag`. Follow-up from an external (Codex) review: `collect_candidates`
    returned every record of a shared store (head_val included) and the cost counts read every prefix shard;
    both now follow the requested states/splits, and a full evaluate on a shared cache equals a fresh train-only
    run (counts, cost, coverage). `ShardStore.records()` read each shard twice (once for the job id, once in
    get()); it now reads once with the same name, job-id and checksum checks (`test_store_records_detect_corruption`).
88. **Post-training diagnostics per candidate and RNG digests in shards (user request, 5 November 2026; from the
    Codex review, items 2 and 3).** Neither changes data, targets, loss or gradients. (a) `pair_diagnostics`
    scores every candidate that appears in a pair once per model (`ChunkRatioTrainer.candidate_logps`, the
    training rule: student precision, end-token rule, one shared prefix pass per state with
    shared_prefix_scoring, cached FP32 rho exactly where pair_terms reads it), then evaluates pair_loss and the
    diagnostics for all pairs from those scores. x10: 4,770 candidates scored instead of 14,216 pair sides
    (about 3x fewer); at full data the saving grows with the pair count. Groups keep the former chunk-row
    budget. Values equal the per-pair pass up to rounding of the grouping
    (`test_pair_diagnostics_score_each_candidate_once`: default, shared prefix, unscored end token, token budget).
    (b) Shards record `rng_digest` (sha256 of the generator state bytes) instead of the state as a JSON integer
    list (19 KB each, eight per candidate shard). The before-state is that of `stream(seed, name)` and the
    after-state follows by replaying the job, so the digests identify both
    (`test_shards_record_rng_digests_that_replay`). Candidate shards: about 142 KB -> 2.3 KB in the smoke cache;
    x10 about 236 MB -> 4 MB of JSON written and read. Keys carry `rng_record: sha256`, so no store mixes the
    two formats. Nothing read the stored states.
    Measured on nuocb0739 (gamma 0.1 rerun `crpd_raw-alpha0p03-gamma0p1-g01`, 5 November 2026): 37.4 min including
    a first download of the 8B RM; stage seconds: collect_prefixes 171, collect_candidates 487 (601 with
    head_val before), post_training_diagnostics 102, train_policy 1,360 of which selection 687 (generation 235,
    RM load/transfer 375, RM scoring 77). Correction to 87(b): on the GPU the train records are not bit-identical
    after dropping head_val, because token-budget generation groups change and batched FP32 logits round
    differently near sampling boundaries (7,119 pairs vs 7,108; CPU tests with fixed grouping are identical).
    The draws remain exact categorical samples of the same distributions; select RM +1.12 [0.46, 1.78] (43/21)
    vs +1.18 [0.49, 1.86] (46/18) before.
89. **Gamma sweep 0.1 / 0.05 / 0.03 and the pinned best run (user request, 6 November 2026).** x10, alpha 0.03,
    150 x 128, nuocb0739, one collection (`CRPD_CACHE_TAG=shared`; the 0.05 and 0.03 runs collected nothing new,
    about 20 min each instead of 37). Select RM vs rho (choice split): 0.1 +1.12 [0.46, 1.78]; 0.05 +1.00
    [0.34, 1.65]; 0.03 +0.66 [0.09, 1.23]. Held-out test (64 prompts, confirmation only), student vs rho:
    0.1 RM +0.77 [0.02, 1.51] (40/24), EOS 82.8%, length 526, repetition 26.7%; 0.05 RM +0.54 [-0.20, 1.28]
    (43/20, sign p = 0.005), EOS 92.2%, length 370, repetition 21.3%; 0.03 RM +0.16 [-0.56, 0.88] (30/33),
    EOS 89.1%, length 388; rho EOS 98.4%, length 247, repetition 15.0%. `pd_diagnostics` now also reports
    per-prompt values and a length-controlled RM difference (measurement only): for 0.05 the RM difference
    falls with length (slope -0.0056 per token, correlation -0.33), the OLS difference at equal length is
    +1.23, and on the 59 prompts where both ended with EOS it is +0.93 (42/16, sign p = 0.001), so the gain is not
    a length artifact. Winogrande (hf fp32): 0.1 58.01 (rerun), 0.05 57.85, 0.03 58.09, all within noise of rho
    57.85 and of NARVAL riskkd 58.72. Rule applied: among runs whose held-out behaviour passes (EOS >= 90%), the
    best select score wins: gamma 0.05. Gamma 0.1 keeps the larger held-out RM gain but fails the EOS gate.
    Open: length is still 1.5x rho and repetition 21% vs 15%; the gamma 0.1 pd_diagnostics rerun with the
    length-controlled fields did not write a new file (the old result stands).
90. **2,000-prompt pilot of gamma 0.05 (user request, 6 November 2026), fixed before collection.** Launcher scale
    `x30` (own app and volume `crpd-qwen3-pilot-hda-x30`): crpd_raw, alpha 0.03, gamma 0.05, all turbo settings,
    2 states per prompt, 2,000 train prompts (train-only collection), batch 128, 450 steps (about x10's 2.7 passes
    over the pairs: about 22k pairs x 2.6), RM selection on the 64 select prompts every 150 steps plus the rho
    baseline. Same 20,480 UltraFeedback rows as x10, so the hash-ranked splits agree: identical 64 select and 64
    test prompts, and the train prompts contain x10's 640. Run name `crpd_raw-alpha0p03-gamma0p05-p2k`. After
    training, a server-side chain (`script/modal/eval_vukien/pilot_benchmarks.py::after_pilot`) starts held-out
    `pd_diagnostics` and Winogrande (5-shot) + TruthfulQA-MC2 (0-shot), hf fp32, lm-eval 0.4.9.2, for the pilot
    student, the x10 gamma 0.05 student, rho and NARVAL riskkd, with paired tests. Cost estimate from the measured
    x10 stage times: collection about 34 min (3.1x x10's 658 s), training about 34 min (4.5 s/step), selections
    about 18 min, other about 10 min: about 1.6 h on an A100-80GB at about $3.46/h with RAM and CPU (about $5.5),
    plus held-out diagnostics and benchmarks about $1.5: about $7-9 in total. nuocb0739 had used $12.08 of credit
    this month before launch.
