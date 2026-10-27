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
