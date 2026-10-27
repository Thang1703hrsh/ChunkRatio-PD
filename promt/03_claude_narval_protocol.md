# Task for Claude Code: add a NARVAL-comparable protocol and pilot prerequisites

You are running unattended. Never ask questions or stop for confirmation. Finish every step,
then stop. Read CLAUDE.md, PLAN.md, DECISIONS.md, REPORT.md and REVIEW.md first.

## Rules
- Same rules as prompts/02_claude_review.md: active interpreter only, one commit per change
  (`narval: <what>`), never `git push`, never rewrite history, never delete outside the repo.
- Do NOT change the method mathematics, the loss, routing, targets or any variant semantics.
  The existing ADPA/TVKD protocol (spec §11) must keep working unchanged. Add the new
  protocol as a second, config-selected protocol.
- Every value you cannot verify (few-shot counts, beta, lr, epochs, max length, checkpoint
  revisions, judge versions) must be `null` in configs and fail closed in `validate-config`.
  Never guess them. Record each open input in REPORT.md under "Open inputs (NARVAL)".
- Keep everything runnable on Windows CPU (tests, smoke) and Linux CUDA.

## Step 0: verify the git baseline
DECISIONS.md item 5 says `tbpo-baseline` was absent and Codex created it. Check that the tag
points to the original TBPO commit, i.e. no chunkratio/, reference/ or prompts changes are
inside it beyond what the overnight script committed. Record the tag's commit, its parent
and `git show --stat` in REPORT.md. If the tag is wrong, do not move it; add a new tag
`tbpo-original` at the correct commit, point tests/test_baselines.py and
tests/test_tbpo_runtime.py at it, and explain in DECISIONS.md.

## Step 1: batched KV-cached collection sampler (prerequisite for any pilot)
- Replace the batch-1, no-KV-cache loop in `chunkratio/sampling.py` with batched generation
  using past_key_values. Keep: temperature 1, softmax over raw logits, no top-k/top-p,
  no repetition penalty, no forced EOS, global budget H - n, stop at the first EOS.
- Use one torch.Generator per row, seeded from the job's semantic seed, so that the
  result of a job does not depend on batch composition and resume stays exact.
- Teacher-forced rescoring stays authoritative; generation-time scores are never used.
- Tests: (a) batched and unbatched sampling give identical tokens for identical per-row
  seeds; (b) the controlled-categorical frequency test still passes; (c) resume test still
  passes; (d) a throughput micro-benchmark is logged (not asserted).

## Step 2: decisions to fix before any returns are observed
Append to DECISIONS.md under "Pre-registration (NARVAL pilot)":
- Collection is shared across the three policy-training seeds. Training seeds vary only
  policy init order, pair sampling and minibatch RNG. Remove `seed` from collection and
  head cache keys; keep it for policy stages. Add a test that seeds 0/1/2 reuse the same
  collection shards.
- `correction_replicates = 1` in the new protocol configs (the spec does not require
  replicates; audits remain in `crpd_audit`).
- State subsampling: new config `states_per_prompt` (int or null = all). States are chosen
  uniformly without replacement from the causal schedule positions of the rho prefix
  trajectory, using a dedicated RNG stream `state-select/<prompt_id>`, before any candidate
  or correction is generated. The prompt state (n = 0) is always included. Pilot value: 4.
  Hierarchical weights are unchanged (1/N_prompts * 1/N_states * 1/C(K_s,2)) over the
  selected states. Tests: selection depends only on prompt id and seed; weights still sum
  to 1; no state selection reads any return or score.

## Step 3: displacement diagnostic stage (run before training any student)
Add `python -m chunkratio diagnose-displacement --config ... --prompts N --output ...`:
- From the prompt state and from each selected prefix state, draw R complete rho
  continuations (collection sampling), score them under T+ and T0, and report per state:
  mean, variance, min, max and range of d_s(z) = sum(log T+ - log T0); suffix length;
  the empirical Jensen gap log mean exp(d) - mean d with a bootstrap interval and a
  rare-tail warning when one sample carries > 50 % of the exponential weight.
- For sibling endpoints at the same state, report the spread (max - min) of the estimated
  gaps, which is the omega_s of Method eq. (15a).
- Write JSON plus a short markdown summary. Label it a diagnostic, not a result.
- Test it on the synthetic tree, where the exact phi, mu_rho and gap are known: the
  estimates must converge to the exact values as R grows.

## Step 4: NARVAL-comparable protocol
Create `config/crpd/narval/` with one protocol per pair:
`qwen3_1p7b_to_0p6b.json`, `qwen3_8b_to_1p7b.json`, `llama31_8b_to_llama32_1b.json`.
- SFT data: Deita-10K-V0 for teacher and student (reuse `prepare-data`).
- Preference data: UltraFeedback Binarized (`HuggingFaceH4/ultrafeedback_binarized`,
  `train_prefs` / `test_prefs`). Extend `prepare-data` to read it. Splits: `train` from
  train_prefs; `select` and `head_val` are held out of train_prefs by prompt id with a
  recorded hash; `test` is never used for selection.
- T0 = teacher after Deita SFT; T+ = DPO of T0 on UltraFeedback train split;
  rho = student after the same SFT. For qwen3_1p7b_to_0p6b, T0/T+ must be the SFT/DPO
  Qwen3-1.7B *student* checkpoints of qwen3_8b_to_1p7b, as in NARVAL Table 1; encode this
  as a provenance link and check it in validate-config.
- Base checkpoint ids (Base vs post-trained), DPO beta/lr/epochs/max length: `null`
  until filled from the NARVAL repository.
- Qwen3 chat template: thinking disabled and fixed in config; the end-of-turn token
  `<|im_end|>` must be the EOS used by every role. Extend preflight to check the four
  models' vocab mapping, special tokens, EOS id and output dimension (padded vocab sizes
  must map identically; fail closed otherwise). Add a tiny randomly initialised Qwen3-style
  config (transformers Qwen3 classes, no download) to the HF runtime tests.
- Variants to run: `crpd_main`, `crpd_raw`, `crpd_anchor` (anchor uses UltraFeedback
  train pairs; lambda stays a declared config value). Reference rows: DPO teacher,
  SFT teacher, student SFT, student DPO on UltraFeedback.

## Step 5: NARVAL-comparable evaluation and tables
- Open LLM: six tasks (HellaSwag, ARC, MMLU, TruthfulQA, Winogrande, GSM8K) with per-task
  num_fewshot, metric name (acc / acc_norm / mc2 / exact_match variant) and lm-eval
  version all as config fields, `null` by default and fail closed.
- MT-Bench: report the 8 categories plus average, with the judge identity recorded.
- HH-RLHF pairwise: AlpacaEval-style LC win rate and win rate against a fixed reference
  output file (GPT-4o-mini in NARVAL), judge configurable (GPT-5.4-mini in NARVAL), on a
  pinned HH-RLHF test subset with recorded ids. Plan template only; do not call APIs.
- `report --layout narval`: reproduce the column layout of NARVAL Table 1 and Table 2.
  Accept rows from a `reported.jsonl` with `source="NARVAL Table 1"` and `reported=true`.
  Render them visually separated from measured rows and never mix them into significance
  tests. Add a disclosure line that crpd_main uses prompts only.

## Step 6: verification
- Extend `script/mutation_check.py` with: shared collection across seeds broken,
  state selection reading a score, per-row generator removed, EOS taken from the wrong
  token. Every mutation must be killed.
- Smoke test: add a `--protocol narval-tiny` path using the tiny Qwen3-style models
  through every new stage (prepare-data on a 10-row synthetic UltraFeedback-format file,
  diagnose-displacement, the three variants, report --layout narval with one fake reported
  row marked synthetic).
- Run `python -m pytest -q`, both smoke scripts and the mutation check until all pass.

## Step 7: report
Update REPORT.md, PLAN.md and DECISIONS.md. Add the exact command sequence for the
Qwen3-1.7B -> 0.6B pilot: verify the SFT/DPO rows against NARVAL Table 1, run
diagnose-displacement on 100 prompts, then crpd_main / crpd_raw / crpd_anchor with
one seed. List every `null` input that the user must fill. Final commit:
`narval: protocol complete`.