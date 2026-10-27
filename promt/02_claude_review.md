# Task for Claude Code: strict review and completion of the ChunkRatio implementation

You are running unattended overnight. Nobody will answer questions or approve steps.
Never ask a question and never stop for confirmation. Finish every step below, then stop.

Codex has just implemented ChunkRatio (ChunkRatio-PD v1.1) in this repo. Read `PLAN.md`,
`DECISIONS.md`, `REPORT.md` and `git log --oneline` first. The tag `tbpo-baseline` marks the
original TBPO code. Codex may have stopped early, so treat its claims as unverified.

## Rules
- Python: use the interpreter that is active in this shell. Install missing packages with
  `python -m pip install ...` into that environment only.
- Commit each fix separately: `review: <what was wrong> -> <fix>`.
- Never run `git push`, never rewrite history, never delete files outside the repo.
- If you disagree with a decision in DECISIONS.md, follow the spec, change the code, and
  append your reasoning to DECISIONS.md under "Review".

## Steps
1. Re-read these completely:
   - `ChunkRatio_PD_v1_1_Coding_Spec_rev2.md`
   - `reference/ChunkRatio_PD_Method_1_1.md`
   - `reference/ChunkRatio_Math_Checks.py` and its JSON

   Run the reference script and confirm `ALL CHECKS PASSED`.
2. Check every formula in `chunkratio/` against the spec, line by line:
   - signs, and where `alpha` is placed: `p_hat = sigmoid(alpha * (t_s(c) - t_s(d)))`
     and `m = alpha * (a_theta(c) - a_theta(d))`;
   - `a_theta = log pi_theta - log rho`;
   - the target `t_s(c) = delta_T(c|s) + mu_hat(sc)`, with `mu_hat = 0` at terminal
     endpoints;
   - the loss `softplus(m) - p_hat * m`, with `p_hat` detached;
   - suffix-return labels: reverse cumsum of `log T+ - log T0`, 0 after EOS;
   - routing (no update, `exact_zero_residual`, `reference_mean`);
   - hierarchical weights `1/N_prompts * 1/N_states * 1/C(K_s, 2)`;
   - masks: prompt tokens, padding, EOS included, nothing after EOS;
   - off-by-one shifts between logits and labels;
   - `torch.no_grad()` on T+/T0/rho, and float32-or-higher log-prob sums;
   - chunk boundaries that are causal (they depend only on the prefix);
   - collection sampling separate from evaluation decoding;
   - hyperparameters, variants and metrics matching the spec tables.
3. Confirm that the TBPO-Q/TBPO-A code paths are unchanged in behaviour. Run
   `git diff tbpo-baseline -- train.py trainers.py preference_datasets.py loss/ config/`
   and check that every change is additive and gated by config.
4. Check that the tests are real (mutation check). For at least these mutations, apply the
   change, confirm some test fails, then revert:
   - flip the sign of `delta_T`;
   - drop `mu_hat`;
   - move `alpha` outside the sigmoid;
   - include prompt tokens in the mask;
   - remove `no_grad`.

   If a mutation survives, write the missing test and commit it.
5. Complete anything Codex left unfinished (PLAN.md checklist, "Not done" in REPORT.md,
   missing spec items).
6. Run `python -m pytest -q` and the smoke test (`script/smoke_test.ps1` on Windows).
   Fix every failure until both pass.
7. Write `REVIEW.md`, covering:
   - bugs found, each with file:line, severity, fix and commit hash;
   - mutation-check results;
   - the final test and smoke-test summary lines;
   - remaining doubts;
   - a clear verdict: READY or NOT READY for GPU runs, with the reasons.

   Update REPORT.md where it was wrong. Final commit: `review: complete`.
