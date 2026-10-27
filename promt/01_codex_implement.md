# Task for Codex: implement ChunkRatio (ChunkRatio-PD v1.1) in this TBPO fork

You are running unattended overnight. Nobody will answer questions or approve steps.

## Autonomy rules (read first, obey throughout)
- Work fully autonomously from start to finish. Never ask a question, never ask for
  confirmation, never end a turn with a proposal such as "should I continue?".
- Do not end your turn until step 10 below is complete and committed. If you run out of
  ideas on one item, log it in REPORT.md under "Not done" and move to the next item.
- When something is ambiguous, choose the option most consistent with the spec, record it
  in DECISIONS.md (decision, reason, alternatives considered), and keep going.
- If a command fails, read the error, fix the cause, and retry. If an install fails because
  of the network, use what is already installed and note it in REPORT.md.
- Python: use the interpreter that is active in this shell (the run script activates it).
  Install missing packages with `python -m pip install ...` into that environment only.
- Never run `git push`, never delete files outside this repository, never rewrite git history.
- Another agent (Claude Code) will review your work afterwards, so leave the repository in a
  state where `python -m pytest -q` runs and PLAN.md, DECISIONS.md and REPORT.md are accurate.

## Sources of truth — read ALL of them completely before writing code
1. `ChunkRatio_PD_v1_1_Coding_Spec_rev2.md` (repo root): the implementation spec. Read every
   section, including the verification table and §14.
2. `reference/ChunkRatio_PD_Method_1_1.md`: the method and its theory (notation, equations).
3. `reference/ChunkRatio_Math_Checks.py` and `reference/ChunkRatio_Math_Checks.json`: exact FP64
   finite-tree reference. Run `python reference/ChunkRatio_Math_Checks.py` first and confirm it
   prints `ALL CHECKS PASSED`. Your code must reproduce its identities and fixtures.
4. The existing TBPO code: `train.py`, `trainers.py`, `preference_datasets.py`, `utils.py`,
   `baseline_head.py`, `merge.py`, `loss/`, `config/`, `script/`, `mtbench/`, `winrate_eval/`,
   `diversity_metrics/`. Understand the training loop, the config system, the data pipeline,
   the TBPO-Q/TBPO-A losses and Q-head, and the evaluation scripts before designing anything.

Priority when sources conflict: Coding Spec > Method_1_1 > Math_Checks > existing TBPO
conventions. Log every conflict and its resolution in DECISIONS.md.

## Hard constraints
- Existing TBPO-Q/TBPO-A behaviour must remain unchanged in results. The git tag
  `tbpo-baseline` marks the untouched code. Put new code under `chunkratio/`, plus new config
  files and scripts. Edit shared files (`train.py`, `trainers.py`, `preference_datasets.py`,
  `config/`) only additively, for example a new registry entry or a branch selected by config.
- Follow the spec exactly for:
  - notation: `d_T`, `delta_T`, `a_theta`, `mu_rho`, `alpha`;
  - loss `L = E[softplus(m) - p_hat * m]`, with `p_hat = sigmoid(alpha * (t_s(c) - t_s(d)))`
    and `m = alpha * (a_theta(c) - a_theta(d))`, where `a_theta = log pi_theta - log rho`;
  - target `t_s(c) = delta_T(c|s) + mu_hat(sc)`, with `delta_T = log T+ - log T0` summed
    over the chunk tokens, and `mu_hat = 0` at terminal endpoints;
  - routing: `T+ == T0` gives no update; `rho == T0` gives `exact_zero_residual`;
    otherwise `reference_mean`;
  - mu-head labels: reverse cumulative sums of `log T+ - log T0` along rho rollouts,
    0 after EOS;
  - hierarchical weights `1/N_prompts * 1/N_states * 1/C(K_s, 2)`;
  - the ADPA/TVKD protocol, hyperparameters, all variants (`crpd_main`, `crpd_raw`,
    `crpd_mu_teacher`, `crpd_token`, `crpd_suffix`, `crpd_anchor`, `crpd_audit`), metrics
    and reference rows.
- T+, T0 and rho are frozen. Compute their log-probs under `torch.no_grad()`. Accumulate
  log-probs and log-ratios in float32 or higher, never in bf16.
- Masks: prompt tokens and padding never contribute; EOS is included in the response, and
  nothing after EOS contributes.
- Keep raw collection sampling separate from evaluation decoding, as the spec requires.
- No hard-coded absolute paths. Everything goes through config or CLI flags.
- Everything must run on Windows (PowerShell) and Linux. Use `pathlib`, and provide both
  `.ps1` and `.sh` launch scripts.

## Workflow (git commit after each step with a clear message, e.g. `crpd step 3: ...`)
0. Create `AGENTS.md` and an identical `CLAUDE.md` at the repo root, covering:
   project summary, directory map, how to run the tests and each pipeline stage, and the
   hard constraints above. Create `PLAN.md`, covering:
   - modules and file layout;
   - the data flow: collection -> teacher/reference scoring -> labels -> mu-head ->
     pair construction -> training -> evaluation;
   - a table mapping every spec section to the file(s) that implement it;
   - a checklist with `[ ]` items.

   Create `DECISIONS.md`.
1. `chunkratio/math.py`: pure tensor functions for sequence/chunk log-ratios, suffix returns,
   mu targets, pair target logits, the loss, routing and hierarchical weights.
2. `tests/` (pytest), reproducing every spec fixture:
   - target logit 0.24, `p_hat = 0.55971365`, `dLoss/dm = -0.05971365` at `m = 0`,
     and `loss = log 2` at `m = 0`;
   - suffix returns `[0.3, -0.2, 0.4] -> [0.5, 0.2, 0.4]`;
   - swap symmetry and invariance to a common offset;
   - weights summing to 1;
   - label error `|dp| <= alpha * eps / 2`;
   - mask tests (prompt, padding, after-EOS tokens contribute nothing);
   - a test that frozen-model log-probs carry no gradient.
3. `tests/test_tree_reference.py`: a torch FP64 port of the finite tree. It must reproduce the
   identities in `reference/ChunkRatio_Math_Checks.py` to `<= 1e-10` and match the JSON
   fixtures:
   - two representations of phi;
   - chunk decomposition and partition invariance;
   - the Q^rho identity;
   - forward and reverse Jensen-gap KL;
   - routing cases.
4. Data: causal chunking rules from the spec, sibling sampling from rho, scoring under
   T+/T0/rho, and disk caching with resumable shards.
5. mu-head: a scalar head on frozen rho features, plus its training script.
6. ChunkRatio trainer integrated into the TBPO training loop, with every variant selectable
   by config.
7. Evaluation and reporting scripts for all metrics and tables in the spec. Reuse `mtbench/`,
   `winrate_eval/` and `diversity_metrics/` where possible.
8. End-to-end smoke test (`script/smoke_test.ps1` and `script/smoke_test.sh`) on a tiny,
   randomly initialised causal LM with a few prompts and a few steps. It must run every stage
   and every variant on CPU, and must not download large models.
9. Run the full test suite and the smoke test, and fix everything until all pass.
   Do NOT launch full-scale LLaMA-3.1-8B -> LLaMA-3.2-1B training. Instead write ready-to-run
   scripts in `script/` with the exact spec hyperparameters and an estimate of GPU-hours.
10. Write `REPORT.md`, covering:
    - what was implemented;
    - test and smoke-test results (paste the summary lines);
    - how each item in the spec's verification table is satisfied (file + test name);
    - deviations, "Not done" items with reasons, and remaining risks;
    - the exact command sequence to reproduce the main results table.

    Tick the PLAN.md checklist, then make the final commit `crpd: implementation complete`.

## Quality bar
- Type hints throughout. Every function's docstring cites the spec section or equation it
  implements.
- Assertions on tensor shapes, masks and finiteness. Deterministic seeds.
- Before finishing, re-read the spec end to end and tick every item in PLAN.md. Anything not
  done goes into REPORT.md with the reason.
