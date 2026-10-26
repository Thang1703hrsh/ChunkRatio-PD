# Implementation plan

Data flow: split canonical prompts -> fixed rho prefixes -> sibling generation
-> T+/T0/rho scoring -> correction suffix labels -> mu-head fit/validation/freeze
-> frozen endpoint targets -> hierarchical pairs -> student train/select/export
-> benchmark generation/judging and measured reporting.

## Section map

| Spec | Implementation |
|---|---|
| 0 revision | DECISIONS.md |
| 1 sources | reference/HASHES.txt, tests/test_reference.py |
| 2 invariants | math.py, scoring.py, training.py |
| 3 TBPO | train.py opt-in dispatch, tests/test_baselines.py |
| 4 equations/routing | math.py, targets.py |
| 5 tokens/horizon/decoding | sampling.py, scoring.py |
| 6 scores/features | scoring.py, value_head.py |
| 7 splits/collection/head | collection.py, value_head.py |
| 8 targets/training | targets.py, training.py |
| 9 variants | config.py, collection.py, targets.py |
| 10 artifacts/resume/CLI | artifacts.py, cli.py, training.py |
| 11 protocol/evaluation | config/crpd/, evaluation.py, experiment.py |
| 12 verification | tests/, learned_tree.py, smoke.py |
| 13 delivery | REPORT.md |
| 14 agent task | AGENTS.md, CLAUDE.md |
| 15 references | preserved source documents |

Paths above are relative to chunkratio/ unless qualified.

## Steps

- [x] 0. Read sources, baseline tag, instructions, plan and decisions.
- [x] 1. Pure tensor mathematics.
- [x] 2. Numeric, masks and frozen-gradient fixtures.
- [x] 3. Independent torch FP64 tree port and JSON comparisons.
- [x] 4. Causal collection, scoring and resumable artifacts.
- [x] 5. Scalar head and training stage.
- [x] 6. Trainer, routes, variants and additive TBPO dispatch.
- [x] 7. Evaluation, protocol and table reporting (external execution gaps in REPORT).
- [x] 8. Cross-platform tiny CPU smoke stages and variants (Windows + Git Bash).
- [x] 9. Execute tests/smoke, resolve the stable-wheel block using official CPU nightly.
- [ ] 10. Review spec, verification matrix, report and final commit.

Checked steps mean work delivered/reviewed; executed verification status and
unfulfilled requirements are stated individually in REPORT.md.
