# ChunkRatio-PD v1.1 / TBPO fork

Implement the revision-2 specification without changing inherited TBPO-Q/A results.
Sources: `ChunkRatio_PD_v1_1_Coding_Spec_rev2.md`, then
`reference/ChunkRatio_PD_Method_1_1.md`, then `reference/ChunkRatio_Math_Checks.*`.
Explicit user instructions take precedence. Record conflicts in DECISIONS.md.

## Map and commands

- `chunkratio/math.py`: ratios, suffix returns, routing, soft pair loss, weights.
- `chunkratio/scoring.py`, `sampling.py`, `collection.py`: causal scoring, raw
  categorical collection, prompt splits and resumable artifacts.
- `chunkratio/value_head.py`, `targets.py`, `training.py`: detached reference
  features, trajectory-weighted scalar head, frozen targets, student optimization.
- `chunkratio/cli.py`, `__main__.py`: staged command interface.
- `chunkratio/evaluation.py`, `experiment.py`: measured diagnostics and benchmark adapters.
- `tests/`: own tests only; vendored FastChat tests are excluded by pytest.ini.
- Inherited: `train.py` / `trainers.py` (custom Hydra + PyTorch/FSDP),
  `preference_datasets.py`, `utils.py`, `baseline_head.py`, `loss/`, `config/`.
- Evaluation reuse: `script/eval/general/` (lm-eval),
  `mtbench/FastChat/fastchat/llm_judge/` (MT-Bench single scores),
  `winrate_eval/` (AlpacaEval), `diversity_metrics/` (supplementary).

Commands from repo root, using the active Python interpreter:

```text
python reference/ChunkRatio_Math_Checks.py output/reference-checks.json
python -m pytest -q
python -m chunkratio --help
python -m chunkratio validate-config --config config/crpd/smoke.json
python -m chunkratio smoke --output output/smoke
```

Pipeline stages accept `--config FILE`: validate-config, collect-prefixes,
collect-candidates, collect-correction, fit-head, build-targets, train-policy,
evaluate. `toy-check` runs the independent reference; `learned-tree` runs the
finite-data experiment. See REPORT.md for executed results versus unavailable checks.

## Hard constraints

- New implementation under `chunkratio/`; shared changes additive and opt-in.
  `tbpo-baseline` identifies inherited code. Never relabel ChunkRPO as CRPD.
- Frozen T+, actual DPO reference T0, and exact student initialization rho;
  frozen scoring under no_grad, eval mode; student eval mode with gradients.
- Identical token mapping, special tokens, chat template, output vocabulary.
- Sum valid chunk token log probabilities in FP32 or FP64; exclude prompts,
  padding and post-EOS, include sampled EOS exactly once, no artificial EOS.
- Global response budget H=512; L=32, K=4 (2 rho + 2 T+); no length normalization.
- Target delta_T + endpoint mu_hat; terminal value zero. One detached-rho scalar
  head; reverse cumulative teacher log-ratio labels from rho rollouts; head
  objective sum per trajectory / fixed H, averaged over trajectories.
- Exact identities only: T+=T0 => no optimizer step; rho=T0 => zero residual;
  otherwise reference_mean. Never use score tolerance or family names as identity.
- Soft BCE: p=sigmoid(alpha*(t_c-t_d)); m=alpha*(a_c-a_d), alpha>0 both sides.
  Frozen labels; student gradients intact; prompt/state/pair hierarchical measure.
- Primary: no anchor, KL penalty, variance head, clipping, label smoothing,
  online refresh or weight decay. Only crpd_anchor consumes human pairs.
- Raw categorical collection separate from benchmark-standard evaluation.
- Split duplicate prompt groups before generation; no test-based selection.
- Atomic artifacts, provenance, independent RNG streams and resumable state.
- No absolute paths. Windows and Linux launchers. No full-scale training here.
- Never push, delete outside the repo, or rewrite git history for this task.
- Report actual results; unavailable dependencies and measurements are Not done.
