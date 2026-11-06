# Running CRPD on a 2 × B200 server

A short guide to training the three NARVAL Table 1 pairs and pushing the students to Hugging Face. For what each
stage does and how GPU memory is planned, see [docs/SERVER.md](../../docs/SERVER.md).

| `--pair` | Teacher → Student |
|---|---|
| `qwen3_1p7b_to_0p6b` | Qwen3-1.7B → Qwen3-0.6B |
| `qwen3_8b_to_1p7b` | Qwen3-8B → Qwen3-1.7B |
| `llama31_8b_to_llama32_1b` | Llama-3.1-8B → Llama-3.2-1B |

Run every command from the repository root, on Linux.

## 1. Setup (once)

```bash
git clone https://github.com/Thang1703hrsh/ChunkRatio-PD.git
cd ChunkRatio-PD
git checkout crpd-hda-review      # run from a git clone: provenance records the commit

# Training environment (B200 needs a CUDA 12.8 build of PyTorch)
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-crpd.txt huggingface_hub safetensors numpy nvidia-ml-py

# Benchmark environment, kept separate (lm-eval needs transformers 4.56)
python -m venv ../venv-eval
../venv-eval/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
../venv-eval/bin/pip install -r requirements-eval.txt

# Hugging Face login with a token that has write access (needed for the push)
huggingface-cli login
```

For the Llama pair, accept the licence of `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct` on the
Hub first. The weight check downloads them.

Estimated resources (not yet measured on the server):

- At least 256 GB of host RAM. Two 8B collections can run at once, each loading two FP32 teachers (about 64 GB),
  and the weight check needs about 32 GB.
- About 500 GB of free disk for the Hugging Face cache and the collection data.

## 2. Smoke test (about 15–30 minutes per pair)

```bash
script/server/run_pair.sh --pair qwen3_1p7b_to_0p6b --scale smoke --gpus 0,1 --gammas 0.05 --skip-verify
```

This uses 32 prompts, 10 steps and a 16-prompt held-out check. It only shows that the pipeline runs end to end, so
ignore its scores. Start with the 0.6B pair, then run the two 8B pairs the same way.

## 3. Full run: all three pairs at once, pushed to Hugging Face

```bash
python script/server/run_all.py --gpus 0,1 --scale full --pair-sources within \
    --pair llama31_8b_to_llama32_1b:0.05 \
    --pair qwen3_8b_to_1p7b:0.05 \
    --pair qwen3_1p7b_to_0p6b:0.1 \
    --eval-python ../venv-eval/bin/python \
    --hub-namespace <your-user-or-org>
```

`run_all.py` schedules every pair's jobs over the two GPUs so that neither one sits idle:

- Each collection is split into 6 shards that run on whichever GPU is free.
- While one pair trains on one GPU, the other GPU collects for the next pair.
- The rho and NARVAL benchmarks start right after a pair is prepared.
- Downloads and the weight check run on the CPU, alongside the GPU work.

For each pair the run does the following:

1. Prepare the data and check the teacher weights.
2. Collect.
3. Train for one epoch (`--passes 1`), with 3 checkpoint selections on the 64 select prompts.
4. Run the held-out check (64 test prompts: RM against rho, EOS rate, length).
5. Run Winogrande 5-shot and TruthfulQA-MC2 for the student, rho and NARVAL.
6. **Push the student to Hugging Face.**

Leave out `--hub-namespace` if you do not want to push.

The order of the `--pair` flags sets their priority. The order above was the fastest in a simulation of the
schedule: about 36 h for all three pairs, against about 44 h one pair after another. Both figures come from
estimated stage times scaled from A100 runs, not from B200 measurements.

If the run stops, rerun the same command. Finished jobs are marked in `output/server/<pair>/full/schedule/` and are
skipped, and every stage resumes from what it saved. A failed job stops only the later jobs of its own pair.

Results for each pair:

- Training and held-out: `output/server/<pair>/full/runs/<run>/seed0/` (`diagnostics.json`, `heldout.json`)
- Benchmarks: `output/server/<pair>/full/benchmarks/summary.md`
- Logs: `output/server/<pair>/full/logs/`

### Why these γ values

There was no γ sweep for the 8B teachers, so two of the three values below are predictions.

| Pair | γ | Reason |
|---|---:|---|
| Qwen3-1.7B → 0.6B | **0.1** | Measured. With `--pair-sources within`, γ 0.1 was the best held-out result so far: RM +0.64 against rho (95% CI [0.13, 1.18], 41 wins / 21 losses), EOS rate 0.95 (rho 0.98), length 333 tokens (rho 247). |
| Qwen3-8B → 1.7B | **0.05** | Predicted. On the 0.6B pair, larger γ moved the student further from rho (longer answers, lower EOS rate, worse benchmarks at γ 1). An 8B teacher pair is likely to give larger teacher gaps, so the more conservative value is the safer bet. |
| Llama-3.1-8B → 3.2-1B | **0.05** | Predicted, for the same reason. The Llama pair also uses a different tokenizer and chat template from the measured pair. |

γ scales the teacher gap in the targets (`teacher_gap_scale`), and α stays at 0.03. To try a second value for a pair
anyway, list both, for example `--pair qwen3_8b_to_1p7b:0.05,0.1`. Both values then train on the same collection
and both are pushed. Decide in advance which one is the result, and report the other as an ablation.

## 4. What gets pushed to Hugging Face

Each pair pushes one student to `<namespace>/crpd-<pair>-full-<run>`. That gives three checkpoints for the three
pairs. rho and NARVAL are already on the Hub, so they are not pushed.

The push (`script/server/push_hub.py`):

- Creates a **private** repo. Add `--hub-public` to `run_all.py` for a public one.
- Makes one commit with the weights, the tokenizer, a model card (sources, how the checkpoint was chosen, held-out
  and benchmark results) and `crpd_provenance.json`.
- Checks that the weights' sha256 on the Hub match the local files, and records the repo and commit in
  `runs/<run>/seed0/hub.json`.
- Refuses to push when checkpoint selection kept rho (the student would be rho itself) or when `heldout.json` is
  missing.
- Does nothing on a rerun if the weights have not changed.

To evaluate a pushed model again later, for example from another machine:

```bash
../venv-eval/bin/python script/server/benchmarks.py --pair qwen3_8b_to_1p7b --gpus 0,1 \
    --models crpd=<namespace>/<repo>@<commit> --output output/server/hub-eval
```

`benchmarks.py` skips any model that already has results in its `--output` folder, so give it a new folder to
evaluate from scratch.
