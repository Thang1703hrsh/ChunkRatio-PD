# Chạy CRPD trên server 2 × B200

Hướng dẫn ngắn để chạy 3 cặp model của NARVAL Table 1 bằng `script/server/run_pair.py`.
Chi tiết đầy đủ (từng stage làm gì, memory plan) xem [docs/SERVER.md](../../docs/SERVER.md).

| `--pair` | Teacher → Student |
|---|---|
| `qwen3_1p7b_to_0p6b` | Qwen3-1.7B → Qwen3-0.6B |
| `qwen3_8b_to_1p7b` | Qwen3-8B → Qwen3-1.7B |
| `llama31_8b_to_llama32_1b` | Llama-3.1-8B → Llama-3.2-1B |

Mọi lệnh chạy từ thư mục gốc của repo, trên Linux.

## 1. Cài đặt (một lần)

```bash
git clone <repo> && cd ChunkRatio-PD          # cần bản git clone (provenance ghi commit)

# Môi trường train (B200 cần PyTorch bản CUDA 12.8)
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-crpd.txt huggingface_hub safetensors numpy nvidia-ml-py

# Môi trường benchmark, tách riêng (lm-eval cần transformers 4.56)
python -m venv ../venv-eval
../venv-eval/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
../venv-eval/bin/pip install -r requirements-eval.txt

# Đăng nhập Hugging Face (token có quyền write nếu muốn push)
huggingface-cli login
```

Cặp Llama: vào Hugging Face chấp nhận licence của `meta-llama/Llama-3.1-8B` và `meta-llama/Llama-3.1-8B-Instruct`
trước, nếu không stage `verify` sẽ dừng.

Ước lượng tài nguyên (chưa đo trên server): RAM máy khoảng 40 GB cho mỗi tiến trình train (tối đa 4 tiến trình
cùng lúc, tức khoảng 160 GB), ổ đĩa trống khoảng 500 GB cho HF cache và dữ liệu collection.

## 2. Kiểm tra kế hoạch (không tốn GPU)

```bash
python script/server/run_pair.py --pair qwen3_1p7b_to_0p6b --scale x10 --gpus 0,1 --gammas 0.1,0.05 --dry-run
```

In ra memory plan và mọi lệnh sẽ chạy, không chạy gì.

## 3. Smoke test (khoảng 15–30 phút mỗi cặp)

```bash
script/server/run_pair.sh --pair qwen3_1p7b_to_0p6b --scale smoke --gpus 0,1 --gammas 0.05 --skip-verify
```

32 prompt, 10 step, held-out 16 prompt. Chỉ để chắc chắn pipeline chạy thông, kết quả không có ý nghĩa.
Nên chạy cặp 0.6B trước, rồi tới hai cặp 8B.

Đo bộ nhớ GPU ở kích thước batch thật (tuỳ chọn):

```bash
python script/server/memory_probe.py --pair qwen3_8b_to_1p7b \
    --prompts output/server/qwen3_8b_to_1p7b/data/rows2048/ultrafeedback_train.jsonl --output probe_8b.json
```

## 4. Sweep x10 để chọn γ (khoảng 15–45 phút mỗi cặp)

```bash
script/server/run_pair.sh --pair qwen3_8b_to_1p7b --scale x10 --gpus 0,1 \
    --pair-sources within --gammas 0.1,0.05,0.03
```

Chọn γ **trước khi** chạy full, theo đúng luật này:

1. Bỏ các run có EOS held-out dưới 90%.
2. Trong các run còn lại, lấy run có select RM cao nhất (`diagnostics.json → policy.selection_score`).

Dòng cuối của log in sẵn RM held-out so với rho, EOS và độ dài của từng run. Kết quả nằm ở
`output/server/<pair>/x10/runs/<run>/seed0/`. Không chọn γ theo Winogrande hay TruthfulQA, vì đó là tập test.

## 5. Chạy full với γ đã chọn

```bash
script/server/run_pair.sh --pair qwen3_8b_to_1p7b --scale full --gpus 0,1 \
    --pair-sources within --gammas 0.1 \
    --stages prepare,verify,collect,train,heldout,benchmarks,push \
    --eval-python ../venv-eval/bin/python --hub-namespace <user-hoac-org>
```

- 1 epoch qua toàn bộ pair (`--passes 1`), chọn checkpoint 3 lần trong lúc train (`--selections 3`).
- `benchmarks`: Winogrande 5-shot và TruthfulQA-MC2 cho student, rho và NARVAL, có kiểm định cặp.
  Kết quả ở `output/server/<pair>/full/benchmarks/summary.md`.
- `push`: đẩy student lên `<namespace>/crpd-<pair>-full-<run>`, xem mục 6.

Bị ngắt giữa chừng thì chạy lại **đúng lệnh cũ**, mọi stage đều tiếp tục từ chỗ đã lưu.

Thời gian ước lượng trên 2 × B200 (quy đổi từ A100, chưa đo): Qwen 0.6B khoảng 7–8 h, Llama khoảng 16–19 h,
Qwen 8B khoảng 21–24 h.

### Chạy cả 3 cặp cùng lúc, không để GPU trống (khuyên dùng)

`run_all.py` lập lịch các job của cả 3 cặp trên 2 GPU:
- Collection được chia thành 6 shard. Các shard của một cặp chạy trên mọi GPU đang rảnh.
- Trong lúc một cặp train trên GPU này, GPU kia collect cho cặp tiếp theo.
- Benchmark rho/NARVAL chạy ngay sau prepare.
- Prepare/verify (chỉ cần CPU) chạy song song với công việc trên GPU.

```bash
python script/server/run_all.py --gpus 0,1 --scale full --pair-sources within \
    --pair llama31_8b_to_llama32_1b:0.05 --pair qwen3_8b_to_1p7b:0.1 --pair qwen3_1p7b_to_0p6b:0.1 \
    --eval-python ../venv-eval/bin/python --hub-namespace <user-hoac-org>
```

- Mỗi `--pair` có dạng `pair:γ[,γ...][:pair_sources]`. Thay γ bằng giá trị bạn chọn ở mục 4.
- Thứ tự các `--pair` là độ ưu tiên. Thứ tự trên (Llama, Qwen 8B, Qwen 0.6B) cho thời gian ngắn nhất trong mô phỏng.
- Thêm `--dry-run` để xem danh sách job và phụ thuộc mà không chạy gì.
- Bị ngắt thì chạy lại đúng lệnh: job đã xong được đánh dấu trong `output/server/<pair>/full/schedule/` và sẽ bị
  bỏ qua. Thêm `--fresh` để bỏ qua các dấu này.
- Một job lỗi chỉ dừng các bước sau của cặp đó. Các cặp khác vẫn chạy tiếp.

Mô phỏng với thời lượng ước tính (chưa đo trên B200): khoảng 36 h cho cả 3 cặp, so với khoảng 44 h nếu chạy
từng cặp một bằng `run_pair.sh`. Giới hạn lý thuyết khi 2 GPU bận 100% là khoảng 33,5 h.

Cần RAM máy nhiều hơn: có lúc 2 collection 8B chạy cùng lúc (mỗi cái nạp 2 teacher FP32, khoảng 64 GB) cộng
với verify (khoảng 32 GB). Nên có từ 256 GB RAM trở lên.

## 6. Push checkpoint lên Hugging Face

**Nên push 3 checkpoint, mỗi cặp 1**: student của lần chạy full với γ đã chọn ở mục 4. rho và NARVAL đã có
sẵn trên Hub nên không cần push. Student x10/x30 chỉ train trên vài trăm tới vài nghìn prompt nên không đại
diện cho method; launcher từ chối push chúng trừ khi thêm `--hub-allow-pilot`.

Muốn có ablation thì khai báo trước và push thêm tối đa 1 run mỗi cặp (ví dụ cùng γ với `--pair-sources all`),
và báo cáo nó cả khi kém hơn.

Stage `push` (`script/server/push_hub.py`):

- Tạo repo **private** (thêm `--hub-public` nếu muốn public).
- Một commit gồm weights, tokenizer, model card (nguồn gốc, cách chọn checkpoint, held-out, benchmarks nếu đã
  chạy) và `crpd_provenance.json`.
- Kiểm tra sha256 của weights trên Hub khớp với file local, ghi `runs/<run>/seed0/hub.json` (repo, commit).
- Từ chối push nếu bước chọn checkpoint giữ lại rho (khi đó student chính là rho) hoặc chưa có `heldout.json`.
- Chạy lại không push lần nữa nếu weights không đổi. Thêm `--hub-update-card` để cập nhật model card.

Eval bản đã push:

```bash
# Đúng commit đã push (ghim theo hub.json), sau khi push:
script/server/run_pair.sh --pair qwen3_8b_to_1p7b --scale full --gpus 0,1 --pair-sources within --gammas 0.1 \
    --stages push,benchmarks --bench-from-hub --eval-python ../venv-eval/bin/python --hub-namespace <user-hoac-org>

# Hoặc bất kỳ model nào trên Hub, ra một thư mục riêng:
../venv-eval/bin/python script/server/benchmarks.py --pair qwen3_8b_to_1p7b --gpus 0,1 \
    --models crpd=<user>/<repo>@<commit> --output output/server/hub-eval
```

`benchmarks.py` bỏ qua các model đã có kết quả trong thư mục `--output`. Muốn eval lại từ đầu thì dùng một thư
mục `--output` mới.

## 7. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `No such file or directory: 'git'` hoặc không có `.git` | Chạy từ bản git clone, hoặc `export CRPD_CODE_REVISION=<commit>` |
| `no reference alternative could be downloaded` (Llama) | Chấp nhận licence Llama trên Hub, rồi `huggingface-cli login` |
| CUDA out of memory khi collect | Tự xử lý: launcher giảm một nửa batch rồi chạy tiếp |
| CUDA out of memory khi train 2 tiến trình/GPU | Tự xử lý: chạy lại riêng từng tiến trình; hoặc thêm `--trainings-per-gpu 1` |
| `benchmarks needs --eval-python` | Thêm `--eval-python ../venv-eval/bin/python` |
| `push needs --hub-namespace` | Thêm `--hub-namespace <user-hoac-org>` |

Log của từng bước: `output/server/<pair>/<scale>/logs/`.
