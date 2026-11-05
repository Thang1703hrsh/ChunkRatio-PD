# Phiên bản CRPD tốt nhất hiện tại: γ = 0.1 (cập nhật 05/11/2026)

Ghim theo yêu cầu của người dùng ngày 05/11/2026. Bản này thay `crpd_raw-alpha0p03-noend` ở vai trò
**run CRPD tốt nhất để nghiên cứu**. Model nên dùng thật vẫn là **rho** cho đến khi γ = 0.1 qua được
kiểm tra heldout (mục 4).

## 1. Run

| | |
|---|---|
| Tên run | `crpd_raw-alpha0p03-gamma0p1-gamma` |
| Biến thể | `crpd_raw`, `teacher_gap_scale` γ = 0.1 (DECISIONS.md mục 86) |
| Bộ model | rho, T0, T+ của hda, trùng từng byte với các checkpoint vukien2301 mà nhóm NARVAL dùng |
| Cấu hình | x10: 640 prompt train, 2 state/prompt, K = 4 (2 rho + 2 T+), L = 32, H = 512 |
| Train | α = 0.03, lr 1e-6, 150 bước × 128 cặp, seed 0, có chấm token kết thúc |
| Dữ liệu | 1,513 state, 7,108 cặp, dùng chung collection với γ = 1 và γ = 0.3 |
| Cấu hình đã resolve | `output/hda-x10-gamma/gamma0p1/resolved_config.json` |
| Chẩn đoán train | `output/hda-x10-gamma/gamma0p1/diagnostics.json` |

## 2. Nơi lưu student (định dạng HF, fp32, 2.38 GB)

| Tài khoản Modal | Volume | Đường dẫn |
|---|---|---|
| `trinhson3392` (bản gốc; tài khoản đã vượt giới hạn chi tiêu, chỉ đọc được) | `crpd-qwen3-pilot-hda-x10` | `output/narval/qwen3_1p7b_to_0p6b_hda/pilot/crpd_raw-alpha0p03-gamma0p1-gamma/seed0/student` |
| `nuocb0739` (bản sao) | `crpd-quick-eval` | `crpd-students/gamma0p1-gamma` |

## 3. Kết quả

**Tập select** (64 prompt, RM; dùng để chọn γ):

| γ | RM so với rho (−4.18) | Thắng/thua |
|---|---:|---:|
| 1 | +0.89 [0.09, 1.69] | 38/26 |
| 0.3 | +0.75 [0.09, 1.40] | 35/29 |
| **0.1** | **+1.18 [0.49, 1.86]** | **46/18** |

**Winogrande** (5-shot, hf fp32, lm-eval 0.4.9.2, 1,267 câu; thước đo báo cáo, không dùng để chọn):

| Model | Winogrande | So với NARVAL | Kiểm định dấu với NARVAL | Kiểm định dấu với rho |
|---|---:|---:|---|---|
| `pfw-tail` (vukien2301) | 59.12 | +0.39 | 36/31, p = 0.63 | 66/50, p = 0.16 |
| NARVAL `riskkd` (vukien2301) | 58.72 | — | — | 56/45, p = 0.32 |
| **CRPD γ = 0.1** | **58.33** | −0.39 | 44/49, p = 0.68 | 18/12, p = 0.36 |
| rho (SFT) | 57.85 | −0.87 | 45/56, p = 0.32 | — |
| CRPD γ = 0.3 | 57.62 | −1.10 | 31/45, p = 0.14 | 26/29, p = 0.79 |
| CRPD noend | 57.14 | −1.58 | 29/49, p = 0.03 | 47/56, p = 0.43 |
| CRPD γ = 1 | 56.35 | −2.37 | 29/59, p = 0.002 | 37/56, p = 0.06 |

Kết quả từng câu: volume `crpd-quick-eval` (`nuocb0739`), thư mục `winogrande-crpd-latest/`.
Bản tóm tắt: `output/hda-x10-gamma/winogrande_summary.md`.

Diễn giải:
- γ = 0.1 ngang NARVAL trên Winogrande (khác biệt không có ý nghĩa thống kê) với khoảng 1% số prompt.
  NARVAL `riskkd` train 3,605 bước × 16 cặp (khoảng 57.7k cặp, 1 epoch).
- γ càng nhỏ càng tốt (56.35 → 57.62 → 58.33): ép student bằng đúng chênh lệch của teacher làm student trôi xa rho.
- Đo theo paper: bộ vukien2301 thấp hơn số trong NARVAL Table 1 khoảng 1.4 điểm ở student 0.6B, dù đổi backend,
  dtype hay phiên bản lm-eval. Vì vậy chỉ so tương đối, trên cùng một cách đo.

## 4. Chưa kiểm tra (bắt buộc trước khi dùng hoặc chạy full)

- **Heldout `pd_diagnostics`** (RM test, tỉ lệ EOS, độ dài, độ lặp). Winogrande không phát hiện lỗi không dừng;
  noend và γ = 1 từng có EOS 0–3% trên heldout dù select tốt. Điều kiện: EOS ≥ 90%, độ dài gần 247 token,
  độ lặp gần 15%, RM test không kém rho.
- GSM8K và TruthfulQA (hai task phân biệt các phương pháp rõ nhất trong paper).
- γ < 0.1 (0.05, 0.03): xu hướng chưa dừng; chọn theo select, không theo Winogrande.

## 5. Chạy lại

```bash
export CRPD_SCALE=x10 CRPD_TEACHERS=hda CRPD_VARIANTS=crpd_raw CRPD_ALPHAS=0.03 CRPD_GAMMAS=0.1 \
       CRPD_TURBO=1 CRPD_STEPS=150 CRPD_BATCH_SIZE=128 CRPD_OUTPUT_TAG=gamma
modal deploy script/modal/qwen3_pilot.py
python script/modal/qwen3_pilot.py start
```

Eval Winogrande: `script/modal/eval_vukien/` (`winogrande_crpd_students.py` cho các student CRPD,
`winogrande_vukien.py` cho checkpoint vukien2301, `winogrande_harness_matrix.py` cho phép thử backend/dtype/phiên bản).
