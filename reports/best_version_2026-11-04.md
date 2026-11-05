# Phiên bản tốt nhất hiện tại (cập nhật 04/11/2026)

## 1. Model nên dùng ngay bây giờ: rho (chưa train CRPD)

| | |
|---|---|
| Checkpoint | `hoangducanh1865/qwen3-0.6b-deita-sft-student` |
| Revision | `fc0f225cb63e5db6e84a7e5705c997891ec85d65` |
| Token kết thúc lượt | `<|im_end|>` (chat template ChatML, `enable_thinking=false`) |

Lý do: trên 64 prompt test (không dùng để chọn checkpoint), chưa student CRPD nào tốt hơn rho.
Rho tự kết thúc câu trả lời 98.4%, dài trung bình 247 token, RM test −3.02.

## 2. Student CRPD tốt nhất hiện có (chỉ để nghiên cứu, chưa nên dùng)

| | |
|---|---|
| Tên run | `crpd_raw-alpha0p03-noend` |
| Tài khoản Modal | `trinhson3392` |
| Volume | `crpd-qwen3-pilot-hda-x10` |
| Student (HF format) | `output/narval/qwen3_1p7b_to_0p6b_hda/pilot/crpd_raw-alpha0p03-noend/seed0/student` |
| Cấu hình | crpd_raw, K=4, 2 state/prompt, alpha 0.03, 150 bước × 128 cặp, lr 1e-6 cosine, token kết thúc không chấm |
| Kết quả, bản sao local | `output/hda-x10-noend/` (`diagnostics.json`, `pd_diagnostics.json`) |

| Chỉ số | rho | noend | turbo (cũ) |
|---|---:|---:|---:|
| RM tập select (so với rho) | — | +1.09 [+0.32, +1.85] | +0.86 |
| **RM tập test (so với rho)** | — | **−0.48** [−1.24, +0.23] | −0.64 |
| Tỉ lệ tự kết thúc (EOS) | 98.4% | 3.1% | 0% |
| Độ dài trung bình (token) | 247 | 1,011 | 1,024 |
| Lặp 4-gram | 15% | 35% | 34% |

**Lỗi đã biết:** student gần như không tự kết thúc và lặp nhiều. Nguyên nhân chính là student bị ép
sao chép toàn bộ độ chênh log-ratio của teacher (khoảng 47 nats mỗi chunk), nên bị kéo quá xa khỏi rho.
Ngoài ra T+ của bộ hda kết thúc bằng `<|endoftext|>` và viết dài gấp đôi T0.
Chi tiết: DECISIONS.md mục 84.

## 3. Cấu hình code tốt nhất để chạy CRPD (nhanh, đã kiểm chứng)

Code: nhánh `crpd-hda-review`. Thay đổi **chưa commit**; 142 test pass.

Train x10 crpd_raw với mọi tối ưu tốc độ (khoảng 50–55 phút trên A100-80GB, khoảng $2.5):

```bash
export CRPD_SCALE=x10 CRPD_TEACHERS=hda CRPD_ALPHAS=0.03 CRPD_VARIANTS=crpd_raw \
       CRPD_TURBO=1 CRPD_STEPS=150 CRPD_BATCH_SIZE=128 CRPD_OUTPUT_TAG=<ten-run>
modal deploy script/modal/qwen3_pilot.py
python script/modal/qwen3_pilot.py start
```

`CRPD_TURBO=1` bật:
- lấy mẫu trên GPU;
- nhóm sinh theo ngân sách token;
- chấm điểm dùng chung prefix;
- microbatch theo ngân sách token 8,192;
- chạy toàn bộ pipeline trong một process.

Các tối ưu này giữ nguyên kết quả, chỉ lệch ở mức làm tròn FP32.
Thêm `CRPD_NO_END_TOKEN=1` nếu muốn không chấm token kết thúc (đã thử, không sửa được lỗi).

Đánh giá sau khi train. **Luôn chạy bước này**, vì điểm trên tập select không đủ tin cậy:

```bash
CRPD_DIAG_TAG=<ten-run> modal deploy script/modal/pd_diagnostics.py
CRPD_DIAG_TAG=<ten-run> python script/modal/pd_diagnostics.py
```

Kết quả ở `output/pd_diagnostics_<ten-run>.json` trên volume, gồm:
- RM trên tập test;
- tỉ lệ EOS, độ dài, mức lặp;
- độ lệch so với rho;
- phân tích theo loại cặp.

## 4. Bước tiếp theo đề xuất

Thử giảm mức sao chép (γ < 1, ví dụ 0.1 và 0.3): student chỉ cần đạt γ × độ chênh của teacher.
Đây là thay đổi phương pháp, cần ghi vào DECISIONS.md. Chi phí khoảng $3.5 mỗi lần thử.
Nếu không hiệu quả, đổi sang cặp teacher có cùng token kết thúc lượt.
