# Hướng tốt nhất (ghim cố định)

Cập nhật ngày 06/11/2026 (DECISIONS.md mục 89). Chỉ thay khi một run mới tốt hơn trên **tập test**, đồng thời tỉ lệ
EOS và độ dài gần với rho. Điểm tập select không đủ để thay.
Chi tiết: [best_version_2026-11-05.md](best_version_2026-11-05.md) (γ = 0.1 và quét γ); bản cũ:
[best_version_2026-11-04.md](best_version_2026-11-04.md).

| Vai trò | Lựa chọn |
|---|---|
| Model dùng thật | **rho**: `hoangducanh1865/qwen3-0.6b-deita-sft-student` @ `fc0f225c` (trùng từng byte với `vukien2301/qwen3-0.6b-deita-sft-student`) |
| Run CRPD tốt nhất (nghiên cứu) | **`crpd_raw-alpha0p03-gamma0p05-g01`** (γ = 0.05): Modal `nuocb0739`, volume `crpd-qwen3-pilot-hda-x10` |
| Cấu hình code | **x10 crpd_raw turbo, γ = 0.05**: `CRPD_TURBO=1 CRPD_GAMMAS=0.05 CRPD_CACHE_TAG=shared`, 150 bước × 128 cặp, α 0.03, commit `23838ac`+, nhánh `crpd-hda-review` |

Quy tắc chọn: trong các run có EOS heldout ≥ 90%, chọn run có RM select cao nhất.

| Chỉ số | rho | **γ = 0.05** | γ = 0.1 | γ = 0.03 | NARVAL `riskkd` |
|---|---:|---:|---:|---:|---:|
| RM select so với rho | — | **+1.00 [0.34, 1.65]** | +1.12 [0.46, 1.78] | +0.66 [0.09, 1.23] | — |
| RM test so với rho | — | **+0.54 [−0.20, 1.28]**, 43/20 (p = 0.005) | +0.77 [0.02, 1.51], 40/24 | +0.16, 30/33 | — |
| RM test khi cùng độ dài / khi cả hai có EOS | — | **+1.23 / +0.93** (42/16, p = 0.001) | chưa đo | +0.64 / +0.70 | — |
| EOS heldout | 98.4% | **92.2%** | 82.8% | 89.1% | — |
| Độ dài trung bình | 247 | **370** | 526 | 388 | — |
| Lặp 4-gram | 15.0% | 21.3% | 26.7% | 23.7% | — |
| Winogrande (hf fp32) | 57.85 | 57.85 | 58.01 (58.33 lần đầu) | 58.09 | 58.72 |

γ = 0.05 là run CRPD đầu tiên vừa hơn rho trên test (đa số prompt, p = 0.005) vừa qua ngưỡng EOS. Mức hơn không phải
do câu trả lời dài: RM giảm theo độ dài, và khi cùng độ dài thì mức hơn còn lớn hơn. Chưa hơn NARVAL trên Winogrande
(chênh lệch nằm trong nhiễu). Còn thiếu trước khi chạy full: độ dài vẫn gấp 1.5 lần rho, độ lặp 21%; GSM8K và
TruthfulQA so với NARVAL `riskkd`; pilot 2,000 prompt.
