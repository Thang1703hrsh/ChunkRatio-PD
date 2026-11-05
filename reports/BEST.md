# Hướng tốt nhất (ghim cố định)

Cập nhật ngày 05/11/2026 theo yêu cầu người dùng (bản trước ghim ngày 04/11/2026). Chỉ thay khi một run mới
tốt hơn trên **tập test**, đồng thời tỉ lệ EOS và độ dài gần với rho. Điểm tập select không đủ để thay.
Chi tiết đầy đủ: [best_version_2026-11-05.md](best_version_2026-11-05.md); bản cũ:
[best_version_2026-11-04.md](best_version_2026-11-04.md).

| Vai trò | Lựa chọn |
|---|---|
| Model dùng thật | **rho**: `hoangducanh1865/qwen3-0.6b-deita-sft-student` @ `fc0f225c` (trùng từng byte với `vukien2301/qwen3-0.6b-deita-sft-student`) |
| Run CRPD tốt nhất (nghiên cứu) | **`crpd_raw-alpha0p03-gamma0p1-gamma`** (γ = 0.1): Modal `trinhson3392`, volume `crpd-qwen3-pilot-hda-x10`; bản sao `nuocb0739`, volume `crpd-quick-eval`, `crpd-students/gamma0p1-gamma` |
| Cấu hình code | **x10 crpd_raw turbo, γ = 0.1**: `CRPD_TURBO=1 CRPD_GAMMAS=0.1`, 150 bước × 128 cặp, alpha 0.03, nhánh `crpd-hda-review` |

| Chỉ số | rho | NARVAL `riskkd` | **γ = 0.1** | noend (bản ghim cũ) |
|---|---:|---:|---:|---:|
| RM select so với rho | — | — | **+1.18 [0.49, 1.86]** | +1.09 [0.32, 1.85] |
| Winogrande (hf fp32) | 57.85 | 58.72 | **58.33** | 57.14 |
| RM test, EOS, độ dài | −3.02, 98.4%, 247 | — | **chưa đo** | −0.48, 3.1%, 1,011 |

Còn thiếu trước khi dùng hoặc chạy full: `pd_diagnostics` heldout cho γ = 0.1 (EOS, độ dài, RM test),
GSM8K, TruthfulQA; thử γ 0.05 và 0.03.
