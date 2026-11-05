# Hướng tốt nhất (ghim cố định)

Ghim ngày 04/11/2026. Chỉ thay khi một run mới tốt hơn trên **tập test**, đồng thời tỉ lệ EOS
và độ dài gần với rho. Điểm tập select không đủ để thay.
Chi tiết đầy đủ: [best_version_2026-11-04.md](best_version_2026-11-04.md).

| Vai trò | Lựa chọn |
|---|---|
| Model dùng thật | **rho**: `hoangducanh1865/qwen3-0.6b-deita-sft-student` @ `fc0f225c` |
| Run CRPD tốt nhất (nghiên cứu) | **`crpd_raw-alpha0p03-noend`**: Modal `trinhson3392`, volume `crpd-qwen3-pilot-hda-x10` |
| Cấu hình code | **x10 crpd_raw turbo**: `CRPD_TURBO=1 CRPD_NO_END_TOKEN=1`, 150 bước × 128 cặp, alpha 0.03, nhánh `crpd-hda-review` |

| Chỉ số (64 prompt test) | rho | noend |
|---|---:|---:|
| RM so với rho | — | −0.48 [−1.24, +0.23] |
| EOS | 98.4% | 3.1% |
| Độ dài | 247 | 1,011 |

Thử tiếp theo: γ < 1 (0.1, 0.3), cặp teacher có cùng token kết thúc lượt.
