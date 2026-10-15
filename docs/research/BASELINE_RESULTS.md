# Hasil baseline

Semua angka Intel/Colab di dokumen ini berstatus `source: user_reported_google_colab`, bukan reproduksi oleh Codex. Detail asli dan hitungan ulang: [data historis](../../experiments/historical/user_reported_colab.json).

| Model | Parameter | Mean val F1 (%) | Sample std (pp) | Mean waktu dilaporkan (s) |
|---|---:|---:|---:|---:|
| guminet_nano | 8,407 | 89.438107 | 2.212523 | 82.180545 |
| guminet_tiny | 28,907 | 92.352679 | 0.866806 | 97.997958 |
| guminet_v2_plain | 67,603 | 86.958787 | 1.737214 | 149.087100 |
| guminet_v2_residual | 67,603 | 87.084525 | 1.458019 | 157.419380 |
| v2_refined_d1_w075 | 17,047 | 90.264559 | 1.980008 | 97.751080 |

Tiny dipertahankan sebagai baseline utama V1 berdasarkan mean validation F1 lebih tinggi
dan variasi lebih kecil dalam tiga seed ini. Nano dipertahankan sebagai baseline ultra-ringan.
Ini keputusan operasional dalam subset ini, bukan bukti Tiny selalu lebih baik.
Validation dipakai untuk memilih epoch; test results tidak boleh digunakan untuk memilih
arsitektur berikutnya. Accuracy dan macro F1 adalah metrik berbeda.

Tabel lengkap tujuh model awal tidak tersedia. Parameter program dapat dihitung untuk
model lain tetapi tidak menyediakan accuracy/training time yang hilang. Referensi kode
lama disimpan di tests/fixtures/legacy_models_v010.py untuk regression compatibility.
