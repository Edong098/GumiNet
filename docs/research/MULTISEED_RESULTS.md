# Hasil multi-seed

Semua angka Intel/Colab di dokumen ini berstatus `source: user_reported_google_colab`, bukan reproduksi oleh Codex. Detail asli dan hitungan ulang: [data historis](../../experiments/historical/user_reported_colab.json).

| Model | Parameter | Mean val F1 (%) | Sample std (pp) | Mean waktu dilaporkan (s) |
|---|---:|---:|---:|---:|
| guminet_nano | 8,407 | 89.438107 | 2.212523 | 82.180545 |
| guminet_tiny | 28,907 | 92.352679 | 0.866806 | 97.997958 |
| guminet_v2_plain | 67,603 | 86.958787 | 1.737214 | 149.087100 |
| guminet_v2_residual | 67,603 | 87.084525 | 1.458019 | 157.419380 |
| v2_refined_d1_w075 | 17,047 | 90.264559 | 1.980008 | 97.751080 |

Mean dan sample standard deviation dihitung ulang dari tiga nilai individual, ddof=1.
F1 dan accuracy memakai unit persen; std memakai percentage points. Summary prompt
berpresisi lebih rendah tetap disimpan beserta delta pembulatan dalam JSON, tidak ditimpa.
Nano mean accuracy 89,630%; Tiny 92,407%; V2 refined 90,370% (hasil hitung ulang).
Accuracy V2 Plain/Residual per-seed tidak tersedia. Waktu disebut training time oleh
pengguna tetapi batas timer tidak diketahui: bukan otomatis process CPU time.

Perbedaan V2 Residual−Plain sekitar 0,126 percentage points pada mean F1, dengan
arah perubahan antarseed tidak konsisten. N=3 kecil, tidak ada klaim signifikansi.
RNG, split, library dan hardware yang tidak diketahui dapat memengaruhi reproduksi.
