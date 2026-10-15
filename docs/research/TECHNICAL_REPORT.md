# Laporan teknis penelitian GumiNet V1/V2

Semua angka Intel/Colab di dokumen ini berstatus `source: user_reported_google_colab`, bukan reproduksi oleh Codex. Detail asli dan hitungan ulang: [data historis](../../experiments/historical/user_reported_colab.json).

## 1. Latar belakang

CNN ringan perlu menyeimbangkan representasi, biaya training dan latency CPU, bukan hanya parameter.

## 2. Tujuan GumiNet

CPU-first, CUDA opsional, operator PyTorch standar, experimental research project.

## 3. Desain baseline V1

Nano/Tiny tetap; DW–PW+BN/ReLU dengan pooling adaptif. Tiny baseline kualitas validation dan Nano efisiensi.

## 4. Desain eksperimental V2

Plain/Residual lama tetap; refined D1 W075 setara operasi V1 width sama. Kandidat satu residual aktif terpisah.

## 5. Dataset dan pembagian

Intel Kaggle: buildings/forest/sea; train900/val180/test300 (300/60/100 per kelas). Manifest asli belum tersedia.

## 6. Hardware dan environment

Historis: Google Colab CPU threads2; CPU model/RAM/library tidak tersedia. Lingkungan engineering baru dicatat terpisah, tidak dianggap Colab.

## 7. Konfigurasi training

128x128, batch4, accumulation2, AdamW0.001, cosine, maksimum10epoch, seed42/123/2026. Augmentation/weight decay/patience/timer asli belum lengkap.

## 8. Metode evaluasi

Validation accuracy dan macroF1 untuk best epoch; test accuracy/F1 terpisah. Sample std ddof1. Tidak ada uji signifikansi.

## 9. Hasil pengujian tujuh model awal

Prompt menyebut tujuh, tetapi tabel lengkap tujuh model tidak tersedia. Skor yang tidak dilaporkan tidak diada-adakan.

## 10. Hasil multi-seed

| Model | Parameter | Mean val F1 (%) | Sample std (pp) | Mean waktu dilaporkan (s) |
|---|---:|---:|---:|---:|
| guminet_nano | 8,407 | 89.438107 | 2.212523 | 82.180545 |
| guminet_tiny | 28,907 | 92.352679 | 0.866806 | 97.997958 |
| guminet_v2_plain | 67,603 | 86.958787 | 1.737214 | 149.087100 |
| guminet_v2_residual | 67,603 | 87.084525 | 1.458019 | 157.419380 |
| v2_refined_d1_w075 | 17,047 | 90.264559 | 1.980008 | 97.751080 |


## 11. Hasil refinement V2

Refined17.047 parameter: mean F1 90.264559% vs Plain86.958787%; depth/width berubah bersama. Ini observasi user-reported, bukan bukti mekanisme residual.

## 12. Parameter dan waktu CPU

Tiny28.907/~97.998s; Nano8.407/~82.181s; V2Plain67.603/~149.087s; V2Residual67.603/~157.419s; refined17.047/~97.751s. Waktu historis bukan process CPU seconds terverifikasi.

## 13. Analisis V1

Tiny validasi lebih stabil di subset ini; Nano tetap efisien. Tidak menyimpulkan dari test ranking karena checkpoint/seed test tidak diketahui.

## 14. Analisis V2

Depth1default tanpa identity; depth2enam identity. Kandidat19.975 parameter menambah satu blok48→48; residual-control identik tanpa skip tersedia.

## 15. Kesalahan klasifikasi

Buildings recall paling rendah; lihat ERROR_ANALYSIS untuk matriks. Gambar asli tidak tersedia; penyebab domain/label/downsampling belum diketahui.

## 16. Keterbatasan penelitian

N=3, tiga kelas, metadata asli tidak lengkap, multiple-comparison bias akibat validation dipakai berulang. Dataset lain wajib untuk generalisasi lintas domain.

## 17. Hipotesis pengembangan

Kernel/stride/kanal/aktivasi dan residual satu stage diuji satu faktor. Diagnostik gradient bukan bukti feature preservation semantik.

## 18. Rencana ablation

Mulai refined → repeat-control → repeat+skip, lalu lokasi/aktivasi. YAML dan prosedur Colab ada pada ABLATION_PROTOCOL. Full dataset tidak dijalankan otomatis.

## 19. Kesimpulan sementara

Pertahankan Nano/Tiny; V2 belum terbukti lebih unggul. Refinement width/depth menjanjikan pada catatan pengguna, tetapi novelty, generalisasi dan mekanisme membutuhkan pengujian terkontrol.
