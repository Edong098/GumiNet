# Sejarah eksperimen

Semua angka Intel/Colab di dokumen ini berstatus `source: user_reported_google_colab`, bukan reproduksi oleh Codex. Detail asli dan hitungan ulang: [data historis](../../experiments/historical/user_reported_colab.json).

1. V1 depthwise-separable diimplementasikan pada commit awal toolkit `62a411f`.
2. Nano/Tiny/Small menyediakan width/depth berbeda, tetap CPU-first.
3. Smoke CPU sintetis memverifikasi software; bukan benchmark objek dunia nyata.
4. Pengguna melaporkan eksperimen Intel Image Classification (Kaggle), buildings/forest/sea.
5. Ada rujukan perbandingan tujuh konfigurasi awal, tetapi tabel lengkap tujuh run tidak
   diberikan. Jangan mengisi skor Small/LowExpansion/model lain dari asumsi.
6. Multi-seed 42/123/2026 tersedia untuk Nano, Tiny, V2 Plain, V2 Residual, dan V2 refined.
7. V2 awal lebih besar/lambat pada laporan ini dan residual tidak konsisten antar-seed.
8. Refinement depth/width menemukan D1 W0.75 sebagai kandidat 17.047 parameter.
9. Audit menemukan D1 expansion1 tanpa skip setara operasi V1 width sama; tes numerik
   memverifikasi forward dan input-gradient setelah pemetaan bobot eksplisit.
10. Tahap sekarang menambahkan opsi V1, satu residual stage V2, observability dan ablation.

Tidak ada tanggal run asli, log/checkpoint Intel atau manifest split yang tersedia.
Tanggal perubahan repository bukan tanggal eksperimen pengguna. Data seed42 D1 W075
pada refinement dan tabel multi-seed saling tumpang tindih (satu run, beda pembulatan).
Model aktif: Nano untuk efisiensi, Tiny baseline utama V1, V2 Plain kontrol, V2 Residual
eksperimen skip. Model lain tetap tersedia. Tidak ada V3 atau factory default diganti.
