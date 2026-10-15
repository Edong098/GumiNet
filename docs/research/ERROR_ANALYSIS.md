# Analisis kesalahan test historis

Semua angka Intel/Colab di dokumen ini berstatus `source: user_reported_google_colab`, bukan reproduksi oleh Codex. Detail asli dan hitungan ulang: [data historis](../../experiments/historical/user_reported_colab.json).

Urutan baris aktual dan kolom prediksi: buildings, forest, sea.

Nano: accuracy86,3333%, macroF1 86,1203%, benar259/300.
```
75 16  9
 1 97  2
 9  4 87
```
Recall buildings75%, forest97%, sea87%.

Tiny: accuracy85%, macroF1 84,7575%, benar255/300.
```
74 18  8
 1 98  1
13  4 83
```
Recall buildings74%, forest98%, sea83%. Buildings tersulit pada kedua matriks ini.
Seed/checkpoint test tidak diberikan. Jangan menghubungkan test tersebut dengan salah
satu best epoch multi-seed yang belum diketahui, atau menyimpulkan downsampling penyebabnya.

Periksa gambar train/val (blur, komposisi, domain, label, duplicate/subject leakage),
confidence/calibration dan definisi label. Gambar kesalahan asli tidak tersedia sehingga
analisis visual belum dilakukan dan tidak dibuat ilustrasi prediksi palsu. Test harus
dibekukan; error analysis ini merupakan catatan hasil terdahulu, bukan objective selection.
