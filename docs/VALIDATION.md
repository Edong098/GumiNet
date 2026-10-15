# Hasil verifikasi aktual

Lingkungan: Linux x86_64, Python 3.12.14, PyTorch 2.14.1+cpu,
AMD EPYC 9V74, PyTorch threads=2. CUDA tidak tersedia. Daftar dependency aktual ada di
`requirements/validated-cpu-python312.txt`; daftar ini khusus lingkungan CPU/Linux
Python 3.12 yang diuji, bukan lock universal untuk seluruh platform.

## Pengujian

- `pytest -q --junitxml=runs/validation/pytest.xml`: **81 passed, 0 failed, 0 skipped**,
  52,66 detik pada run final. Dua warning deprecation berasal dari legacy ONNX exporter.
- Seluruh delapan registry diuji forward/backward CPU pada jumlah kelas 2/3/7 dan
  resolusi 32/128/160/192/224.
- Dataset lazy, RGB/deterministic transform, korupsi, mapping, exact-file leakage,
  stratified split dan isolasi test set dari tuning diuji.
- Training mengubah weights dan menghasilkan loss/accuracy/log/checkpoint/grafik.
  Resume CPU workers=0 scheduler=none identik bitwise dengan run tanpa interupsi.
- Gradient accumulation pada window terakhir yang pendek diuji terhadap batch besar
  dengan BN dibekukan/dropout nol; BN microbatch normal tetap tidak ekuivalen.
- Fine-tuning head/partial/full diuji; head-only training mempertahankan weights dan
  running statistics backbone secara persis.
- Evaluasi, safe checkpoint metadata, inference batch, CLI help dan alur CLI
  train/evaluate/benchmark/predict diuji.
- Optuna satu trial satu epoch, lalu study yang sama dilanjutkan satu trial lagi di
  SQLite; kedua trial selesai, export best params/CSV berhasil.
- Ketujuh ablation dijalankan satu epoch pada data kecil; runner comparison dilatih
  untuk Nano dan SimpleCNN. Tidak ada klaim kualitas klasifikasi dari pengujian ini.
- Adapter MobileNetV2/V3-Small, ShuffleNetV2, EfficientNetV2-S, GhostNet dan MobileNetV4
  diuji forward CPU from-scratch; bukan training/evaluasi penuh seluruh pembanding.
- ONNX checker dan parity Runtime pada batch/resolusi berbeda serta smoke quantization
  lulus. Ini tidak membuktikan parity untuk semua kemungkinan bentuk/input.
- FastAPI TestClient mencakup health/info/predict, file rusak, MIME dan batas byte.
  Server uvicorn juga benar-benar dijalankan: GET health/info dan multipart predict
  masing-masing HTTP 200; server kemudian dihentikan.
- `scripts/validate_notebook.py`: seluruh **10 sel kode** dieksekusi memakai kernel
  Jupyter CPU lokal, termasuk training, resume, evaluasi, inference, benchmark dan ONNX.
  Tidak menjalankan autentikasi Google Drive atau hosting Colab asli.
- Ruff, dependency consistency (`pip check`), instalasi editable dan build package
  diuji; lihat artefak lokal dist dan log validasi. CI workflow disediakan, belum
  dieksekusi di GitHub karena tidak dilakukan push.

## Smoke learning, bukan hasil riset

`scripts/verify_baseline.py` memakai Nano, tiga kelas warna, 128×128, train/val/test
masing-masing 24 gambar, 10 epoch, seed 42. Train loss 1,1039 → 0,9363; best validation
epoch 9. Synthetic test accuracy/macro F1 = 1,0. Angka ini hanya menunjukkan jalur
pembelajaran berfungsi pada data sangat mudah dan tidak menyatakan kemampuan objek nyata.

Laporan lengkap: `benchmarks/synthetic_validation.json`. Parameter Nano 8.407;
estimasi conv/linear MAC per gambar 2.092.224. Pengukuran contoh median CPU batch=1
1,028 ms (warm-up 3, repeats 20); tidak dapat digeneralisasi ke laptop/edge lain.
Peak RSS sampling epoch terakhir sekitar 450,2 MiB; bukan bukti kebutuhan minimum RAM
atau peak aktivasi tensor. Model weights FP32 33.628 byte; checkpoint lengkap 182.949 byte.

Parity ONNX FP32 max absolute error 2,384e-7. Eksperimen dynamic INT8 classifier
max absolute error 0,00538 pada lima input acak, toleransi 0,1. File ONNX justru
bertambah 40.020 → 43.234 byte; tidak ada klaim speedup/ukuran lebih kecil.

## Perbaikan selama implementasi

Ekspor CSV tuning awal gagal karena helper Optuna membutuhkan pandas. Diganti dengan
CSV standar Python; uji SQLite/resume kemudian lulus. Pengujian ablation menangkap
konflik argumen default expansion/residual pada factory V2; sudah diperbaiki dan
seluruh tujuh eksperimen diuji. Plot laporan memakai Figure independen agar tidak
mengganti backend notebook pengguna.

## Batas validasi dan langkah berikutnya

Belum diuji: CUDA/AMP pada hardware GPU, Windows/macOS, hosted Colab/Drive,
pretrained download eksternal, training pembanding skala penuh, dataset objek nyata,
full-CNN INT8 calibration, atau deployment mobile. Tidak ada pretrained GumiNet resmi.
V2 belum terbukti lebih baik. Sampling RAM mungkin melewatkan puncak singkat dan
high-water proses dapat dipengaruhi eksperimen sebelumnya. Jalankan eksperimen nyata
multi-seed dengan proses terpisah, split per subjek, dan budget baseline sebanding.

Install script/start instructions cloud disimpan sebagai draft. Penyimpanan draft
bukan publikasi snapshot atau validasi pemulihan di task baru. Pengguna dapat meninjau,
menyimpan perubahan dan memublikasikan environment melalui pengaturan lingkungan.
Tidak ada push GitHub, publikasi PyPI, atau deployment eksternal dilakukan.
