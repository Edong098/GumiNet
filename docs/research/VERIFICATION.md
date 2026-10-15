# Verifikasi engineering refinement

## Hasil aktual

- Baseline sebelum perubahan: 81 passed, dua warning legacy ONNX exporter.
- Suite refinement: **124 passed, 0 failed, 0 skipped**, 62,51 detik.
  JUnit lokal: runs/refinement-validation/pytest-final.xml. Dua warning deprecation ONNX
  tetap diketahui; TLS/checksum/assertion tidak dinonaktifkan.
- Setelah pelengkapan metadata ID/timestamp/source profiler, uji fresh-process profiler
  dijalankan lagi: **1 passed**. Legacy timing/provenance yang hilang ditandai null/tidak
  lengkap, bukan direkonstruksi sebagai angka CPU time palsu.
- Source lama v0.1.0 dipertahankan sebagai reference fixture. Delapan registry lama
  mempertahankan model_config dan strict state_dict keys serta exact output parity.
  Checkpoint legacy yang dibentuk dari reference lama dapat dimuat oleh loader baru.
- Checkpoint sintetis yang benar-benar tertinggal dari tahap sebelumnya juga berhasil
  dipakai GumiPredictor. Ini bukan checkpoint Intel pengguna (belum tersedia).
- V1 vs V2 D1/expansion1/no-skip width0.75: output dan input gradient persis sama
  setelah pemetaan eksplisit Conv/BN/Linear; bukan conversion checkpoint otomatis.
- Semua registry lama dan preset baru forward/backward CPU, jumlah kelas/resolusi
  berbeda. Kernel5, LeakyReLU, stride/kanal baru, projection post/pre/linear,
  residual incompatible/compatible dan parameter grid historis diuji.
- Residual zero-learned-branch probe menghasilkan derivative identity1 saat aktif dan
  derivative0 saat nonaktif; candidate stage3 memiliki satu identity aktif, 19.975 parameter.
- Summary tool memakai generator lokal tanpa global reseeding (regression test),
  mempertahankan weights/buffers serta CPU RNG caller dan melaporkan shape,
  MAC, bytes output, feature moments dan finite gradients. Bukan pengukuran semantic features.
- Training/resume/evaluation/inference/CLI/dataset/Optuna/ONNX/API lama tetap diuji.
- Hash manifest train/val berubah saat file berubah, resume mismatch ditolak, test tidak
  dipakai selection. Budget wall/processCPU dapat menghentikan run di batas epoch.
- 19 run sintetis tambahan satu epoch: enam V1, delapan V2, lima training-only.
  Laporan ringkas: synthetic_group_smoke.json. Tidak ada training Intel penuh.
- Empat profiling model dilakukan dengan PID terpisah secara berurutan. Hasil nyata
  berada di benchmarks/refinement_cpu_profile.json/csv; tanpa accuracy dataset asli.
- Ruff lint/format, pip check, editable environment dan build wheel/sdist diverifikasi.
  Package wheel dipasang pada target terpisah untuk pemeriksaan import/forward preset.

## Lingkup dan yang belum diuji

CPU Linux/Python3.12/Torch2.14.1+cpu pada environment cloud; GPU tidak tersedia.
Notebook lama diaudit source, tetapi tidak dijalankan ulang pada hosting Colab dalam
fase ini. Petunjuk Colab penelitian memerlukan checkout yang sudah berisi perubahan
lokal ini (belum di-push). Tidak ada data/log/checkpoint asli Intel untuk reproduksi,
analisis gambar kesalahan atau verifikasi split. Tidak ada klaim novelty, superiority,
full-CNN INT8, mobile portability, maupun causal feature preservation.

Regression tidak menjamin semua checkpoint eksternal arbitrer; ia mencakup formatv1
reference toolkit lama dan artefak sintetis lokal yang tersedia. Perubahan opt-in disimpan
secara eksplisit pada model_config; loader tidak mengganti struktur secara diam-diam.

## Artefak dan source

Audit: AUDIT.md; sejarah/data: EXPERIMENT_HISTORY.md dan experiments/historical;
ukuran/shape: architecture_measurements.json; profiler: benchmarks/refinement_cpu_profile.*;
protokol/manual Colab: ABLATION_PROTOCOL.md; laporan19bagian: TECHNICAL_REPORT.md.
Pengujian sintetis dan data pengguna disimpan terpisah dengan source label masing-masing.
Commit dibuat lokal per fase; tidak ada push, release, publikasi PyPI atau V3 pada tugas ini.

Regenerasi ukuran/shape/diagnostik: `python scripts/measure_architectures.py --output runs/architecture_measurements.json`. Seed42 dipasang sebelum setiap model; ini bukan accuracy eksperimen.
