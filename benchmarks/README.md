# Artefak pengukuran

`synthetic_validation.json` adalah hasil nyata dari `scripts/verify_baseline.py` pada
lingkungan CPU cloud. Data terdiri atas warna sintetis, 24 gambar per split, seed 42.
Ini verifikasi software, bukan benchmark akurasi objek dunia nyata atau bukti
keunggulan arsitektur. Tidak ada dataset atau weights yang didistribusikan.

Pengukuran latency/throughput hanya untuk instance, input, threads dan versi yang
tercatat. Sampling RAM setiap 250 ms dapat melewatkan transient peaks; peak proses
seumur hidup bukan ukuran aktivasi. Runtime shared/cloud dapat bervariasi.
Kuantisasi classifier pada run ini justru memperbesar file ONNX (40.020 → 43.234 byte);
tidak ada klaim compression atau speedup. Lihat BENCHMARK.md dan docs/VALIDATION.md.
