# Roadmap

- [x] Baseline V1, empat kandidat V2, Nano/Tiny/Small.
- [x] Lazy folder dataset, trainer CPU/GPU opsional, checkpoint/resume dan evaluasi.
- [x] Optuna, ablation, benchmark, fine-tuning, inference, ONNX, API, CLI dan notebook.
- [x] CPU tests dan workflow build package lokal/CI.
- [ ] Training/evaluasi multi-seed pada dataset nyata dan perangkat RAM terbatas.
- [ ] Profiling peak aktivasi hidup dan memory per eksperimen dalam proses terisolasi.
- [ ] Validasi CUDA/AMP pada hardware GPU, Windows/macOS, dan Colab hosting asli.
- [ ] Full CNN INT8 dengan calibration dataset dan pengujian perangkat edge.
- [ ] Ablation tambahan, latency variance/energy, publikasi hasil termasuk hasil negatif.
- [ ] Pretrained weights dengan provenance dataset dan model card setelah eksperimen.
- [ ] Publikasi paket/release hanya setelah persetujuan pemilik.

## Research Progress

- [x] Arsip numerik dan laporan historis pengguna (bukan reproduksi independen).
- [x] Audit kesetaraan V1/V2, safeguards defaults dan checkpoint reference v0.1.0.
- [x] Refinement opt-in V1 dan kandidat V2 satu residual aktif ~20k parameter.
- [x] Observability shape/gradient/activation estimate, YAML groups dan isolated CPU profiler.
- [x] Experiment ID/source/split content manifest dan JSON/CSV/Markdown report.
- [ ] Reproduksi Intel dengan split/log asli serta ablation multi-seed nyata.
- [ ] Validasi mekanisme fitur/generalization, BN kecil, CUDA dan perangkat laptop nyata.

Fokus tetap V1/V2, tanpa V3. Lihat docs/research/FUTURE_EXPERIMENTS.md.
