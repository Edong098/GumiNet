# Protokol ablation

Gunakan split asli train900/val180/test300 yang identitasnya harus dipastikan. Runner
ablate hanya membuka train/val dan memilih checkpoint dengan val macro F1. Jangan
membuka test untuk memilih kandidat. Seeds42/123/2026, CPUthreads2, input128, batch4,
accumulation2, AdamW lr0.001, cosine10epoch. Unknown historical hyperparameters harus
dicatat sebagai asumsi eksplisit, bukan diklaim reproduksi identik.

- `configs/research/v1_ablation.yaml`: baseline, kernel5, delayed stride, dua kanal,
  LeakyReLU. Masing-masing satu faktor terhadap baseline.
- `configs/research/v2_ablation.yaml`: refined baseline; repeat tanpa skip; repeat dengan
  skip; lokasi alternatif; stride/kanal; projection activation. Field control menyatakan
  pasangan yang benar: candidate_residual vs single_repeat_control hanya skip berbeda.
  Projection variants memakai kandidat sebagai kontrol. Lokasi stage mengubah dimensi
  blok juga: laporkan budget aktual, bukan klaim isolated fixed-budget effect.
- `configs/research/training_ablation.yaml`: arsitektur tetap, satu dari lr/scheduler/
  augmentation/batch berubah. BatchNorm microbatch ikut terpengaruh pada batch-size study.
- `configs/research/equal_time.yaml`: budget tambahan 100 detik training wall, ceiling1000
  epoch, dicek pada akhir epoch sehingga dapat overshoot satu epoch. Scheduler horizon
  harus dilaporkan; ini protokol tambahan berbeda dari cosine10epoch.
  Set training_time_budget_metric: process_cpu untuk budget process CPU seconds.

Jalankan dari root checkout yang berisi perubahan terbaru:
```bash
guminet ablate --config configs/research/v2_ablation.yaml
guminet profile --config configs/research/profile.yaml
```

Colab:
```python
%cd /content/GumiNet
!pip install -e .
import yaml
from pathlib import Path
c = yaml.safe_load(Path("configs/research/v2_ablation.yaml").read_text())
c["dataset"] = "/content/intel_three_class"  # train/val, split tetap milik Anda
c["output_dir"] = "/content/guminet_v2_ablation_run01"
Path("/content/ablation_run01.yaml").write_text(yaml.safe_dump(c))
!guminet ablate --config /content/ablation_run01.yaml
```
Tidak menjalankan full dataset otomatis saat implementasi. Gunakan output_dir baru;
runner menolak menimpa run yang sudah memiliki checkpoint last.pt. Profiling RSS
per-model gunakan subprocess exec (profile), jangan high-water mark runner ablate.
