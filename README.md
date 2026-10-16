# GumiNet

**Experimental Research Architecture · PyTorch · CPU-first · MIT**

## Latar belakang

GumiNet dikembangkan sebagai arsitektur Convolutional Neural Network (CNN) ringan untuk mempelajari keseimbangan antara kemampuan klasifikasi citra, waktu komputasi, dan penggunaan memori pada perangkat dengan sumber daya terbatas, terutama komputer berbasis CPU tanpa GPU khusus. Nama **GumiNet** berasal dari kata **“Gumi”** dalam bahasa Sasak, Lombok, yang berkaitan dengan bumi atau tanah, serta **“Net”** yang merupakan singkatan dari **Network** atau jaringan. Nama tersebut mencerminkan semangat pengembangan jaringan saraf yang membumi, sederhana, dan dapat diakses dengan perangkat komputasi yang lebih terjangkau.

## Tujuan pengembangan

Secara umum, GumiNet bertujuan mengembangkan dan mengevaluasi keluarga CNN ringan untuk klasifikasi citra dengan mempertimbangkan keseimbangan antara kemampuan klasifikasi dan biaya komputasi pada perangkat dengan sumber daya terbatas.

Secara khusus, pengembangan GumiNet bertujuan untuk:

1. Menyediakan baseline CNN yang dapat dilatih dan digunakan untuk inferensi pada CPU, dengan GPU sebagai akselerator opsional.
2. Menguji pengaruh kedalaman, lebar kanal, ukuran kernel, posisi downsampling, aktivasi, dan residual connection melalui eksperimen terkontrol.
3. Mengukur accuracy, macro F1, jumlah parameter, estimasi MACs, penggunaan RAM, waktu pelatihan, dan latency inferensi.
4. Menyediakan pipeline pelatihan, evaluasi, checkpoint, dan pencatatan eksperimen yang dapat direproduksi.
5. Mengidentifikasi konfigurasi yang sesuai untuk perangkat sasaran berdasarkan hasil pengukuran, serta mengevaluasi keterbatasan dan generalisasinya pada dataset lain.

## Status penelitian

GPU merupakan akselerator opsional. Tidak ada jaminan setiap konfigurasi cocok untuk
laptop RAM 4 GB. Belum tersedia pretrained weights resmi. Kini tersedia hasil Intel/Colab
yang dilaporkan pengguna, tersimpan terpisah dari pengujian software lokal; belum
direproduksi mandiri. V1 adalah baseline depthwise-separable; V2 adalah hipotesis eksperimen,
belum terbukti lebih unggul atau novel. Parameter/FLOPs kecil tidak menjamin latency rendah.
EfficientNetV2 juga secara eksplisit meneliti efisiensi training; lihat [RESEARCH.md](RESEARCH.md).

## Instalasi

Python >=3.10, PyTorch >=2.3. Dari checkout proyek ini:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
# Fitur opsional:
python -m pip install -e '.[tuning,benchmark,export,api,dev]'
guminet info
guminet models
```

Untuk GPU, pasang build PyTorch yang sesuai CUDA dari petunjuk resmi
[pytorch.org](https://pytorch.org/get-started/locally/), lalu `pip install -e .`.
Jangan memasang build CPU di atas build CUDA yang akan dipakai.
FastAPI, Optuna, torchvision/timm, dan ONNX tidak wajib untuk model inti.
Kode dapat dikembangkan di VS Code dengan interpreter virtual environment tersebut;
tidak memerlukan Codex. Repo: [Edong098/GumiNet](https://github.com/Edong098/GumiNet).
Perubahan lokal harus tersedia dalam checkout Anda; tidak ada push otomatis.

## Mulai cepat

```python
from guminet import create_model, guminet_tiny
import torch

model = create_model("guminet_tiny", num_classes=3)
logits = model(torch.randn(2, 3, 160, 160))
assert logits.shape == (2, 3)
```

```bash
# Data sintetis untuk memverifikasi alur saja. Menolak menimpa folder yang sudah ada.
python scripts/make_synthetic.py dataset
# CPU: ubah device: cpu di YAML; device: auto memakai CUDA bila tersedia.
guminet train --config configs/low_resource.yaml
guminet evaluate --config configs/evaluate.yaml
guminet predict --image dataset/test/class_0/0.png --checkpoint runs/low_resource/best.pt
guminet tune --config configs/tuning.yaml
guminet benchmark --config configs/benchmark.yaml
guminet export --config configs/export.yaml
guminet ablate --config configs/ablation.yaml
```

Semua command menyediakan `--help`. Konfigurasi `low_resource`, `balanced`, dan
`performance` dapat diubah melalui YAML. Untuk GPU gunakan `configs/performance.yaml`;
AMP hanya diaktifkan pada CUDA. Mulai dengan batch kecil dan ukur memori aktual.

## Model

V1: stem 3×3 stride 2, 3→16; enam GumiBlock (depthwise 3×3 → BN → ReLU →
pointwise 1×1 → BN → ReLU) dengan kanal 24/32/48/64/96/128 dan stride
2/1/2/1/2/1; global adaptive average pooling → dropout 0.2 → linear.
Resolusi default 160, preset 128/160/192/224, jumlah kelas dinamis.
Tidak ditemukan arsitektur eksperimen lama; alias `guminet_v1_original` tidak dibuat.

| Registry | Width | Pengulangan per stage | Expansion | Identity residual | Parameter (3 kelas) |
|---|---:|---:|---:|---|---:|
| guminet_v1 | 1.0 | 1 | — | Tidak | 28.907 |
| guminet_nano | 0.5 | 1 | — | Tidak | 8.407 |
| guminet_tiny | 1.0 | 1 | — | Tidak | 28.907 |
| guminet_small | 1.25 | 2 | — | Tidak | 102.859 |
| guminet_v2_plain | 1.0 | 2 | 1.0 | Tidak | 67.603 |
| guminet_v2_residual | 1.0 | 2 | 1.0 | Kompatibel saja | 67.603 |
| guminet_v2_low_expansion | 1.0 | 2 | 1.5 | Tidak | 175.123 |
| guminet_v2_low_expansion_residual | 1.0 | 2 | 1.5 | Kompatibel saja | 175.123 |

Angka dihitung dari `sum(p.numel() for p in model.parameters())`, termasuk parameter
BatchNorm yang dilatih; bukan buffer running statistics. Tiny identik dengan V1,
Nano mengecilkan width, Small mengubah width dan depth. Kandidat V2 memakai stage
berulang agar residual identity benar-benar aktif tanpa shortcut projection.
Lihat [ARCHITECTURE.md](ARCHITECTURE.md).

## Dataset

```text
dataset/
  train/class_a/*.jpg
  train/class_b/*.jpg
  val/class_a/*.jpg
  val/class_b/*.jpg
  test/class_a/*.jpg
  test/class_b/*.jpg
```

Kelas diurutkan konsisten di semua split. Folder test opsional untuk training.
Gambar dimuat saat dibutuhkan, dikonversi RGB, di-resize persegi, lalu dinormalisasi
mean/std ImageNet (tanpa menyiratkan pretrained). Training memakai flip dan brightness;
validasi dan inference deterministik. Verifikasi korupsi hanya membuka satu file
pada satu waktu; file rusak dicatat pada `dataset.corrupted`, kelas tanpa gambar valid ditolak.
`dataset.summary()` memberi distribusi kelas. SHA256 mendeteksi duplikasi file persis
antar-split; tidak mendeteksi near-duplicate atau subjek sama dalam gambar berbeda.

```python
from guminet.data import stratified_split
stratified_split("unsplit", "dataset_new", seed=42)
```

Split menyalin file ke folder baru, minimal tiga gambar valid per kelas. Lakukan
pemisahan per pasien/subjek/sesi jika domain memerlukannya, sebelum tuning.
Daftar path disimpan di RAM; decoded dataset tidak di-cache. Storage, jumlah path,
dan waktu I/O tetap membatasi dataset sangat besar.

## Training, evaluasi, dan checkpoint

[TRAINING.md](TRAINING.md) membahas CPU/GPU, resume, fine-tuning dan preprocessing.
Trainer menyimpan `best.pt`, `last.pt`, JSON/CSV, dan grafik loss/accuracy.
Best checkpoint dipilih dengan validation macro F1. Test hanya digunakan untuk evaluasi
akhir. Checkpoint berisi arsitektur, parameter, optimizer/scheduler/scaler, RNG,
class mapping, ukuran gambar, normalisasi, dan konfigurasi.
Loading menggunakan `weights_only=True` tanpa fallback pickle tak terbatas.

Evaluasi menghasilkan accuracy, precision/recall/F1 per kelas, macro/weighted F1,
classification report, confusion matrix, JSON, CSV, dan PNG.

## Tuning, benchmark, dan penelitian

[TUNING.md](TUNING.md): Optuna TPE, median pruning, SQLite, sequential trials dan
ablation satu faktor. [BENCHMARK.md](BENCHMARK.md): latency dengan warm-up/repeats,
forward/backward/optimizer timings, throughput, RAM/VRAM dan batas estimasi MACs.
Adapter pembanding: SimpleCNN, MobileNetV2/V3-Small, ShuffleNetV2 x0.5, EfficientNetV2-S;
GhostNet/MobileNetV4 opsional melalui timm. Nyatakan pretrained vs from-scratch.
Jangan menyimpulkan kualitas klasifikasi dari ukuran model atau smoke test sintetis.

## Inference, ONNX, API, dan Colab

```python
from guminet import GumiPredictor
predictor = GumiPredictor.from_checkpoint("runs/low_resource/best.pt", device="auto")
print(predictor.predict("image.jpg"))
# predictor.predict_batch(["a.jpg", "b.jpg"], batch_size=4)
```

[docs/EXPORT_API.md](docs/EXPORT_API.md) menjelaskan ONNX Runtime, validasi numerik,
eksperimen kuantisasi classifier INT8, dan FastAPI opsional (`/health`, `/model-info`,
`/predict`) dengan batas upload. Notebook [GumiNet_Colab.ipynb](notebooks/GumiNet_Colab.ipynb)
meliputi instalasi, dataset, visualisasi, training, validasi, Drive, resume,
benchmark, inference, dan ONNX. GPU/Colab Pro tidak diwajibkan. Eksekusi lokal notebook:
`python scripts/validate_notebook.py`. Mount Drive hanya dijalankan pada Colab.

## Struktur dan validasi

```text
guminet/
  models/ data/ engine/ evaluation/ tuning/ benchmark/ inference/ export/ utils/
  cli.py  __init__.py
configs/     YAML untuk training, evaluasi, tuning, benchmark, export, ablation
notebooks/   Tutorial Colab
examples/    Training CPU dan server API
scripts/     Generator data sintetis dan validator notebook
tests/       Model, data, training/resume, CLI, tuning, ONNX, API
benchmarks/  Hasil validasi terukur tanpa weights atau dataset
docs/        Panduan export/API dan hasil verifikasi
.github/workflows/cpu.yml  CPU tests dan build package
```

```bash
python -m pip install -e '.[tuning,export,api,dev]'
pytest -q
ruff check guminet tests scripts examples
python -m build
```

Hasil validasi refinement terbaru (**124 tes lulus**) dan batas pengujiannya tersedia di
[laporan verifikasi refinement](docs/research/VERIFICATION.md).
[Laporan implementasi awal](docs/VALIDATION.md) dipertahankan sebagai arsip.
Workflow GitHub Actions menjalankan CPU tests dan package build pada push/pull request;
hasil eksekusi remote perlu dilihat pada tab Actions. Roadmap di [ROADMAP.md](ROADMAP.md); kontribusi di [CONTRIBUTING.md](CONTRIBUTING.md).

Snapshot dependency yang divalidasi pada Linux CPU/Python 3.12 tersedia di
[requirements/validated-cpu-python312.txt](requirements/validated-cpu-python312.txt).
Pasang wheel CPU Torch/torchvision terlebih dahulu sebelum memakai snapshot itu;
untuk platform lain gunakan dependency di pyproject.toml dan jalankan ulang tests.
Training pembanding yang memakai engine sama: `guminet compare --config configs/comparison.yaml`.
Validasi alur sintetis mandiri: `python scripts/verify_baseline.py`.


## Research Progress

V1 dan Nano/Tiny/Small → verifikasi CPU → laporan Intel Kaggle tiga kelas → catatan
perbandingan tujuh konfigurasi (tabel lengkap belum tersedia) → multi-seed42/123/2026 →
identifikasi biaya V2/residual yang tidak konsisten → refinement width/depth → temuan
kesetaraan V1/V2 tertentu → opsi satu residual aktif dan ablation terkontrol.

[Riwayat](docs/research/EXPERIMENT_HISTORY.md), [laporan teknis](docs/research/TECHNICAL_REPORT.md),
[hasil multi-seed](docs/research/MULTISEED_RESULTS.md) dan
[data historis terstruktur](experiments/historical/user_reported_colab.json).
Seluruh angka Colab diberi source:user_reported_google_colab, bukan reproduksi Codex.

Model aktif tetap Nano (efisiensi), Tiny (baseline utama V1), V2 Plain (kontrol), dan
V2 Residual (eksperimen). Registry lama tetap tersedia. Preset opt-in baru:
`v2_refined_d1_w075` (17.047 parameter) dan `v2_candidate_stage3` (19.975, satu skip aktif;
jumlah untuk3kelas). Tidak ada V3 atau perubahan default factory.

```bash
guminet architecture --config configs/research/architecture.yaml
guminet profile --config configs/research/profile.yaml
guminet ablate --config configs/research/v1_ablation.yaml
guminet ablate --config configs/research/v2_ablation.yaml
guminet ablate --config configs/research/training_ablation.yaml
```

Ablation menggunakan validation, tidak membaca test. Set path dataset/output sebelum
menjalankan; full dataset tidak otomatis dilatih saat implementasi. Run baru mencatat
experiment ID, timestamp, git/source hash, train/val content manifest, konfigurasi,
dependency/hardware, parameter/MAC, waktu wall/processCPU dan checkpoint lineage dalam
JSON/CSV/Markdown. Lihat [protokol Colab](docs/research/ABLATION_PROTOCOL.md).
Status tetap experimental: ada risiko multiple-comparison bias pada validation sama,
dan generalisasi memerlukan dataset lain. Belum ada bukti V2 lebih unggul atau novelty.
