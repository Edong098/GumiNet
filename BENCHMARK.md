# Benchmark yang dapat direproduksi

`guminet benchmark --config configs/benchmark.yaml` mengukur model salinan agar
weights/BN pengguna tidak berubah. Warm-up mendahului repeated inference dan training;
CUDA disinkronkan sebelum/sesudah timing. Catat ukuran input, batch, thread CPU,
versi Python/Torch, OS, CPU dan nama GPU bila ada.

Parameter total/trainable dihitung aktual. FP32 weight bytes = parameter × 4,
bukan ukuran file. `serialized_state_dict_bytes` diukur dengan file sementara;
`checkpoint_file_bytes` hanya terisi bila path checkpoint diberikan dan dapat mencakup
optimizer/history. MACs hanya estimasi conv/linear; FLOPs=2×MACs dengan konvensi
multiply-add dua operasi. BN/ReLU/pooling/add tidak termasuk.
`summed_leaf_output_bytes` adalah jumlah byte output operator, dapat menghitung ulang
tensor inplace, dan **bukan** peak aktivasi hidup atau peak training RAM.

Latency median/p95 adalah latency per batch, tidak termasuk decode, preprocessing,
dan transfer input ke device. Forward/backward/optimizer diukur dengan SGD pada batch
sintetis. Synthetic throughput/duration bukan hasil epoch dataset nyata. Waktu epoch
dan images/s dataset nyata ada di history GumiTrainer. RAM RSS diukur via psutil,
peak proses adalah high-water mark seumur hidup; CUDA peak adalah allocated tensor
memory selama bagian training benchmark, bukan seluruh penggunaan driver/device.
Luncurkan CLI di proses terpisah per model saat membandingkan peak RAM.

## Pembanding

```python
from guminet.benchmark import create_comparison_model, GumiBenchmark
model = create_comparison_model("mobilenet_v2", num_classes=3, pretrained=False)
metrics = GumiBenchmark(model, device="cpu").run(input_size=160, repeats=20)
```

Adapter: `simple_cnn`, `mobilenet_v2`, `mobilenet_v3_small`,
`shufflenet_v2_x0_5`, `efficientnet_v2_s` (lebih mahal, opsional),
`ghostnet_100`/`mobilenetv4_conv_small` melalui timm bila dipasang.
`pretrained=True` pada torchvision/timm dapat mengunduh weights dan mengganti head;
GumiNet tidak menyediakan pretrained weights. Jangan membandingkan akurasi pretrained
vs from-scratch tanpa label eksplisit. Adapter tidak mengubah preprocessing otomatis;
pretrained torchvision/timm memiliki rekomendasi masing-masing yang perlu dicatat.

Untuk penelitian gunakan split, resolusi, augmentasi, normalization, optimizer,
budget epoch, batch efektif, seed dan hardware sebanding; laporkan juga hasil dengan
hyperparameter sesuai tiap baseline. Ulangi latency dan training pada perangkat target,
sertakan variance. `plot_comparison` menerima rows dengan nama dan metrik untuk grafik
performance/resource. Parameter kecil saja tidak membuktikan latency atau akurasi lebih baik.

## Training pembanding dengan prosedur sama

`guminet compare --config configs/comparison.yaml` memakai GumiTrainer yang sama,
seed/split/transform/optimizer dan budget sama untuk setiap adapter. Model dipilih
berdasarkan val macro F1 lalu dinilai pada test (atau val bila test tidak ada, dengan
label eksplisit). Artefak tiap model dipisahkan, termasuk checkpoint, CSV/JSON/PNG.
`pretrained_models` adalah daftar nama yang memakai bobot awal pretrained; default kosong.
Jangan memakai hasil test comparison sebagai objective tuning.

Training dan benchmark kini juga mencatat `sampled_peak_training_ram_mb` melalui
sampling RSS setiap 250 ms selama bagian training. Ini pengukuran terpisah dari peak
seumur hidup, tetapi dapat melewatkan puncak sangat singkat; tidak sama dengan peak
aktivasi. Field `dataset_sampled_peak_training_ram_mb` pada compare berasal dari epoch
terakhir. Jalankan proses baru dan beberapa ulangan per model untuk pengukuran ilmiah.

## Research Progress dan profiler terisolasi

`guminet profile --config configs/research/profile.yaml` memakai exec proses baru secara
berurutan per model. Jangan membandingkan lifetime RSS dari beberapa model pada runner
ablation yang sama. Sampling RAM tetap250ms dan dapat melewatkan puncak singkat.
Model asli plus salinan benchmark, runtime dan optimizer ada dalam lingkup proses.
Feature/gradient diagnostics diambil setelah pengukuran benchmark. `guminet architecture`
menyediakan shape/MAC/aktivasi/residual report; estimasi aktivasi bukan peak RSS.
Trainer kini mencatat process CPU seconds terpisah dari training wall time. Laporan
user Colab tidak memiliki batas timer pasti; jangan disamakan dengan timer baru.
Lihat [RESOURCE_ANALYSIS](docs/research/RESOURCE_ANALYSIS.md).
