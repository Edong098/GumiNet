# Training dan fine-tuning

Gunakan folder train/val/test dengan class mapping sama. Jalankan dari root checkout:
`guminet train --config configs/low_resource.yaml`. Edit path dataset/output dahulu.
`device: cpu` memaksa CPU; `auto` memilih CUDA jika tersedia. Seed dipasang oleh CLI
sebelum inisialisasi model. Penggunaan Python langsung harus memanggil
`seed_everything(seed)` sebelum `create_model`.

| Profile | Input | Batch | Workers | AMP |
|---|---:|---:|---:|---|
| low_resource | 128 | 4 | 0 | Tidak |
| balanced | 160 | 16 | 0 | Tidak |
| performance | 224 | 32 | 2 | Hanya CUDA |

Semua nilai dapat diubah via YAML. Tidak ada decoded-image caching. `threads: 2`
merupakan titik awal CPU, bukan nilai optimal universal. Windows multiprocessing
memerlukan `if __name__ == "__main__":` untuk entry point Python Anda.

AdamW/SGD + cross entropy, scheduler cosine/step/none, clipping opsional,
accumulation, early stopping berdasarkan validation macro F1 tersedia. Accumulation
membagi gradient dengan jumlah sampel aktual per window termasuk window terakhir.
BatchNorm tetap melihat microbatch, sehingga accumulation tidak identik dengan batch
besar untuk model dengan BatchNorm. `patience: 0` menonaktifkan early stopping.

Tqdm menampilkan batch loss/running accuracy. Ringkasan epoch mencatat val loss/accuracy/F1,
lr, waktu, images/s, RSS dan peak VRAM. Throughput training tidak termasuk evaluasi,
plot dan penulisan checkpoint; `epoch_seconds` mencakup evaluasi, bukan I/O checkpoint.
Log JSON/CSV dan loss/accuracy PNG disimpan pada output_dir.

## Resume

Tambahkan `resume: runs/low_resource/last.pt` ke YAML dan tingkatkan `epochs` menjadi
total epoch yang diinginkan, bukan tambahan epoch. Arsitektur, mapping dan preprocessing
harus cocok. Optimizer, scheduler, scaler, RNG Python/NumPy/Torch/CUDA serta generator
DataLoader dikembalikan. `last.pt` adalah kondisi akhir epoch; `best.pt` dipilih berdasarkan
validation macro F1. Scheduler yang di-resume mempertahankan horizon konfigurasi awal;
untuk memperpanjang jadwal cosine, mulai eksperimen fine-tuning baru, jangan menganggap
resume otomatis membuat jadwal baru. Exact resume diuji CPU dengan workers=0 dan scheduler
none; lintas GPU/toolchain atau worker paralel dapat berbeda.

Checkpoint menyimpan model_config, state_dict, class mapping, input_size, mean/std,
konfigurasi, epoch, best metric, history, optimizer/scheduler/scaler dan RNG. Loading
menggunakan `torch.load(weights_only=True)`, validasi versi/mapping/preprocessing dan
strict state_dict. Tetap perlakukan file asing sebagai input tak tepercaya; pembatasan
pickle tidak mencegah semua serangan resource exhaustion. Tidak ada pretrained weights
palsu; file hilang menghasilkan pesan yang jelas.

## Fine-tuning

```python
from guminet import GumiTrainer
from guminet.engine import prepare_finetuning
from guminet.data import make_loaders

loaders = make_loaders("dataset_new", input_size=128, batch_size=4)
model, groups = prepare_finetuning("runs/old/best.pt", num_classes=2,
    mode="partial", unfreeze_last=2, backbone_lr=1e-4, head_lr=1e-3)
trainer = GumiTrainer(model, {"input_size": 128, "output_dir": "runs/finetune"},
    loaders["train"].dataset.class_to_idx, optimizer_groups=groups)
trainer.fit(loaders["train"], loaders["val"])
```

`head` membekukan seluruh backbone termasuk statistik BN; `partial` membuka stage
terakhir; `full` membuka seluruh backbone. Head diganti untuk jumlah kelas baru.
Preprocessing harus dipilih konsisten dengan checkpoint asal jika transfer diinginkan.
CLI mendukung `finetune_checkpoint`, `finetune_mode`, `unfreeze_last`, `backbone_lr`,
dan `lr` (head). Jangan memakai `resume` sekaligus mengganti head.

## Evaluasi

`guminet evaluate --config configs/evaluate.yaml` membaca split yang diberikan tanpa
augmentasi, memakai preprocessing dari checkpoint. Output JSON/CSV/PNG mencakup
confusion matrix dan report per kelas. Test set tidak dipakai memilih hyperparameter.
Gunakan output_dir berbeda untuk eksperimen berbeda agar artefak tidak tertimpa.
