# Analisis resource

Waktu training Colab adalah user-reported wall seconds dengan batas timer tidak diketahui.
Model/hardware/library/split asli belum lengkap sehingga tidak boleh digabung sebagai
benchmark yang dikontrol sempurna. V2 refined dilaporkan ~97,75s vs Plain149,09s;
depth dan width berubah bersama pada perbandingan ini, mekanismenya belum terisolasi.

`model_architecture_summary` memakai copy/eval/CPU, hooks dan fork RNG. Mencatat stem+
block shapes, residual aktif, feature moments, parameter, conv/linear MACs, jumlah byte
output dan maksimum output tunggal. MACs tidak menghitung BN/ReLU/pooling/add; FLOPs2×MAC.
Estimated activation bytes adalah sum output stem+blok, bukan peak live activation,
backward saved tensors atau RSS. `summed_leaf_output_bytes` dapat menghitung alias/inplace.
Input acak tidak menguji feature preservation semantik.

`guminet profile` menjalankan satu Python exec baru per model secara berurutan, warm-up
lalu repeated measurements. JSON/CSV/Markdown mencatat PID, hardware/threads, serialized
weights, MAC, timings forward/backward/optimizer, throughput, RSS lifetime dan sampling.
RSS mencakup seluruh runtime, model asli plus salinan benchmark; bukan memori model saja.
Observability/gradient summary dijalankan sesudah benchmark agar tidak menambah timer;
RAM metrics benchmark dicatat sebelum diagnostic berikutnya.

RSS sampling250ms dapat melewatkan transient peak. Lifetime ru_maxrss berbeda dari RSS
sampel; Linux/Mac/Windows counters berbeda. Trainer mencatat wall training loop dan
process CPU time secara terpisah (CPU time dapat melebihi wall dengan multithreading).
Dataset hashing dilakukan sekali sebelum timer epoch dan membutuhkan I/O penuh train/val.
Tidak ada klaim laptop4GB pasti cukup atau parameter kecil pasti latency rendah.

## Pengukuran engineering lokal (bukan Colab/Intel)

Empat proses baru berurutan, input128, batch1, thread2, warm-up3, inference20repeats,
training5steps. Model belum dilatih; waktu berikut bukan hasil dataset pengguna.

| Model | Param | MAC | Infer median ms/batch | Forward ms | Backward ms | Sampled RAM MiB |
|---|---:|---:|---:|---:|---:|---:|
| guminet_nano | 8407 | 2092224 | 1.2649 | 2.0413 | 2.7473 | 435.93 |
| guminet_tiny | 28907 | 5953920 | 1.2958 | 1.9155 | 2.3436 | 437.17 |
| v2_refined_d1_w075 | 17047 | 3709728 | 1.0525 | 1.6416 | 2.0185 | 438.73 |
| v2_candidate_stage3 | 19975 | 4410144 | 1.1053 | 2.4759 | 1.9917 | 439.12 |

JSON/CSV lengkap: benchmarks/refinement_cpu_profile.*. Masing-masing mencatat ID,
timestamp UTC, commit/source hash, PID, konfigurasi, hardware/dependencies dasar dan
architecture summary. Host CPU/runtime bersama dapat bervariasi; satu proses per model
belum merupakan replikasi independen lintas waktu/perangkat atau uji signifikansi.
Parameter berbeda tidak berbanding lurus dengan latency pada run ini; tidak ada klaim
keunggulan atau generalisasi ke laptop. Sampling RSS bukan peak eksak.
