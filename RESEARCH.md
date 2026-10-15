# Agenda riset GumiNet

GumiNet mengeksplorasi keseimbangan classification performance, parameter, activation
memory, peak training RAM, CPU training time/throughput, inference latency dan portability.
CNN ringan dan efisiensi training telah banyak diteliti. V1 adalah baseline dengan
operator depthwise-separable yang telah dikenal, bukan kontribusi novelty terverifikasi.

Hipotesis V2: mixing kanal dengan expansion sempit dan identity residual mungkin
mempertahankan representasi dengan overhead terbatas. Expansion menambah parameter
dan aktivasi; depthwise kecil tidak otomatis lebih cepat pada semua CPU.
Hipotesis harus diuji, termasuk hasil negatif. Minimal branching dan operator standar
memudahkan porting tetapi tidak membuktikan dukungan semua runtime/perangkat.

## Landasan

- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,
  Howard et al. (2017), https://arxiv.org/abs/1704.04861 — depthwise separable.
- MobileNetV2, Sandler et al. (2018), https://arxiv.org/abs/1801.04381 — inverted residual/linear bottleneck.
- Searching for MobileNetV3, Howard et al. (2019), https://arxiv.org/abs/1905.02244.
- ShuffleNet V2, Ma et al. (2018), https://arxiv.org/abs/1807.11164 — panduan efisiensi aktual.
- EfficientNetV2: Smaller Models and Faster Training, Tan & Le (2021),
  https://arxiv.org/abs/2104.00298 — juga mempertimbangkan efisiensi training,
  termasuk fused-MBConv dan progressive learning.
- GhostNet, Han et al. (2020), https://arxiv.org/abs/1911.11907.

## Protokol yang disarankan

1. Dataset nyata dengan split train/val/test tetap dan pemeriksaan subject leakage.
2. Pembanding SimpleCNN/MobileNets/ShuffleNet, EfficientNetV2 bila budget cukup.
3. Label pretrained/from-scratch; seed, optimizer, augmentation, batch efektif dan
   budget sama, serta eksperimen tuning baseline yang dilaporkan terpisah.
4. Ablation satu faktor: width, depth, downsampling, expansion, residual, resolusi.
5. Minimal tiga seed, rata-rata dan variasi accuracy/macro F1, peak RAM terisolasi,
   throughput dan latency pada perangkat nyata. Laporkan FLOP convention dan threads.
6. Evaluasi test satu kali setelah pemilihan berdasarkan validation; hindari bias tuning.

Belum ada bukti peningkatan V2, novelty, state-of-the-art, atau kelayakan universal
pada RAM 4 GB. Hasil sintetis hanya membuktikan jalur perangkat lunak dapat dieksekusi.

## Research Progress

Hasil Intel/Colab pengguna kini disimpan dengan provenance user_reported_google_colab.
Tiny mean validation F1 92,353% dan Nano89,438%; V2 Plain86,959%, Residual87,085%,
refined90,265%. Ini n=3 pada subset3kelas, bukan reproduksi Codex atau perbandingan
dengan EfficientNet/MobileNet/ShuffleNet. Lihat [laporan teknis](docs/research/TECHNICAL_REPORT.md).
Audit membuktikan kesetaraan operasi D1/expansion1/no-skip V1/V2 pada width sama.
Kandidat single-stage residual harus dibandingkan terhadap kontrol repeat tanpa skip.
Banyak percobaan di validation sama meningkatkan multiple-comparison bias; jumlah
percobaan, control utama dan split harus dicatat. Evaluasi lintas dataset diperlukan.
