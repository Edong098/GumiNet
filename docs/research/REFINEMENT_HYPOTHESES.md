# Hipotesis refinement

- Kernel5 mungkin menangkap konteks spasial lebih luas dengan biaya MAC/aktivasi berbeda.
- Delayed downsampling mungkin membantu detail bangunan, tetapi belum ada analisis gambar
  yang mendukung diagnosis kausal tersebut.
- Kanal balanced atau late emphasis mungkin memberi trade-off representasi/budget lebih baik.
- Identity pada satu stage mungkin memperbaiki aliran gradien tanpa menggandakan semua stage.
- Projection activation sebelum/sesudah sum atau linear dapat memengaruhi informasi negatif.
- Batch kecil mungkin memengaruhi BN; evaluasi statistik BN perlu studi terpisah.

Counter-hypotheses: tambahan blok dapat memperlambat CPU/tidak membantu generalisasi;
aktivasi linear dapat merusak training; parameter lebih sedikit belum tentu lebih cepat.
Statistik feature mean/variance/zero fraction dan gradient norms pada input acak hanya
diagnostik engineering. Tidak membuktikan feature preservation semantik atau accuracy.
