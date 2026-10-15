# Audit source sebelum refinement

Basis audit: commit `62a411f4e01aaacc858143ff56f8f81f6db54b80`. Working tree bersih
pada awal tugas. Audit mencakup source model, trainer/checkpoint, data/evaluation,
benchmark/comparison, tuning/ablation, inference/API/export, CLI, YAML, tests, notebook,
dan Markdown. Angka Colab baru pada instruksi belum terdapat dalam CSV/notebook/log
asli repositori; artefak lama hanya smoke sintetis. Data historis tidak dapat dicocokkan
ke checkpoint atau split manifest asli yang belum disediakan.

## Temuan sebelum perubahan

1. V1 stem stride 2 dan stride blok [2,1,2,1,2,1] menghasilkan resolusi
   64→32→32→16→16→8→8 untuk input 128. Tidak ada bukti bahwa ini penyebab error buildings.
2. V2 depth=1 expansion=1 tanpa skip mempunyai DW/BN/ReLU/PW/BN/ReLU seperti V1.
   ReLU V2 ditempatkan di luar `layers` tetapi secara fungsi sama. Names state_dict
   berbeda; jangan memuat weights silang langsung tanpa pemetaan eksplisit.
3. V2 residual depth=1 default kanal selalu berubah, sehingga semua identity nonaktif.
   V2 depth=2 asli memiliki enam identity aktif (blok pengulangan tiap stage).
4. V2 plain/residual asli 67.603 parameter vs Tiny 28.907 (3 kelas). Menggandakan
   semua stage meningkatkan biaya; perubahan width/depth bukan bukti mekanisme baru.
5. BN memakai microbatch; gradient accumulation tidak menambah ukuran batch statistik BN.
   Risiko microbatch kecil dicatat; baseline BN tidak akan diganti diam-diam.
6. Factory default dan metadata format_version=1 memadai, tetapi baseline compatibility
   perlu reference implementasi lama dan tes strict keys/numerical parity, bukan counts saja.
7. Trainer menyimpan RNG dan generator; seed untuk init model harus dipasang sebelumnya.
   Belum ada manifest dataset, git/source identity, ID eksperimen atau aggregate run report.
8. `epoch_seconds` mencakup validation, bukan checkpoint I/O; `train_seconds` adalah
   wall time bagian training, bukan process CPU time. Jangan menamai keduanya sama.
9. RAM sampling 250 ms bukan peak eksak; ru_maxrss adalah lifetime proses. Ablation lama
   berjalan dalam proses sama, sehingga high-water mark antar-model tidak terisolasi.
10. Estimasi MAC conv/linear valid dengan konvensi FLOPs=2 MACs. Jumlah byte leaf outputs
    menghitung alias/inplace berulang, bukan aktivasi hidup atau saved tensors backward.
11. Ablation lama hard-coded; tidak dapat memilih kelompok/per-model kwargs via YAML.
    Comparator lama melaporkan test per model; jangan menggunakan command itu untuk selection.
12. Tuning tidak membaca test; inference memulihkan preprocessing; export memakai operator
    standar dan legacy exporter (deprecation diketahui). Tidak perlu mengubah jalur ini.
13. Notebook lama sintetis, bukan log eksperimen Intel. Klaim lama 'belum ada hasil nyata'
    perlu diperjelas: kini ada laporan pengguna, belum reproduksi mandiri.

## Keputusan rekayasa berdasarkan audit

Pertahankan defaults, state_dict paths, format_version=1 dan registry lama. Tambahkan
opsi kernel/aktivasi/stage-depth/residual-location hanya secara opt-in. Kandidat V2
menambah satu blok identity compatible pada stage terpilih, bukan menggandakan semua
stage. Tambahkan alat observability non-mutating, checkpoint regression, experiment
tracking dan proses profiling terpisah. Semua perubahan diuji dengan data kecil;
tidak menjalankan dataset Intel penuh. Hasil numerical equivalence dan ukuran aktual
akan dicatat dalam laporan verifikasi refinement sesudah pengujian.
