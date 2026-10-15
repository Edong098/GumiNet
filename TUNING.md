# Tuning dan ablation

Pasang `pip install -e '.[tuning]'`, lalu `guminet tune --config configs/tuning.yaml`.
`GumiTuner` menggunakan TPE, median pruning, SQLite persistence, dan n_jobs=1.
Default lima trial × lima epoch. Learning rate, batch, optimizer, weight decay,
dropout, resolusi dan scheduler dituning. Objective adalah best validation macro F1,
bukan test score. Split test tidak dimuat selama tuning. Semua trial memakai seed
awal sama untuk mengurangi variasi inisialisasi; hasil akhir tetap perlu multi-seed.

Study bernama sama dan database sama melanjutkan trial yang sudah tersimpan.
`trials` adalah jumlah trial tambahan pada pemanggilan itu. Mengubah search space
categorical pada study lama tidak didukung Optuna; buat nama/database baru.
`best_params.json` mencatat konfigurasi terbaik, `trials.csv` menyimpan status,
parameter, waktu training dan metrik memori. PRUNED bukan trial selesai.
RAM peak merupakan high-water mark proses seumur hidup, bukan peak terisolasi tiap
trial. Untuk perbandingan peak yang adil jalankan studi/eksperimen pada proses baru.

`guminet ablate --config configs/ablation.yaml` menjalankan tujuh eksperimen:
baseline V2-Plain, width 0.75, depth 1, perpindahan stride, expansion 1.5, residual,
dan input 128. Masing-masing mengubah satu faktor dari baseline. `ablations.json`
memuat desain dan hasil; CSV/PNG memuat F1/accuracy, parameter/MAC, latency, RAM,
training throughput dan waktu epoch. Model terbaik berdasarkan val F1 diukur di CPU.
Jadwal, seed, split, augmentasi dan budget epoch sama. Resolusi dan komputasi memang
berbeda sesuai faktor. Lakukan beberapa seed dan ukur peak RAM dengan proses terpisah
untuk laporan ilmiah; runner sederhana ini berjalan berurutan dalam satu proses.
Tidak ada klaim keunggulan dari eksperimen sintetis.

Trainer juga mencatat peak RSS sampling setiap 250 ms selama training, terpisah dari
high-water mark proses. Sampling dapat melewatkan lonjakan singkat dan tidak mengukur
aktivasi tensor secara eksklusif.
