# Analisis V2

V2 awal depth2/width1 memiliki 67.603 parameter; Plain dan Residual sama karena identity
addition tanpa bobot. Residual depth2 aktif enam kali. Pada depth1 default kanal berubah
pada semua stage, sehingga mengaktifkan flag residual saja tidak menambah identity aktif.

D1 W0.75 expansion1 plain 17.047 parameter. DW/BN/ReLU/PW/BN/ReLU setara V1 pada width
sama; penempatan ReLU dalam container berbeda tidak menciptakan fungsi baru. Regression
test memetakan leaf weights eksplisit dan memverifikasi output/input-gradient identik.
Ini temuan kesetaraan konfigurasi, bukan checkpoint conversion otomatis.

Kandidat opt-in `v2_candidate_stage3`: width0.75, depth1, stage_depths [1,1,1,2,1,1],
residual_stages [3] (indeks stage nol). Satu blok 48→48 stride1 ditambahkan setelah
stage keempat, memberi satu identity aktif dan 19.975 parameter. Kontrol dengan repeat
sama dan residual=false juga 19.975 parameter. Inilah pasangan yang mengisolasi skip.

Lokasi alternatif stage1 memakai stage_depths [1,2,1,1,1,1], 17.935 parameter. Memindahkan
blok antar-stage sekaligus mengubah dimensi/biaya blok; ini trade-off lokasi, bukan
perbandingan budget parameter yang persis identik. Tidak ada projection shortcut berat.

Projection activation `post` adalah baseline; `pre` mengaktifkan sebelum sum, `linear`
tidak mengaktifkan setelah proyeksi. Uji pre/linear dipasangkan dengan kandidat residual
aktif; tanpa skip, pre dan post setara. Expansion1.5 legacy tetap tersedia, bukan default.
Seluruh kandidat merupakan hipotesis; belum dilatih ulang pada dataset Intel asli.
