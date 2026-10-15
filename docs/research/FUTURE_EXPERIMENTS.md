# Eksperimen selanjutnya

1. Kunci identitas subset dan split pengguna; arsipkan manifest, config, logs dan environment.
2. Eksperimen pertama: V2 refined baseline → satu repeat tanpa skip → repeat identik
   dengan skip (stage3), seeds42/123/2026, budget10epoch. Bandingkan kandidat skip vs
   repeat-control untuk mekanisme residual; baseline vs repeat-control untuk biaya depth.
3. Profile masing-masing dalam proses baru pada CPU laptop target; jangan mengambil
   kesimpulan dari parameter count atau cloud EPYC saja.
4. Uji lokasi alternatif, kemudian projection activation dengan control yang tepat.
5. V1 kernel/stride/kanal satu faktor, lalu training-only dengan arsitektur tetap.
6. Tambahkan budget wall/process CPU yang setara dengan horizon scheduler eksplisit.
7. Evaluasi BN microbatch terpisah; jangan mengganti normalization baseline di tengah studi.
8. Setelah pemilihan berbasis validation selesai, evaluasi test sekali dan laporkan semua
   percobaan. Tambah dataset lain untuk generalisasi, bukan hanya angka subset Intel.
9. Benchmark MobileNet/ShuffleNet/EfficientNet dengan prosedur sebanding sebelum klaim.

Tidak ada V3, pretrained weights baru, release, push atau publikasi PyPI otomatis.
