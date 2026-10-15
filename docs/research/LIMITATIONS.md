# Keterbatasan

Dataset hanya tiga kelas Intel dan subset kecil; belum ada generalisasi lintas domain.
Tiga seed tidak cukup membuktikan signifikansi atau ranking universal. Banyak percobaan
pada validation sama meningkatkan multiple-comparison/selection bias; laporkan jumlah
percobaan, pra-daftarkan pasangan utama, gunakan nested validation atau validation baru
jika tersedia. Jangan memakai test yang sudah dilihat untuk pemilihan desain berikutnya.

Catatan pengguna bukan reproduksi independen. Tidak ada log/notebook Intel asli, hash
split, CPU model/versi library/peak RAM historis atau checkpoint untuk memverifikasi asal
angka. Tidak ada skor tujuh model awal lengkap; null berarti tidak diketahui.
D1 W075 refinement dan seed42 multi-seed adalah run yang sama, bukan tambahan n.

Operasi tertentu V1/V2 setara, sehingga penamaan versi tidak membuktikan mekanisme baru.
Kandidat baru hanya diuji fungsional dengan input/dataset sintetis kecil; belum diuji
accuracy objek asli, CUDA/AMP hardware, hosted Colab, mobile atau novelty.
Parameter/MAC/activation estimates bukan latency/peak memory terukur.
Data loader hashing exact-file tidak mendeteksi semua near-duplicate atau subject leakage.
Checkpoint regression dari reference v0.1.0 memakai weights test-generated, bukan artefak
training Intel pengguna yang belum disediakan.
