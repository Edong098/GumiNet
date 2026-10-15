# Analisis V1

Nano 8.407 dan Tiny 28.907 parameter (3 kelas) tetap baseline asli. Tiny menunjukkan
mean validation F1 92,353% vs Nano 89,438% pada laporan pengguna; test Nano justru
86,3333% vs Tiny 85%. Test seed/checkpoint belum diketahui: jangan memasangkan test
tersebut dengan run validasi tertentu atau menyimpulkan ranking umum.

Pipeline V1 memakai depthwise 3x3, pointwise 1x1, BN dan ReLU. Stride total 16 pada
input128 memberi output8x8. BN memakai microbatch4 meskipun accumulation2. Early
feature maps dapat mahal; memory footprint perlu diukur, bukan hanya parameter.

Refinement default-off: kernel5 (33.387 parameter Tiny), downsampling yang lebih lambat
di awal (parameter tetap), kanal balanced [16,24,40,56,72,88,112] dan late emphasis
[16,20,28,40,56,96,144], serta LeakyReLU0.1. Parameter alternatif kanal masih dalam
sekitar ±10% baseline, tetapi setiap biaya MAC/aktivasi harus dilaporkan. Kernel5
bukan otomatis lebih baik; delayed stride bukan diagnosis error buildings.
