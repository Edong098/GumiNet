# Export dan API opsional

## ONNX

`pip install -e '.[export]'`, lalu `guminet export --config configs/export.yaml`.
Export membaca checkpoint secara weights-only, menjalankan model pada CPU/eval,
memakai opset 17, dan dynamic batch/height/width. Legacy TorchScript exporter dipilih
secara eksplisit agar tidak memerlukan onnxscript. PyTorch baru mengeluarkan peringatan
deprecation; perubahan exporter di masa depan perlu parity test ulang.

Validasi ONNX checker dan ONNX Runtime membandingkan logits FP32 pada batch 2 resolusi
checkpoint dan batch 1 resolusi +16. Toleransi atol=1e-4, rtol=1e-3. File JSON sidecar
menyimpan class mapping, normalization, resolusi training dan hasil parity. Preprocessing
RGB/resize/normalisasi tetap tanggung jawab pemanggil ONNX, bukan operator dalam graph.

```python
from guminet.export import onnx_predict
# x: float32 NumPy array NCHW yang sudah dinormalisasi seperti checkpoint.
logits = onnx_predict("runs/export/guminet.onnx", x)
```

Kuantisasi adalah eksperimen dynamic INT8 pada MatMul/Gemm (classifier); convolutions
sengaja tetap FP32 demi kompatibilitas CPU Runtime. Jangan menyebutnya full INT8 CNN.
`validate_quantized` membandingkan output pada data acak dengan toleransi longgar 0.1;
ini hanya smoke parity, bukan jaminan accuracy dataset nyata atau speedup. Uji ulang
accuracy/latency dengan data representatif pada hardware target. Quantization CNN penuh
memerlukan calibration dataset dan studi operator/runtime tersendiri.

Untuk Android/iOS/edge diperlukan build ONNX Runtime yang cocok, pengaturan provider,
pre/postprocessing identik, serta uji perangkat. Tidak ada klaim langsung berjalan di
semua platform; belum ada deployment mobile yang diuji.

## FastAPI lokal

```bash
pip install -e '.[api]'
python examples/serve.py --checkpoint runs/low_resource/best.pt
```

Endpoint `GET /health`, `GET /model-info`, `POST /predict` dengan multipart field `file`.
MIME dan format aktual dibatasi JPEG/PNG/WebP, payload maksimal 5 MiB, maksimal
20 juta pixel, file rusak ditolak. Inference dijalankan dalam threadpool dengan lock
agar request bersamaan tidak menumpuk penggunaan model. Batas byte berlaku pada isi
file yang dibaca; production gateway tetap perlu batas request body dan concurrency.
Ini demo lokal tanpa autentikasi/rate limiting; bukan deployment production.

```bash
curl -F 'file=@image.jpg;type=image/jpeg' http://127.0.0.1:8000/predict
```

Contoh loopback hanya perintah lokal, bukan tautan preview cloud. Model inti tidak
mengimpor FastAPI sampai factory aplikasi dipanggil.
