# Kontribusi

Gunakan Python >=3.10 dan virtual environment. Pasang PyTorch CPU terlebih dahulu,
lalu `pip install -e '.[tuning,export,api,dev]'`. Jalankan `pytest -q`,
`ruff check guminet tests scripts examples`, dan `python -m build` sebelum mengusulkan PR.
Jalankan notebook dengan `python scripts/validate_notebook.py` bila tutorial berubah.

Pertahankan V1 sebagai baseline. Tambahkan arsitektur eksperimental dengan nama baru,
unit test CPU forward/backward, metadata checkpoint yang dapat direkonstruksi, dan
hasil pengukuran dengan protokol jelas. Jangan menambahkan benchmark/accuracy fiktif,
pretrained weights palsu, dataset besar, checkpoint, virtual environment, atau secrets.
Hasil penelitian perlu seed/split/hardware dan pembeda pretrained/from-scratch.

Gunakan Git dan VS Code biasa; Tidak ada push/publish otomatis.
Lisensi MIT yang sudah ada tetap dipertahankan.
