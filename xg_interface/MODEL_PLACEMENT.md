# Panduan Penempatan Model File

## 📁 Lokasi Model File

Model file **xG** harus ditempatkan di direktori **root** (folder utama) dari aplikasi ini.

### Nama File dan Lokasi yang Diharapkan:
```
xg_interface/
├── xg_model.joblib    ← LETAKKAN MODEL FILE DI SINI
├── app.py
├── requirements.txt
├── setup.bat
├── run.bat
└── src/
    └── ...
```

## 🔧 Cara Menempatkan Model

1. **Jika Anda sudah memiliki model trained (.joblib file):**
   - Copy file model Anda ke folder `xg_interface/`
   - Rename file tersebut menjadi `xg_model.joblib`
   - Pastikan file berada di level yang sama dengan `app.py`

2. **Jika Anda belum memiliki model:**
   - Aplikasi akan otomatis membuat dummy model untuk demonstrasi
   - Dummy model akan disimpan sebagai `xg_model.joblib` di folder root
   - Anda bisa mengganti dummy model ini dengan model yang sudah di-train

## 📋 Format Model yang Didukung

Model harus berupa **scikit-learn Pipeline**, **LightGBM model**, atau **Calibrated model** yang sudah di-train dengan format:
- **Input**: DataFrame dengan kolom-kolom yang sesuai (lihat `constants.py`)
- **Output**: Probabilitas goal (0-1)
- **Format file**: `.joblib` (menggunakan `joblib.dump()`)

### Model yang didukung:
- Scikit-learn pipelines
- LightGBM classifiers  
- CalibratedClassifierCV dengan LightGBM
- Model lain yang kompatibel dengan joblib

## ✅ Verifikasi

Untuk memastikan model sudah ditempatkan dengan benar:

1. Jalankan aplikasi dengan `run.bat` atau `streamlit run app.py`
2. Jika model ditemukan, aplikasi akan langsung load model Anda
3. Jika model tidak ditemukan, aplikasi akan menampilkan warning dan membuat dummy model

## 🚨 Troubleshooting

**Problem**: Error "ModuleNotFoundError: No module named 'lightgbm'"
- **Solution**: Install LightGBM: `pip install --user lightgbm` or run `setup.bat`

**Problem**: Error "Model file not found"
- **Solution**: Pastikan file bernama `xg_model.joblib` berada di folder root

**Problem**: Error saat loading model
- **Solution**: Pastikan model kompatibel dengan joblib

**Problem**: Prediction error
- **Solution**: Pastikan model di-train dengan fitur yang sama seperti di `constants.py`

## 📝 Catatan Penting

- Model file **TIDAK** di-commit ke git (sudah ada di `.gitignore`)
- Backup model file Anda di tempat yang aman
- Untuk production, gunakan model yang sudah properly trained dan validated
