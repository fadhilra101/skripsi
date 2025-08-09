# xG Prediction Interface

Aplikasi Streamlit untuk prediksi Expected Goals (xG) dari data tembakan sepak bola.

## Fitur

- **Prediksi Batch**: Upload file CSV berisi data tembakan untuk mendapatkan prediksi xG untuk setiap event
- **Simulasi Tembakan**: Simulasi tembakan custom dengan mengatur berbagai parameter
- **Visualisasi**: Shot map dengan nilai xG yang ditampilkan pada lapangan sepak bola

## Instalasi

### Opsi 1: User Installation (Recommended)
Jika mengalami permission error:
```bash
setup.bat
```

### Opsi 2: Virtual Environment (Safest)
Untuk isolasi package yang lebih baik:
```bash
setup_venv.bat
```

### Opsi 3: Manual Installation
```bash
pip install --user -r requirements.txt
```

### Opsi 4: Admin Installation
Run Command Prompt sebagai Administrator, lalu:
```bash
pip install -r requirements.txt
```

## Setup Model

Sebelum menjalankan aplikasi, baca panduan penempatan model di [`MODEL_PLACEMENT.md`](MODEL_PLACEMENT.md).

**TL;DR**: Letakkan file model Anda (`xg_model.joblib`) di folder root aplikasi ini.

## Cara Menjalankan

### Jika menggunakan User Installation
```bash
run.bat
```

### Jika menggunakan Virtual Environment
```bash
run_venv.bat
```

### Manual
```bash
streamlit run app.py
```

## Struktur Project

```
xg_interface/
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_manager.py        # Model loading and creation utilities
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── dataset_prediction.py   # Dataset prediction page
│   │   └── custom_shot.py          # Custom shot simulation page
│   └── utils/
│       ├── __init__.py
│       ├── constants.py            # Application constants and mappings
│       ├── data_processing.py      # Data preprocessing functions
│       └── visualization.py        # Visualization utilities
└── README.md
```

## Format Data Input

Untuk prediksi batch, file CSV harus memiliki kolom-kolom berikut:

- `minute`: Menit terjadinya tembakan (0-120)
- `second`: Detik terjadinya tembakan (0-59)
- `play_pattern`: Pola permainan (ID StatsBomb)
- `position`: Posisi pemain (ID StatsBomb)
- `shot_technique`: Teknik tembakan (ID StatsBomb)
- `shot_body_part`: Bagian tubuh untuk menembak (ID StatsBomb)
- `shot_type`: Jenis tembakan (ID StatsBomb)
- `shot_open_goal`: Gawang kosong (0/1)
- `shot_one_on_one`: Situasi satu lawan satu (0/1)
- `shot_aerial_won`: Duel udara dimenangkan (0/1)
- `under_pressure`: Di bawah tekanan (0/1)
- `start_x`: Koordinat X tembakan (0-120)
- `start_y`: Koordinat Y tembakan (0-80)
- `type_before`: Jenis event sebelum tembakan (ID StatsBomb)

## Teknologi

- **Streamlit**: Framework web app
- **Pandas**: Manipulasi data
- **NumPy**: Komputasi numerik
- **Scikit-learn**: Machine learning
- **Matplotlib**: Plotting
- **mplsoccer**: Visualisasi lapangan sepak bola

## Model

Aplikasi ini menggunakan model Logistic Regression dengan preprocessing menggunakan OneHotEncoder untuk fitur kategorikal. Jika model tidak ditemukan, aplikasi akan membuat model dummy untuk demonstrasi.
