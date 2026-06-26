# 🚀 ML × Streamlit — Machine Learning Framework Template

**Template reusable & production-grade** untuk project machine learning menggunakan Python native + Streamlit. Arsitektur bersih seperti framework profesional, tapi tanpa overhead library berat.

> **Config-driven** — Ganti project baru cukup edit `config.yaml`, tanpa rewrite code.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|---|---|
| ⚙️ **Config-Driven** | Satu file `config.yaml` mengatur dataset, model, pipeline, dan UI |
| 🔗 **Pipeline Engine** | Preprocessing modular & chainable — missing values, outlier, encoding, scaling |
| 📋 **Experiment Tracker** | Built-in logging setiap training run (tanpa MLflow/W&B) |
| 🤖 **Model Registry** | Dynamic import — tambah model baru cukup tambah YAML entry |
| 🎯 **Dual-Task** | Satu template untuk **classification** DAN **regression** |
| 🔮 **Dynamic Form** | Form prediksi auto-generate dari config features |
| 🎨 **Theme Engine** | 4 preset tema (dark premium, ocean, forest, minimal light) |
| 📊 **Auto EDA** | Profiling dataset otomatis — distribusi, korelasi, missing values |
| 🧠 **Train via UI** | Training langsung dari Streamlit + cross-validation |
| 📈 **Model Comparison** | Dashboard perbandingan metrik semua model |
| 📄 **Batch Prediction** | Upload CSV → predict → download results |
| 📥 **Export System** | Download metrics, predictions, comparison sebagai CSV/JSON |

---

## 📁 Struktur Project

```
├── app.py                    # Entry point Streamlit
├── config.yaml               # Konfigurasi project (EDIT INI)
├── requirements.txt          # Dependencies
│
├── config/                   # Config engine
│   └── settings.py           # YAML loader + config accessor
│
├── core/                     # Core framework
│   ├── pipeline.py           # Pipeline engine (chainable steps)
│   ├── experiment.py         # Experiment tracker
│   └── exceptions.py         # Custom exceptions
│
├── src/                      # UI Layer (Streamlit)
│   ├── assets/               # CSS + theme engine
│   ├── components/           # Reusable components (cards, charts, forms)
│   ├── layouts/              # Sidebar + header
│   ├── routes/               # Dynamic router
│   └── views/                # Page views (home, eda, training, dll)
│
├── services/                 # Business logic
│   ├── data_loader.py        # Load + profiling dataset
│   ├── preprocessing.py      # Pipeline-based preprocessing
│   ├── training.py           # Training + cross-validation + tuning
│   ├── evaluation.py         # Metrik (classification & regression)
│   ├── prediction.py         # Single + batch prediction
│   └── export.py             # Export ke CSV/JSON
│
├── models/                   # Model layer
│   ├── base.py               # Model wrapper (consistent API)
│   ├── registry.py           # Model registry (dynamic import)
│   ├── train_model.py        # CLI: python -m models.train_model
│   └── evaluate_model.py     # CLI: python -m models.evaluate_model
│
├── utils/                    # Utilities
│   ├── constants.py          # Path + konstanta
│   ├── file_handler.py       # Save/load artifacts
│   ├── helpers.py            # Formatting + helpers
│   └── logger.py             # Structured colored logging
│
├── dataset/                  # Taruh CSV di sini
├── artifacts/                # Model artifacts (auto-generated)
├── experiments/              # Experiment logs (auto-generated)
└── notebooks/                # Jupyter notebooks
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/sulthannata02/maclearstream-v1.git
cd maclearnstream-v1

# Buat virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Siapkan Dataset

Taruh file CSV di folder `dataset/`:

```
dataset/
└── data.csv
```

### 3. Edit Config

Edit `config.yaml` sesuai dataset kamu:

```yaml
project:
  name: My Project
  task_type: classification    # atau regression

dataset:
  path: dataset/data.csv
  target_column: target        # nama kolom target

features:
  - name: feature_1
    label: Feature Satu
    type: number
    min: 0
    max: 100
    default: 50
```

### 4. Jalankan

```bash
streamlit run app.py
```

---

## 🤖 Cara Tambah Model Baru

Cukup tambah entry di `config.yaml`:

```yaml
models:
  - name: xgboost
    display_name: XGBoost
    class: xgboost.XGBClassifier
    task: classification
    params:
      n_estimators: 200
      max_depth: 6
      learning_rate: 0.1
```

> Pastikan package sudah terinstall: `pip install xgboost`

---

## 🔧 Cara Kustomisasi Pipeline

Edit section `pipeline` di `config.yaml`:

```yaml
pipeline:
  steps:
    - name: missing_values
      strategy: median          # median | mean | mode | drop

    - name: outlier_removal
      method: iqr               # iqr | zscore | none
      threshold: 1.5

    - name: encoding
      method: label             # label | onehot | none

    - name: scaling
      method: standard          # standard | minmax | robust | none

    - name: split
      test_size: 0.2
      random_state: 42
      stratify: true
```

---

## 🎨 Tema yang Tersedia

| Tema | Deskripsi |
|---|---|
| `dark_premium` | Dark mode + neon accent + glassmorphism |
| `ocean_blue` | Deep blue gradient + clean cards |
| `forest_green` | Nature palette + soft shadows |
| `minimal_light` | Light mode + sharp typography |

Ganti tema di `config.yaml`:

```yaml
ui:
  theme: ocean_blue
```

Atau pilih langsung dari sidebar di aplikasi.

---

## 💻 CLI Commands

```bash
# Train semua model
python -m models.train_model

# Evaluate semua model
python -m models.evaluate_model
```

---

## 📦 Tech Stack

- **Python** — Bahasa pemrograman utama
- **Streamlit** — Web UI framework
- **Scikit-Learn** — Machine Learning library
- **Plotly** — Interactive visualization
- **PyYAML** — Config parser
- **Joblib** — Model serialization

---

## 📄 License

MIT License — bebas digunakan, dimodifikasi, dan didistribusikan.

---

<p align="center">
  Built with ❤️ using <strong>MacLearStream</strong>
</p>
