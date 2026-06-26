# 🚀 MacLearStream — Machine Learning Framework Template

**A reusable & production-grade template** for machine learning projects using native Python + Streamlit. Built with clean architecture similar to professional enterprise frameworks, but without heavy overhead or dependencies.

> **Config-driven** — Switch to a new dataset or project by simply editing `config.yaml`, zero code refactoring needed.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| ⚙️ **Config-Driven** | A single `config.yaml` manages dataset paths, models, preprocessing pipelines, and UI navigation |
| 🔗 **Pipeline Engine** | Modular & chainable preprocessing steps — missing value handling, outlier removal, encoding, and scaling |
| 📋 **Experiment Tracker** | Built-in run logging for every training experiment (no external dependencies like MLflow/W&B required) |
| 🤖 **Model Registry** | Dynamic import engine — add new custom models by simply appending a YAML entry |
| 🎯 **Dual-Task Support** | Fully supports both **classification** AND **regression** tasks |
| 🔮 **Dynamic Form Builder** | Automatically generates UI prediction forms based on configured feature schemas |
| 🎨 **Theme Engine** | 4 premium pre-built theme presets (Dark Premium, Ocean Blue, Forest Green, Minimal Light) |
| 📊 **Automated EDA** | Comprehensive auto-profiling — distributions, correlation matrices, missing value trackers, and target balances |
| 🧠 **Train via UI** | Perform model training directly from Streamlit with built-in K-Fold cross-validation |
| 📈 **Model Comparison** | Interactive comparison dashboard for multi-model benchmark evaluation |
| 📄 **Batch Prediction** | Upload CSV files → run batch predictions → download results instantly |
| 📥 **Export System** | Download evaluation metrics, batch predictions, and model comparison tables as CSV/JSON |

---

## 📁 Project Structure

```
├── config.yaml               # Master project configuration (EDIT THIS)
├── requirements.txt          # Package dependencies
│
├── cmd/                      # Main applications & commands
│   └── app/                  # Application entry point directory
│       └── main.py           # Streamlit entry point (streamlit run cmd/app/main.py)
│
├── config/                   # Config engine layer
│   └── settings.py           # YAML loader & global config accessors
│
├── core/                     # Core framework engine
│   ├── pipeline.py           # Preprocessing pipeline engine (chainable steps)
│   ├── experiment.py         # Lightweight experiment tracker
│   └── exceptions.py         # Custom framework exception hierarchy
│
├── src/                      # UI Layer (Streamlit)
│   ├── assets/               # CSS injection & theme engine
│   ├── components/           # Reusable UI components (cards, charts, forms, widgets)
│   ├── layouts/              # Header & navigation sidebar
│   ├── routes/               # Dynamic router dispatcher
│   └── views/                # Modular page views (home, eda, training, evaluation, etc.)
│
├── services/                 # Business logic & orchestration layer
│   ├── data_loader.py        # Dataset loading, validation, & auto-profiling
│   ├── preprocessing.py      # Pipeline execution service
│   ├── training.py           # Training orchestrator & cross-validation service
│   ├── evaluation.py         # Classification & regression evaluation dispatch
│   ├── prediction.py         # Single & batch prediction execution
│   └── export.py             # CSV/JSON byte export service
│
├── models/                   # Model abstraction layer
│   ├── base.py               # Abstract wrapper for uniform model APIs
│   ├── registry.py           # Dynamic registry using importlib
│   ├── train_model.py        # CLI command: python -m models.train_model
│   └── evaluate_model.py     # CLI command: python -m models.evaluate_model
│
├── utils/                    # Shared utilities
│   ├── constants.py          # Path resolvers & global constants
│   ├── file_handler.py       # Artifact save/load utilities (pickle/joblib/JSON)
│   ├── helpers.py            # Custom formatting & calculation helpers
│   └── logger.py             # Structured colored logging
│
├── dataset/                  # Place your dataset CSV files here
├── artifacts/                # Automatically generated model & scaler artifacts
├── experiments/              # Automatically generated experiment run logs
└── notebooks/                # Experimental Jupyter notebooks
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/sulthannata02/maclearstream-v1.git
cd maclearnstream-v1

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Place your dataset CSV file inside the `dataset/` directory:

```
dataset/
└── data.csv
```

### 3. Edit Configuration

Update `config.yaml` to match your project and dataset specifications:

```yaml
project:
  name: My Project
  task_type: classification    # or regression

dataset:
  path: dataset/data.csv
  target_column: target        # name of the target variable column

features:
  - name: feature_1
    label: Feature One
    type: number
    min: 0
    max: 100
    default: 50
```

### 4. Run the Application

```bash
streamlit run cmd/app/main.py
```

---

## 🤖 Adding New Models

To integrate a new machine learning algorithm, simply add an entry under the `models` section in `config.yaml`:

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

> Ensure the corresponding package is installed in your environment (e.g., `pip install xgboost`).

---

## 🔧 Customizing the Preprocessing Pipeline

Adjust the `pipeline` section in `config.yaml` to customize data transformation steps:

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

## 🎨 Available Themes

| Theme ID | Description |
|---|---|
| `dark_premium` | Dark mode + vivid neon accents + glassmorphism cards |
| `ocean_blue` | Deep ocean gradient palette with clean, structured panels |
| `forest_green` | Nature-inspired colorway with soft drop shadows |
| `minimal_light` | Clean light mode with sharp typography and subtle borders |

Set your preferred default theme in `config.yaml`:

```yaml
ui:
  theme: ocean_blue
```

Alternatively, switch themes dynamically using the dropdown selector in the application sidebar.

---

## 💻 CLI Commands

You can run training and evaluation pipelines directly from the terminal without starting the web UI:

```bash
# Train all configured models
python -m models.train_model

# Evaluate all trained models & display comparison table
python -m models.evaluate_model
```

---

## 📦 Tech Stack

- **Python** — Core programming language
- **Streamlit** — Interactive web UI framework
- **Scikit-Learn** — Machine Learning ecosystem
- **Plotly** — Dynamic & interactive visualizations
- **PyYAML** — Configuration parsing engine
- **Joblib** — High-performance artifact serialization

---

## 📄 License

MIT License — free to use, modify, and distribute for any personal or commercial project.

---

<p align="center">
  Built with ❤️ using <strong>MacLearStream</strong>
</p>
