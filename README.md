# Guide: Project Overview, Dataset Analysis, and Modeling

This repository contains the source code and results for a final year thesis project focusing on prompt injection detection and classification. 

The project pipeline covers:
1. **Exploratory Data Analysis (EDA) & Data Cleaning:** Token-level insights, visualizations, label distributions, subword token statistics, and automated text cleaning/duplicate removal.
2. **Modeling (BiLSTM):** Training and evaluating a Bidirectional LSTM neural network classifier for prompt threat categorization.

---

## Project Structure

WUB Thesis Work/
├── .venv/
├── Code/
│   ├── EDA/
│   │   └── dataset_analysis_and_cleaning.py
│   └── Model/
│       ├── bilstm_classifier.py
│       └── transformer_classifier.py
├── dataset_1/                      # (e.g., Raw, clean parquet files, and EDA plots)
├── dataset_2/                      # (e.g., Raw, clean parquet files, EDA plots, and model outputs)
├── requirements.txt
└── README.md

---

## Requirements

| Item | Recommended | Notes |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS | Windows (PowerShell) also supported |
| Python | 3.12.3 | See compatibility table below |

**Python compatibility:**

| Version | Status |
|---|---|
| 3.13.x and newer | ⚠️ Untested — see note below |
| 3.12.x | ✅ Recommended |
| 3.11.x | ✅ Expected to work |
| 3.10.x | ⚠️ May work |
| 3.9.x  | ❌ Not recommended |

> **Using Python 3.13+?** Data-science packages (`numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, etc.) sometimes lag behind new Python releases in publishing prebuilt wheels. Check PyPI first or stick to 3.12.x to avoid slow source-build fallbacks.

Check your installed version:

python3 --version      # Ubuntu/Linux
python --version       # Windows PowerShell

---

## Setup

### Ubuntu / Linux

# 1. Navigate to the project root
cd "WUB Thesis Work"

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Confirm the interpreter
which python && python --version

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

### Windows (PowerShell)

# 1. Navigate to the project root
Set-Location "C:\path\to\WUB Thesis Work"

# 2. Create a virtual environment with Python 3.12
py -3.12 -m venv .venv
# fallback if py launcher is configured differently: python -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate.ps1

# 4. Confirm the interpreter
python --version

# 5. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

> **Execution policy error?** If activation fails, run:
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

---

## Running the Pipeline

Ensure your raw parquet files are placed under your dataset folder following this structure:
`dataset_X/raw_data_dir/train-00000-of-00001.parquet` (similarly for `test` and `validation`).

### Step 1: Run Exploratory Data Analysis & Cleaning
To analyze and clean a dataset (e.g., `dataset_2`), execute the EDA script from the project root:

python Code/EDA/dataset_analysis_and_cleaning.py \
    --dataset_dir dataset_2 \
    --text_column text \
    --binary_column label_binary \
    --category_column label_category

This step will:
- Read raw parquet splits, clean text contents, and drop duplicate records.
- Generate raw and clean EDA plots, token distribution summaries, and per-class word clouds inside respective subfolders (`dataset_2/train_result/`, `dataset_2/full_result/`, etc.).
- Save the cleaned parquet files directly to `dataset_2/clean_data_dir/`.

### Step 2: Train and Evaluate the BiLSTM Classifier
Once the clean data is generated, you can train and test the BiLSTM classification model using:

python Code/Model/bilstm_classifier.py \
    --dataset_dir dataset_2 \
    --text_column text \
    --category_column label_category \
    --epochs 3 \
    --batch_size 128

This step will:
- Build a subword vocabulary from the training set.
- Train the BiLSTM model with real-time ETA monitoring per epoch and evaluate validation scores.
- Save model checkpoints (`best_model.pt`), classification reports, confusion matrices, ROC curves, and test metrics to `dataset_2/bilstm_result/`.

### Deactivating the Virtual Environment
deactivate

---

## Output Architecture Summary

| Directory Path | Contents |
|---|---|
| `dataset_X/raw_data_dir/` | Original input parquet datasets |
| `dataset_X/clean_data_dir/` | Processed and deduplicated parquet datasets |
| `dataset_X/*_result/` | Raw and `clean_` prefixed plots, stats tables, and word clouds |
| `dataset_X/bilstm_result/` | Saved weights, performance curves, and prediction summaries |