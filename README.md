This repository contains the code and data for a final year thesis project on text classification. It currently covers two datasets (`dataset_1`, `dataset_2`) and three stages:

1. **Dataset analysis and cleaning**: EDA, visualization, and cleaning of each dataset.
2. **Model training**: a BiLSTM classifier and a Transformer classifier.
3. **Indirect prompt labeling**: manual labeling of indirect-prompt candidates for `dataset_1`.

Further modules will be added as the project progresses.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [First-Time Setup](#first-time-setup)
- [Running the Scripts Later](#running-the-scripts-later)
- [Running the Pipeline](#running-the-pipeline)
- [Output](#output)
- [Notes](#notes)

---

## Pipeline Overview

For each dataset (split into train, validation, and test sets, plus their combination as a full set), the EDA and cleaning stage performs:

- Exploratory data analysis (EDA)
- Missing value analysis
- Label and category distribution analysis
- Augmentation ratio analysis
- Visualizations: label distribution, text length, per-label text statistics, category distribution
- Per-class word clouds (TF-IDF and Chi-Squared weighted)
- Dataset cleaning and duplicate removal

Plots are generated **before** and **after** cleaning; files prefixed with `clean_` come from the cleaned data.

The model stage then trains classifiers and saves metrics, plots, and checkpoints under the dataset's result directories.

---

## Project Structure

```text
WUB Thesis Work/
├── .venv/
├── README.md
├── requirements.txt
├── Code/
│   ├── EDA/
│   │   └── dataset_analysis_and_cleaning.py
│   ├── Model/
│   │   ├── bilstm_classifier.py
│   │   └── transformer_classifier.py
│   └── dataset_1_manual_labeling_indirect_prompts.py
├── dataset_1/
│   ├── raw_data_dir/                    # input: train / validation / test parquet files
│   ├── clean_data_dir/                  # cleaned splits + full_dataset.parquet
│   │   └── indirect_detection/
│   │       └── indirect_candidates.parquet
│   ├── train_result/                    # EDA plots (raw + clean_)
│   ├── validation_result/
│   ├── test_result/
│   └── full_result/                     # combined set
└── dataset_2/
    ├── raw_data_dir/                    # input: train / validation / test parquet files
    ├── clean_data_dir/                  # cleaned splits + full_dataset.parquet
    ├── train_result/                    # EDA plots (raw + clean_)
    ├── validation_result/
    ├── test_result/
    ├── full_result/
    ├── bilstm_result/                   # BiLSTM outputs
    └── classifier_result/               # Transformer outputs
        ├── checkpoints/
        └── label_mapping.json
```

Each raw split is a file named `train-00000-of-00001.parquet`, `validation-00000-of-00001.parquet`, or `test-00000-of-00001.parquet`. Result directories are created automatically if they don't already exist.

---

## Requirements

| Item | Recommended | Notes |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS | Windows (PowerShell) also supported |
| Python | 3.12.3 | See compatibility table below |

**Python compatibility:**

| Version | Status |
|---|---|
| 3.13.x and newer | ⚠️ Untested (see note below) |
| 3.12.x | ✅ Recommended |
| 3.11.x | ✅ Expected to work |
| 3.10.x | ⚠️ May work |
| 3.9.x  | ❌ Not recommended |

> **Using Python 3.13+?** This project hasn't been tested on it. Data-science packages (numpy, scipy, pandas, scikit-learn, wordcloud, etc.) sometimes lag behind new Python releases in publishing prebuilt wheels, which can cause `pip install -r requirements.txt` to fail or attempt a slow source build. Before trying, check that every package in `requirements.txt` has a published wheel for your Python version on [PyPI](https://pypi.org). If in doubt, stick to 3.12.x.

Check your installed version:

```bash
python3 --version      # Ubuntu/Linux
python --version       # Windows PowerShell
```

---

## First-Time Setup

Do this once, the first time you use the project on a machine.

### Ubuntu / Linux

```bash
# 1. Navigate to the project root
cd "WUB Thesis Work"

# 2. Create a virtual environment (use python3.12 if available)
python3 -m venv .venv
# or, if needed: python3.12 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Confirm the interpreter
which python && python --version

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**If Python 3.12 isn't installed at all** (no `python3.12` binary present), install it via the deadsnakes PPA, since Ubuntu's default repositories don't always carry the exact 3.12.x build:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```

Then create the environment as usual:

```bash
python3.12 -m venv .venv
```

**If Python 3.12 is installed but `venv` is missing:**

```bash
sudo apt update
sudo apt install python3.12-venv
python3.12 -m venv .venv
```

**If Python 3.12 still can't be installed** (e.g. restricted permissions, unsupported Ubuntu release), check what's available and fall back:

```bash
ls /usr/bin/python*
python3.11 -m venv .venv     # fallback, in order of preference: 3.12 > 3.11 > 3.10
```

### Windows (PowerShell)

```powershell
# 1. Navigate to the project root
cd "C:\path\to\WUB Thesis Work"

# 2. Check available Python versions
py -0

# 3. Create a virtual environment with Python 3.12
py -3.12 -m venv .venv
# fallback: py -m venv .venv

# 4. Activate it
.\.venv\Scripts\Activate.ps1

# 5. Confirm the interpreter
python --version
Get-Command python

# 6. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Execution policy error?** If activation fails with *"running scripts is disabled on this system"*, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then retry activation.

**If Python 3.12 isn't installed at all:**

1. Download the Python 3.12.x installer from the [official Python website](https://www.python.org/downloads/release/python-3123/).
2. Run the installer, check **"Add python.exe to PATH"**, then choose **Install Now** (or **Customize installation** if you want to control the install path).
3. Restart PowerShell so it picks up the new installation.
4. Confirm it's available:
   ```powershell
   py -0
   py -3.12 --version
   ```
5. Create the environment as usual:
   ```powershell
   py -3.12 -m venv .venv
   ```

**If Python 3.12 still can't be installed** (e.g. restricted permissions), fall back to whatever version `py -0` lists, in order of preference: 3.12 > 3.11 > 3.10.

---

## Running the Scripts Later

Once the environment exists and dependencies are installed, you don't need to repeat the setup. Each new terminal session only needs two steps:

**Ubuntu / Linux**

```bash
cd "WUB Thesis Work"
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
cd "C:\path\to\WUB Thesis Work"
.\.venv\Scripts\Activate.ps1
```

Your prompt will show `(.venv)` when the environment is active. Then run any script from [Running the Pipeline](#running-the-pipeline).

You only need to re-run `pip install -r requirements.txt` if `requirements.txt` has changed.

When you're finished (PowerShell or bash):

```bash
deactivate
```

---

## Running the Pipeline

> **Always run scripts from the project root** (`WUB Thesis Work/`), not from inside `Code/`. Paths such as `--dataset_dir dataset_2` are resolved relative to the root.

The commands below use Linux/macOS path separators. On Windows PowerShell, use backslashes for the script path (e.g. `python .\Code\EDA\dataset_analysis_and_cleaning.py ...`); the arguments are identical.

### Step 1: EDA and cleaning

Run this first. It reads the raw splits from `<dataset_dir>/raw_data_dir/` and writes cleaned splits to `<dataset_dir>/clean_data_dir/`, plus plots to the `*_result/` directories.

**dataset_2**

```bash
python Code/EDA/dataset_analysis_and_cleaning.py --dataset_dir dataset_2 --text_column text --binary_column label_binary --category_column label_category
```

**dataset_1**

```bash
python Code/EDA/dataset_analysis_and_cleaning.py --dataset_dir dataset_1 --text_column <text_col> --binary_column <binary_col> --category_column <category_col>
```

Replace the `<...>` placeholders with the column names used in `dataset_1`.

| Argument | Description |
|---|---|
| `--dataset_dir` | Dataset folder containing `raw_data_dir/` (e.g. `dataset_2`) |
| `--text_column` | Column containing the text |
| `--binary_column` | Column containing the binary label |
| `--category_column` | Column containing the category label |

What the script does:

1. Loads the training, validation, and test datasets.
2. Combines them into a full dataset.
3. Runs EDA and generates statistical, text, and category distribution plots for each split.
4. Generates per-class word clouds.
5. Cleans each dataset, removes duplicates, and saves the cleaned splits.

### Step 2: BiLSTM classifier

Trains on the cleaned data in `<dataset_dir>/clean_data_dir/`, so run Step 1 first.

```bash
python Code/Model/bilstm_classifier.py --dataset_dir dataset_2 --text_column text --category_column label_category
```

| Argument | Description |
|---|---|
| `--dataset_dir` | Dataset folder (e.g. `dataset_2`) |
| `--text_column` | Column containing the text |
| `--category_column` | Column containing the category label |

### Step 3: Transformer classifier

<!-- TODO: verify these arguments against transformer_classifier.py -->

```bash
python Code/Model/transformer_classifier.py --dataset_dir dataset_2 --text_column text --category_column label_category
```

Run `python Code/Model/transformer_classifier.py --help` to see all available options.

### Indirect prompt labeling (dataset_1)

```bash
python Code/dataset_1_manual_labeling_indirect_prompts.py
```

This works with the candidates in `dataset_1/clean_data_dir/indirect_detection/indirect_candidates.parquet`.

---

## Output

**EDA and cleaning** (same layout for `dataset_1` and `dataset_2`):

| Directory | Contents |
|---|---|
| `<dataset>/clean_data_dir/` | Cleaned train / validation / test splits and `full_dataset.parquet` |
| `<dataset>/train_result/` | Plots from the training set (raw and `clean_`) |
| `<dataset>/validation_result/` | Plots from the validation set (raw and `clean_`) |
| `<dataset>/test_result/` | Plots from the test set (raw and `clean_`) |
| `<dataset>/full_result/` | Plots from the combined dataset (raw and `clean_`) |

Each `*_result/` directory contains four plots per version: `*_eda_plots.png`, `*_category_distribution_log.png`, `*_per_class_wordclouds.png`, and `*_subword_token_stats_by_label.png`.

**Models** (`dataset_2`):

| Directory / file | Contents |
|---|---|
| `dataset_2/bilstm_result/best_model.pt` | Best BiLSTM checkpoint |
| `dataset_2/bilstm_result/classification_report.txt` | Per-class precision, recall, F1 |
| `dataset_2/bilstm_result/test_metrics.json` | Test-set metrics |
| `dataset_2/bilstm_result/test_predictions.parquet` | Test-set predictions |
| `dataset_2/bilstm_result/training_history.json` / `.png` | Training curves and their raw values |
| `dataset_2/bilstm_result/confusion_matrix.png`, `roc_curves.png` | Evaluation plots |
| `dataset_2/classifier_result/checkpoints/` | Transformer checkpoints |
| `dataset_2/classifier_result/label_mapping.json` | Label-to-index mapping |

---

## Notes

- Always use the `.venv` virtual environment to avoid conflicts with system-wide packages.
- Preferred Python order: **3.12.x → 3.11.x → 3.10.x**.
- Raw dataset files must remain in `<dataset_dir>/raw_data_dir/` under their original names. If you rename them, update the corresponding paths in `Code/EDA/dataset_analysis_and_cleaning.py`.
- **Keep secrets out of the repository.** Never commit API keys or access tokens (such as a GitHub token file). Add them to `.gitignore`, and revoke and regenerate any token that has been committed.
- Large generated files (model checkpoints, parquet data) are best excluded via `.gitignore` or tracked with [Git LFS](https://git-lfs.com).
- **Reproducibility tip:** if distributing this project, pin package versions in `requirements.txt` (e.g. `pandas==2.2.2` rather than `pandas`). This ensures consistent behavior across Python versions and operating systems, and makes the fallback version chain above reliable in practice.
