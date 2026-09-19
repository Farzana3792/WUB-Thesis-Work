# Final Year Thesis

## Guide: Dataset Analysis and Cleaning

This is the initial component of a larger final year thesis project. It covers setup and usage for the dataset analysis and cleaning stage; further modules will be added as the project progresses.

This stage provides a reproducible pipeline for exploratory data analysis (EDA), visualization, and cleaning of a text classification dataset split into training, test, and validation sets (and their combination).

For each dataset split, the pipeline performs:

- Exploratory data analysis (EDA)
- Missing value analysis
- Label and category distribution analysis
- Augmentation ratio analysis
- Visualizations: label distribution, text length, per-label text statistics, category distribution
- Per-class word clouds (TF-IDF and Chi-Squared weighted)
- Dataset cleaning and duplicate removal

---

## Project Structure

```text
WUB Thesis Work/
├── .venv/
├── requirements.txt
└── dataset/
    ├── dataset_analysis_and_cleaning.py
    ├── raw_data_dir/
    │   ├── train-00000-of-00001.parquet
    │   ├── test-00000-of-00001.parquet
    │   └── validation-00000-of-00001.parquet
    ├── train_result/        # generated plots (training set)
    ├── test_result/         # generated plots (test set)
    ├── validation_result/   # generated plots (validation set)
    └── full_result/         # generated plots (combined set)
```

Result directories are created automatically if they don't already exist.

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

> **Using Python 3.13+?** This project hasn't been tested on it. Data-science packages (numpy, scipy, pandas, scikit-learn, wordcloud, etc.) sometimes lag behind new Python releases in publishing prebuilt wheels, which can cause `pip install -r requirements.txt` to fail or attempt a slow source build. Before trying, check that every package in `requirements.txt` has a published wheel for your Python version on [PyPI](https://pypi.org). If in doubt, stick to 3.12.x.

Check your installed version:

```bash
python3 --version      # Ubuntu/Linux
python --version       # Windows PowerShell
```

---

## Setup

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
2. Run the installer — check **"Add python.exe to PATH"**, then choose **Install Now** (or **Customize installation** if you want to control the install path).
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

## Running the Pipeline

```bash
cd dataset
python dataset_analysis_and_cleaning.py       # Ubuntu/Linux
python .\dataset_analysis_and_cleaning.py     # Windows
```

The script will:

1. Load the training, test, and validation datasets.
2. Combine them into a full dataset.
3. Run EDA and generate statistical, text, and category distribution plots for each split.
4. Generate per-class word clouds.
5. Clean each dataset and remove duplicates.

### Deactivating the environment (powershell or bash)

```bash
deactivate
```

---

## Output

| Directory | Contents |
|---|---|
| `dataset/train_result/` | Plots from the training set |
| `dataset/test_result/` | Plots from the test set |
| `dataset/validation_result/` | Plots from the validation set |
| `dataset/full_result/` | Plots from the combined dataset |

---

## Notes

- Always use the `.venv` virtual environment to avoid conflicts with system-wide packages.
- Preferred Python order: **3.12.x → 3.11.x → 3.10.x**.
- Raw dataset files must remain in `dataset/raw_data_dir/`. If you rename them, update the corresponding paths in `dataset_analysis_and_cleaning.py`.
- **Reproducibility tip:** if distributing this project, pin package versions in `requirements.txt` (e.g. `pandas==2.2.2` rather than `pandas`). This ensures consistent behavior across Python versions and operating systems, and makes the fallback version chain above reliable in practice.