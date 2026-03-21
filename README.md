# Chicago Crime Analysis

Predictive analysis of Chicago Police Department crime data: classify crime type, beat (location), and time-of-day from incident attributes. This project includes data loading, cleaning, exploratory visualizations, and supervised modeling (Logistic Regression, Random Forest, Gradient Boosting, etc.).

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Troubleshooting](#troubleshooting)

---

## Requirements

<!-- Python version and system dependencies needed before installing Python packages -->
- **Python** 3.11 or later
- **macOS (LightGBM)**: Homebrew-installed `libomp` (see [Troubleshooting](#troubleshooting) if LightGBM fails)

---

## Installation

<!-- Install project dependencies via pip (uses pyproject.toml). Alternative: uv, poetry. -->

### Option A: Pip (recommended)

```bash
# From the project root
cd chicago-crime-analysis
pip install -e .
```

### Option B: UV (faster)

```bash
uv pip install -e .
```

### Option C: Poetry

```bash
poetry install
```

### Verify installation

```bash
python -c "from src import load_raw_crimes, run_full_pipeline; print('OK')"
```

---

## Data Setup

<!-- Where to get the raw crime CSV and where to place it so notebooks and scripts can find it. -->

1. **Download** the Chicago Crimes dataset from the [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2).
2. Export or download as **CSV**.
3. Save the file as `crimes.csv` in `data/raw/`:

   ```text
   chicago-crime-analysis/
   └── data/
       └── raw/
           └── crimes.csv   # place your file here
   ```

4. Optionally, create the processed output folder:

   ```bash
   mkdir -p data/processed
   ```

---

## Project Structure

<!-- Brief explanation of each major folder and file so users know where to look. -->

```text
chicago-crime-analysis/
├── data/
│   ├── raw/
│   │   └── crimes.csv           # Raw crime data (you provide)
│   └── processed/
│       └── crimes_cleaned.csv   # Output of cleaning step
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Load, inspect, missing values
│   ├── 02_data_cleaning.ipynb        # Clean, bin, filter, save
│   ├── 03_exploratory_data_analysis.ipynb  # Plots and EDA
│   └── 04_data_modeling.ipynb       # Classification models
├── src/
│   ├── load_data.py    # Load raw CSV, overview, missing analysis
│   ├── preprocess.py   # Clean, bin, save/load cleaned data
│   ├── eda.py          # EDA plots (geographic, temporal, heatmaps)
│   └── modeling.py     # Train/eval models, experiment logging
├── pyproject.toml      # Project metadata and dependencies
└── README.md           # This file
```

---

## How to Run

<!-- Step-by-step instructions for the standard workflow: notebooks first, then optional CLI usage. -->

### 1. Run notebooks (recommended for first time)

Run the notebooks in order. Each one assumes the previous step has been completed.

| Order | Notebook                       | Purpose                                                        |
|-------|--------------------------------|----------------------------------------------------------------|
| 1     | `01_data_understanding.ipynb`  | Load data, inspect structure, check missing values            |
| 2     | `02_data_cleaning.ipynb`      | Clean data, add features, save to `data/processed/`           |
| 3     | `03_exploratory_data_analysis.ipynb` | Visualize crime patterns, geography, time                 |
| 4     | `04_data_modeling.ipynb`      | Train classifiers (LR, RF, GB) and compare results            |

**From Jupyter / VS Code**

- Open each notebook and run all cells.
- Notebooks use `../data/raw/crimes.csv` and `../data/processed/crimes_cleaned.csv` (relative to `notebooks/`).

### 2. Run from the command line (scripts)

<!-- Use the src package as a library; run modules directly for quick tests. -->

From the project root:

```bash
# Ensure you're in the project root
cd chicago-crime-analysis
```

**EDA (all plots)**

```bash
python -m src.eda
```

**Full modeling pipeline** (loads cleaned data, trains all tasks, prints reports)

```bash
python -m src.modeling
```

**Custom usage in Python**

```python
from src import load_raw_crimes, clean_data, save_cleaned, run_all_eda, run_full_pipeline

# Load and clean
df = load_raw_crimes()
df_clean = clean_data(df)
save_cleaned(df_clean)

# Run EDA
run_all_eda(df=df_clean)

# Run modeling pipeline (sample 200k rows by default)
result = run_full_pipeline(sample_size=200_000)
```

---

## Troubleshooting

<!-- Common issues and fixes, especially LightGBM on macOS. -->

### LightGBM fails: `libomp.dylib` or OpenMP not found (macOS)

LightGBM uses OpenMP; on macOS it often needs `libomp` from Homebrew:

```bash
brew install libomp
```

Then reinstall LightGBM:

```bash
pip install --force-reinstall lightgbm
```

### `FileNotFoundError` for `crimes.csv`

- Confirm `data/raw/crimes.csv` exists.
- Run from the project root (`chicago-crime-analysis/`), or use absolute paths in your code.

### `crimes_cleaned.csv` not found

- Run `02_data_cleaning.ipynb` (or call `clean_data` + `save_cleaned` in Python) first.
- EDA and modeling expect cleaned data in `data/processed/crimes_cleaned.csv`.

---

## License

Add your preferred license here.
