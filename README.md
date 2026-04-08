# Intrusion Detection Using State-of-the-Art Machine Learning on Corrected Network Datasets


## Overview

This project investigates whether state-of-the-art foundation and deep learning models (TabPFN, TabICL) can improve upon traditional ML baselines (Random Forest, XGBoost) for network intrusion detection. Experiments cover both within-dataset classification and cross-dataset generalization using two corrected network flow datasets: **LycoS-IDS2017** and **LycoS-Unicas-IDS2018**.

## Repository Structure

```
├── datasets/       # Preprocessed network flow datasets (LycoS-IDS2017, LycoS-Unicas-IDS2018)
├── notebooks/      # Jupyter notebooks for model training, evaluation, cross-dataset experiments, and splitting raw LycoS-Unicas-IDS2018 files
├── results/        # Output metrics, confusion matrices, and figures
└── utils/          # Helper scripts for preprocessing, feature selection, and evaluation
```

## Datasets

Datasets are not included in this repository due to size. Download them from the official sources:

- **LycoS-IDS2017**: [https://lycos-ids.univ-lemans.fr/download-lycos-ids2017.html](https://lycos-ids.univ-lemans.fr/download-lycos-ids2017.html)
- **LycoS-Unicas-IDS2018**: https://github.com/MarcoCantone/LycoS-Unicas-IDS2018

Place the downloaded files inside the `datasets/` directory before running any notebooks. Then run `lycos2018_to_parquet_splits.ipynb` to create the splits for the LycoS-Unicas-IDS2018 dataset which is consistent with how it is done in LycoS-IDS2017.

## Environment Setup

### Prerequisites

- Python 3.10+
- [pip](https://pip.pypa.io/) or [conda](https://docs.conda.io/)

Key packages used:

| Package | Purpose |
|---|---|
| `scikit-learn` | Random Forest, preprocessing, evaluation |
| `xgboost` | XGBoost baseline |
| `tabpfn` | TabPFN foundation model |
| `tabicl` | TabICL foundation model |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Plotting |
| `jupyter` | Running notebooks |

### Launch notebooks

```bash
jupyter notebooks/COMP9150_RF_XGB_CrossDataset.ipynb
```

Then open the relevant notebook from the `notebooks/` directory.

## Models Evaluated

- **Baselines**: Random Forest, XGBoost
- **Foundation models**: TabPFN, TabICL