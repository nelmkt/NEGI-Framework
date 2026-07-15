# NEGI Framework: Remote Sensing and Machine Learning for Urban Greening Energy Assessment

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

An open-source implementation of the Net Energy Gain Index (NEGI) framework for evaluating urban greening under coupled water–energy–climate constraints.

---

## About

This repository accompanies the manuscript:

> **A Remote Sensing and Machine Learning Framework for Evaluating Urban Greening–Energy Trade-Offs in Desalination-Dependent Cities**

![Graphical Abstract](figures/graphica1_abstract.jpg)

The NEGI Framework integrates Google Earth Engine, satellite remote sensing, XGBoost machine learning, and comparative energy assessment to evaluate urban greening strategies in desalination-dependent cities. The repository contains the complete preprocessing scripts, Python implementation, processed datasets, and reproducible workflow used in the accompanying study.

---

# Key Results

- XGBoost surrogate model achieved R² = 0.817 (RMSE = 3.64 °C).
- Introduces the Net Energy Gain Index (NEGI) for comparative urban greening assessment.
- Identifies a comparative energy-optimal vegetation threshold of approximately 5% FVC.
- Identifies a comparative energy-neutral threshold of approximately 15–20% FVC.
- Demonstrates robust vegetation thresholds across all evaluated sensitivity scenarios.

---

# Features

- Landsat 8 preprocessing using Google Earth Engine
- Extraction of NDVI, NDBI, and Land Surface Temperature (LST)
- XGBoost regression for urban thermal prediction
- Fractional Vegetation Cover (FVC) scenario analysis
- Comparative assessment of vegetation cooling and desalination-related irrigation energy
- Net Energy Gain Index (NEGI) computation
- Sensitivity analysis of empirical scaling coefficients
- Automatic generation of publication-ready figures and CSV outputs

---

# Methodological Workflow

1. Acquire Landsat 8 imagery using Google Earth Engine.
2. Derive NDVI, NDBI, and Land Surface Temperature (LST).
3. Train an XGBoost regression model using NDVI and NDBI.
4. Optimize model hyperparameters using GridSearchCV.
5. Evaluate model performance using R², RMSE, and five-fold cross-validation.
6. Simulate urban greening scenarios using Fractional Vegetation Cover (FVC).
7. Estimate vegetation-induced cooling benefits.
8. Estimate desalination-related irrigation energy requirements.
9. Compute the Net Energy Gain Index (NEGI).
10. Perform sensitivity analysis of empirical scaling coefficients.

---

# Repository Structure

```
NEGI-Framework/
│
├── src/
│   └── negi_framework.py
│
├── gee/
│   └── landsat_preprocessing.js
│
├── data/
│   ├── Jeddah_NDVI_NDBI_LST_dataset.csv
│   ├── Scenario_Results.csv
│   └── Sensitivity_Analysis.csv
│
├── figures/
│   └── graphica1_abstract.jpg
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── .gitignore
```

---

# Installation

The workflow requires Python 3.12 (or later).

Install the required packages using

```bash
pip install -r requirements.txt
```

---

# Running the Code

Execute the complete workflow using

```bash
python src/negi_framework.py
```

Running the script performs the complete analytical workflow, including:

- Data loading
- Hyperparameter optimization (GridSearchCV)
- Model training
- Model validation
- Feature importance analysis
- Urban greening scenario simulation
- NEGI computation
- Sensitivity analysis
- Figure generation
- Export of CSV result files

---

# Outputs

Running the workflow automatically generates:

- XGBoost model evaluation metrics
- Feature importance analysis
- Urban greening scenario results
- Net Energy Gain Index (NEGI)
- Sensitivity analysis results
- Publication-ready figures
- CSV files containing scenario and sensitivity results

---

# Data

Satellite observations were derived from Landsat 8 imagery processed using Google Earth Engine.

The processed dataset contains approximately 15,000 randomly sampled observations, including:

- Normalized Difference Vegetation Index (NDVI)
- Normalized Difference Built-up Index (NDBI)
- Land Surface Temperature (LST)

The processed dataset is included to enable complete reproducibility of the published analyses.

---

# Reproducibility

This repository contains the complete implementation used to reproduce the analyses, figures, tables, and scenario results presented in the accompanying manuscript.

Executing the supplied workflow using the included processed dataset reproduces the published model evaluation, scenario analysis, sensitivity analysis, and figures without requiring additional preprocessing.

---

# Citation

If this repository contributes to your research, please cite:

1. The accompanying journal article.

2. This software repository (automatically generated from the included `CITATION.cff` file).

---

# License

This project is released under the MIT License.

See the `LICENSE` file for details.

---

# Contact

**Nelly F. Almaktoum**

Faculty of Computing and Information Technology  
King Abdulaziz University  
Jeddah, Saudi Arabia

📧 **Email:** nalmaktoum0001@stu.kau.edu.sa

🔗 **ORCID:** https://orcid.org/0009-0007-9887-0280

---

## Acknowledgement

If this repository supports your research, please consider starring the repository and citing the accompanying publication.
