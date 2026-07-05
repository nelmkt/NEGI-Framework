# NEGI Framework: Remote Sensing and Machine Learning for Urban Greening Energy Assessment

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

**NEGI Framework** is an open-source Python workflow accompanying the manuscript:

> **A Remote Sensing and Machine Learning Framework for Evaluating Urban Greening–Energy Trade-Offs in Desalination-Dependent Cities**

![Graphical Abstract](figures/graphica1_abstract.jpg)

The framework evaluates the energy implications of urban greening in desalination-dependent cities by integrating satellite remote sensing, machine learning, and comparative energy assessment. Using **Jeddah, Saudi Arabia**, as a case study, the workflow combines Landsat 8 observations, XGBoost regression, and scenario-based analysis to identify vegetation levels that maximize relative net energy performance.

---

# Features

* Landsat 8 preprocessing using Google Earth Engine
* Extraction of NDVI, NDBI, and Land Surface Temperature (LST)
* XGBoost regression for urban thermal prediction
* Fractional Vegetation Cover (FVC) scenario analysis
* Comparative assessment of vegetation cooling and desalination-related irrigation energy
* Net Energy Gain Index (NEGI) computation
* Sensitivity analysis of empirical scaling coefficients
* Automatic generation of publication-ready figures and CSV outputs

---

# Methodological Workflow

1. Acquire Landsat 8 imagery using Google Earth Engine.
2. Derive NDVI, NDBI, and Land Surface Temperature (LST).
3. Train an XGBoost regression model using NDVI and NDBI.
4. Evaluate model performance using R², RMSE, and 5-fold cross-validation.
5. Simulate urban greening scenarios using Fractional Vegetation Cover (FVC).
6. Estimate vegetation-induced cooling benefits.
7. Estimate desalination-related irrigation energy requirements.
8. Compute the Net Energy Gain Index (NEGI).
9. Identify vegetation thresholds that maximize relative net energy performance.
10. Perform sensitivity analysis of the empirical scaling coefficients.

---

# Repository Structure

```
NEGI-Framework/
│
├── gee/
│   └── landsat_preprocessing.js
│
├── data/
│   ├── Jeddah_NDVI_NDBI_LST_dataset.csv
│   ├── Scenario_Results.csv
│   └── Sensitivity_Analysis.csv
│
├── src/
│   └── negi_framework.py
│
├── figures/
│   └── graphical_abstract.jpg
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── .gitignore
```

---

# Installation

The workflow requires **Python 3.10** (or later).

Install all required packages using

```bash
pip install -r requirements.txt
```

---

# Running the Code

Execute the complete workflow using

```bash
python src/negi_framework.py
```

The script automatically performs:

* Data loading
* Model training
* Model validation
* Feature importance analysis
* Urban greening scenario simulation
* NEGI computation
* Sensitivity analysis
* Figure generation
* Export of CSV result files

---

# Outputs

Running the workflow automatically generates:

* XGBoost model evaluation metrics
* Feature importance analysis
* Urban greening scenario results
* Net Energy Gain Index (NEGI)
* Sensitivity analysis results
* Publication-ready figures
* CSV files containing scenario and sensitivity results

---

# Data

Satellite observations were derived from **Landsat 8** imagery processed using **Google Earth Engine**.

The processed dataset contains approximately **15,000 randomly sampled observations**, including:

* Normalized Difference Vegetation Index (NDVI)
* Normalized Difference Built-up Index (NDBI)
* Land Surface Temperature (LST)

---

# Reproducibility

This repository contains the complete implementation used to generate the analyses, figures, tables, and scenario results presented in the accompanying manuscript.

Running the supplied workflow with the included processed dataset reproduces the reported model evaluation, scenario analysis, sensitivity analysis, and publication figures.

---

# Citation

If you use this repository, please cite this software repository and the accompanying journal article.

GitHub will automatically generate a citation from the included **CITATION.cff** file.

---

# License

This project is released under the **MIT License**.

See the **LICENSE** file for details.

---

# Contact

**Nelly F. Almaktoum**

King Abdulaziz University

📧 **Email:** [nalmaktoum0001@stu.kau.edu.sa](mailto:nalmaktoum0001@stu.kau.edu.sa)

🔗 **ORCID:** [https://orcid.org/0009-0007-9887-0280](https://orcid.org/0009-0007-9887-0280)

