### A Machine Learning Framework for Evaluating Urban Greening Under Water–Energy Constraints

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

An open-source implementation of the Normalized Environmental Gain Index (NEGI) framework for evaluating urban greening strategies under coupled cooling–energy trade-offs in desalination-dependent cities.

---

# Overview

This repository accompanies the manuscript:

> **A Decision-Support Framework for Evaluating Urban Greening under Water–Energy Constraints in Desalination-Dependent Cities**

The framework combines:

* Google Earth Engine
* Landsat 8 remote sensing
* Machine learning (XGBoost)
* Spatial cross-validation
* Calibration analysis
* Scenario optimization
* Energy assessment
* Uncertainty quantification
* Sensitivity analysis

to evaluate urban greening strategies while accounting for the energy required to produce irrigation water through desalination.

---

## Framework Highlights

* Google Earth Engine preprocessing
* Landsat-derived NDVI, NDBI, Elevation, and LST
* RandomizedSearchCV hyperparameter optimization
* Nested GroupKFold spatial cross-validation
* Repeated spatial-block validation
* Independent confirmatory spatial holdout
* Model calibration assessment
* Bootstrap uncertainty analysis
* Permutation feature importance
* Empirical NDVI–NDBI scenario generation
* Normalized Environmental Gain Index (NEGI)
* Response surface analysis
* Feature-space support diagnostics
* Parameter sensitivity analysis
* Publication-quality figures
* Fully reproducible analytical pipeline

---

# Methodological Workflow

The framework performs the following steps:

1. Preprocess Landsat imagery using Google Earth Engine.
2. Extract NDVI, NDBI, Elevation, and Land Surface Temperature.
3. Construct the modeling dataset.
4. Optimize XGBoost using RandomizedSearchCV.
5. Benchmark against Linear Regression, Random Forest, and Gradient Boosting.
6. Evaluate performance using:

   * Nested GroupKFold
   * Repeated spatial-block validation
   * Independent spatial holdout
7. Assess calibration.
8. Compute permutation feature importance.
9. Generate empirical urban-greening scenarios.
10. Predict land surface cooling.
11. Estimate desalination-related irrigation energy.
12. Compute the Normalized Environmental Gain Index (NEGI).
13. Quantify uncertainty using bootstrap analysis.
14. Evaluate feature-space support.
15. Perform sensitivity analyses.
16. Export publication-ready figures, tables, and reports.

---

# Repository Structure

```text
NEGI-Framework/

├── NEGI_Framework.py
├── gee/
│   └── landsat_preprocessing.js
│
├── Data/
│   └── processed_dataset.csv
│
├── Results/
│   ├── Main/
│   ├── Supplementary/
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── .gitignore
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/NEGI-Framework.git
cd NEGI-Framework
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Framework

Execute the complete workflow:

```bash
python NEGI_Framework.py
```

The pipeline automatically performs:

* dataset preparation
* model optimization
* spatial validation
* calibration analysis
* residual diagnostics
* feature importance analysis
* scenario simulation
* NEGI computation
* uncertainty analysis
* sensitivity analysis
* figure generation
* report generation
* CSV export

---

# Outputs

The framework automatically exports:

## Figures

* Main manuscript figures
* Supplementary figures
* PNG format
* PDF format

## Tables

* Validation metrics
* Calibration statistics
* Feature importance
* Scenario results
* Sensitivity analyses
* Support diagnostics

## Reports

* Full analytical summary
* Reproducibility report
* QA report
* Pipeline log

---

# Dataset

The processed dataset contains:

* Land Surface Temperature (LST)
* Normalized Difference Vegetation Index (NDVI)
* Normalized Difference Built-up Index (NDBI)
* Elevation

The processed dataset is included to ensure complete reproducibility without rerunning the remote sensing workflow.

---

# Reproducibility

The framework is fully reproducible.

Execution reproduces:

* RandomizedSearchCV optimization
* Nested GroupKFold validation
* Repeated spatial-block validation
* Independent spatial holdout
* Calibration assessment
* Bootstrap uncertainty
* Permutation feature importance
* Scenario optimization
* NEGI computation
* Sensitivity analyses
* Publication figures
* CSV tables
* Summary reports

---

# Citation

If you use this framework in your research, please cite:

1. The accompanying journal article.
2. This software repository via the included `CITATION.cff` file.

---

# License

This project is released under the MIT License.

See the `LICENSE` file for details.

---

# Author

**Nelly F. Almaktoum**

Faculty of Computing and Information Technology (FCIT)
King Abdulaziz University
Jeddah, Saudi Arabia

📧 [nalmaktoum0001@stu.kau.edu.sa](mailto:nalmaktoum0001@stu.kau.edu.sa)

ORCID: [https://orcid.org/0009-0007-9887-0280](https://orcid.org/0009-0007-9887-0280)

---

# Acknowledgements

If this framework contributes to your research, please consider:

* Starring the repository
*  Citing the accompanying publication
*  Sharing improvements through issues or pull requests
