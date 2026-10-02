# A Remote Sensing and Machine Learning Framework for Urban Greening–Energy Trade-Offs

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-007EC6?style=flat)](LICENSE)
[![Email](https://img.shields.io/badge/Email-Contact%20Me-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:nalmaktoum0001@stu.kau.edu.sa)

Code, data exports and result tables for the paper

> **A Remote Sensing and Machine Learning Framework for Evaluating Urban Greening–Energy Trade-Offs in Desalination-Dependent Cities: Spatial Validation and Metric Diagnostics in a Jeddah Case Study**

> Nelly F. Almaktoum

The framework asks a computational question: when can a machine-learning model of land surface temperature (LST) be trusted to say what greening would do, and what does that imply for comparing cooling with the energy cost of irrigation in a desalination-dependent city? Jeddah, Saudi Arabia, is the case study.

## Main results

| Result | Value | Source table |
|---|---|---|
| XGBoost skill for LST, spatial GroupKFold | R² = 0.795, RMSE = 1.184 °C, MAE = 0.903 °C on 19,650 non-greened cells | `tables/model_cv.csv` |
| Measured matched contrast, outside the built-up area | −1.181 °C per greened pixel on 1,006 cells | `tables/headline_block_v8.csv` |
| Leverage of the largest spatial block | 55.7% of the pooled dose-squared leverage | `tables/headline_block_v8.csv` |
| Own-only slope, cells with 1–2 greened pixels | −1.925 °C per pixel; equals a joint own coefficient near −1.0 plus co-varying neighbour greening (exact identity) | `tables_revision_v11/own_only_vs_joint_identity_v11.csv` |
| Feature-space support of the model on validation cells | 14 of 744 outside cells; 0 of 75 fully greened cells | `tables/test_support_counts.csv` |
| Model minus measured cooling, outside dose classes | +1.01 to +7.24 °C (model under-predicts) | `tables/test_by_dose.csv` |

The model predicts LST well and still fails as a predictor of change. The framework's gates therefore released no model-based scenario, and no value of the Normalized Environmental Gain Index (NEGI) is reported.

## Scope and limits

- Trade-offs are evaluated through illustrative scenarios with an assumed irrigation-energy cost. No irrigation volume, building energy use or desalination energy is measured.
- Jeddah is the only site the framework has been run on. Use in other desalination-dependent cities is a design intent that has not been tested; estimates, the fitted model and the ledger must be recalibrated and do not transfer. See `docs/PORTABILITY_AUDIT_V11.md`.
- LST is surface temperature. It is not air temperature, thermal comfort or population benefit.
- Greening is a spectral transition in NDVI, not verified irrigation.
- Interval coverage of the spatial block bootstrap is unproven.

## Repository structure

```text
README.md            this file
LICENSE              licence
CITATION.cff         how to cite
CHANGELOG.md         what changed and when
REPRODUCE.md         every number in the paper -> table, script, command
requirements.txt     Python packages
MANIFEST_SHA256.txt  checksum and size of every file

code/
  gee/               1. Earth Engine export scripts and the saved exports (inputs)
  src/               2. framework modules, analysis scripts, tests
  negi_original/     3. index module (NEGI definitions; not run for the paper)

tables/              result tables (CSV, JSON)
tables_revision_v11/ two algebraic checks added for the paper
figures_png/         figures (PNG)
figures_pdf/         figures (PDF)

docs/
  FILE_GUIDE.md      every script and table, grouped by purpose, in reading order
  DATA.md            data sources, panel columns, what is withheld
  MATH_CHECK_V11.md  leverage shares and the own-only vs joint identity
  NEGI_LOGIC_AUDIT_V11.md  what the index computes, line by line
  PORTABILITY_AUDIT_V11.md  what is Jeddah-specific in the code
  NUMERIC_AUDIT_FRAMEWORK_V11B.md  each number in the paper checked against its table
```

Start with `docs/FILE_GUIDE.md` to find a script or table.

## Workflow

| Stage | What happens | Where |
|---|---|---|
| 1. Satellite processing | Landsat 8 Collection 2 Level-2 composites exported from Google Earth Engine as a panel of 90 m cells (2014, 2015, 2018, 2019, 2024, 2025) | `code/gee/export_panel.py` |
| 2. Machine-learning model | XGBoost prediction of LST, spatial GroupKFold, spatial refits | `code/src/fw_model.py` |
| 3. Matched-contrast benchmark | greened cells against never-vegetated controls; per-pixel slopes; spatial block bootstrap; joint-ring regressions | `code/src/fw_matching.py`, `analyze_*.py` |
| 4. Counterfactual gates | feature-space support screen; model-predicted against measured change | `code/src/fw_validate.py` |
| 5. Pathways and index | evaluated only if the gates are passed | `code/src/fw_decision.py`, `code/negi_original/` |
| 6. Water and energy ledger | hypothetical accounting per degree of measured cooling | `code/src/fw_ledger.py` |

## Install

Python 3.12 was used.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare the data

The two cell panels are stored compressed. Decompress them once, in place:

```bash
python -c "import gzip,shutil; [shutil.copyfileobj(gzip.open(f'code/gee/{n}.gz','rb'), open(f'code/gee/{n}','wb')) for n in ('panel.csv','panel_jun_sep.csv')]"
```

Their SHA256 values after decompression are at the top of `MANIFEST_SHA256.txt`.

## Run

From the repository root (this writes into `tables/` and the figure folders; work in a copy to keep the shipped outputs):

```bash
python code/src/run_framework.py --panel code/gee/panel.csv --out .
```

Follow-up analyses are separate scripts in `code/src`; `REPRODUCE.md` gives the command for each table.

## Tests

```bash
set PYTHONDONTWRITEBYTECODE=1
python -m pytest code/src -p no:cacheprovider
```

The author reports 62 tests passing on the full local package (2 October 2026). The commands in this README were not re-run on this copy of the repository. Three checks are expected to fail here, for layout reasons only:

- `code/src/test_followup_v7.py` (line 28) and `code/src/test_v8.py` (lines 87–109) look for `tables/imagery_sampling_key_v7.csv`, which is withheld (blinded imagery-review key).
- `code/src/audit_claims.py` (lines 160–163) asserts an exact folder layout; this repository has additional folders (`docs/`, `tables_revision_v11/`, `code/negi_original/`) and root files.

## Data availability

Landsat imagery is public through Google Earth Engine. The exports used here are in `code/gee/`; sources and column meanings are in `docs/DATA.md`. Water and energy figures in `tables/decision_*.csv`, `tables/water.csv` and `tables/energy.csv` are assumptions, not measurements.

## Earlier version

The `main` branch holds the first version of this project (a surrogate pipeline with scenario and NEGI values from July 2026). Those scenario and index values are withdrawn: the surrogate behind them was never tested against measured change. They are kept on `main` for the record and are not used by the paper.

## How to cite

See `CITATION.cff`.

## Licence

See `LICENSE`.

## Contact

**Nelly F. Almaktoum**

[nalmaktoum0001@stu.kau.edu.sa](mailto:nalmaktoum0001@stu.kau.edu.sa)
