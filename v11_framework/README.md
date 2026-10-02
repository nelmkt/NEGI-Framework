# v11 framework: spatial validation and metric diagnostics

This folder holds the code, tables and paper for "A Remote Sensing and Machine Learning Framework for Evaluating Urban Greening–Energy Trade-Offs in Desalination-Dependent Cities: Spatial Validation and Metric Diagnostics in a Jeddah Case Study" (`manuscript/MANUSCRIPT_FRAMEWORK_ML_V11B_TEMPLATE_E.docx`).

It is self-contained. The folders `src/`, `gee/`, `Data/` and `Results/` at the top of the repository belong to the earlier pipeline run and are not used by the paper.

## What this is

1. **Satellite processing** (`code/gee`): Landsat 8 Collection 2 Level-2 composites exported from Google Earth Engine as a panel of 90 m cells for 2014, 2015, 2018, 2019, 2024 and 2025.
2. **Machine-learning model** (`code/src/fw_model.py`): XGBoost prediction of land surface temperature (LST), scored by spatial GroupKFold, with a feature-space support screen and spatial refits.
3. **Matched-contrast benchmark** (`code/src/fw_matching.py` and the `analyze_*` scripts): cells that greened compared with never-vegetated controls, zero-intercept per-pixel slopes, spatial block bootstrap, joint-ring regressions.
4. **Counterfactual gates** (`code/src/fw_validate.py`): model-predicted change is compared with measured change before any model-based scenario is released.
5. **Index module** (`code/negi_original/NEGI_Framework.py`): the Normalized Environmental Gain Index (NEGI) and its pathways. The paper uses its definitions and algebra; it reports no values from it.
6. **Hypothetical water and energy ledger** (`code/src/fw_ledger.py`, `fw_decision.py`).

## Scope

The aim is a remote sensing and machine learning framework for evaluating urban greening–energy trade-offs in desalination-dependent cities. Read that with these limits:

- Trade-offs are evaluated through illustrative scenarios with an assumed irrigation-energy cost. No irrigation volume, building energy use or desalination energy is measured.
- Jeddah is the only site the framework has been run on.
- Applicability to other desalination-dependent cities is a design intent that has not been tested. The estimates, the fitted model and the ledger must be recalibrated for another city and do not transfer. See `manuscript/PORTABILITY_AUDIT_V11.md`.
- LST is surface temperature. It is not air temperature, thermal comfort or population benefit.
- High skill in predicting LST does not show that predicted changes in LST are correct. Here the model had R² = 0.795 under spatial cross-validation, had feature-space support for 14 of 744 outside validation cells, and under-predicted measured cooling by 1.01 to 7.24 °C. No model-based scenario or NEGI value is reported for that reason.

## Layout

```
code/negi_original/NEGI_Framework.py   index module (not executed for the paper)
code/src/                              framework modules, analysis scripts, tests
code/gee/                              Earth Engine export scripts and the saved exports
tables/                                result tables (CSV, JSON)
tables_revision_v11/                   two algebraic checks added for the paper
figures_png/, figures_pdf/             figures
manuscript/                            the paper, its build and audit scripts, audit notes
README.md  REPRODUCE.md  CHANGELOG.md  requirements.txt  MANIFEST_SHA256.txt
```

Scripts locate this folder as two levels above their own file and read `code/gee/panel.csv` and `tables/` relative to it, so run every command from inside `v11_framework/`.

## Install

Python 3.12 was used.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From inside `v11_framework/` (this writes into `tables/` and the figure folders; use a copy to keep the shipped outputs):

```
python code/src/run_framework.py --panel code/gee/panel.csv --out .
```

Tests:

```
set PYTHONDONTWRITEBYTECODE=1
python -m pytest code/src -p no:cacheprovider
```

These commands were not run when this release was prepared. The author reports 62 tests passing on the full package on 2026-10-02. `REPRODUCE.md` maps every number in the paper to its table, script and command.

## Withheld file and its effect on the tests

`tables/imagery_sampling_key_v7.csv` is withheld. It is the blinded key for an imagery review that is not complete. Two tests refer to it by name and are expected to fail without it: `code/src/test_followup_v7.py` (line 28, asserts the file exists) and `code/src/test_v8.py` (lines 87-109, asserts the table inventory, which lists the key). This is read from the test source; it was not run. The tests are unchanged.

`code/src/audit_claims.py` (lines 160-163) asserts an exact folder layout that this release does not have (`tables_revision_v11/`, `code/negi_original/` and root files are extra), so that assertion is expected to fail here.

## Not included

- The paper template and the earlier manuscripts that the build script reads for reference lists and two figures.
- The blinded imagery key.

## Licence and citation

The repository's `LICENSE` and `CITATION.cff` at the top level apply. [AUTHOR TO COMPLETE: confirm, and add a DOI after release.]
