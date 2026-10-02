# Changelog

## 2.1.0 (2 October 2026), branch `v11-framework-ml`, release `v2.1.0`

Added

- `tables_revision_v11/model_benchmark_v11.csv` and `code/src/benchmark_models_v11.py`: XGBoost, random forest, gradient boosting and linear regression scored on the same cells, predictors and spatial folds. The XGBoost row reproduces `tables/model_cv.csv`.
- `tables_revision_v11/tradeoff_illustration_v11.csv` and `code/src/tradeoff_illustration_v11.py`: NEGI evaluated on measured contrasts for illustration, and the hypothetical irrigation energy per degree of measured cooling.

Changed

- Root `requirements.txt` records pandas 3.0.3, the version in the author's environment.
- `docs/Wahaj_Manuscript_numeric_audit.md` is the audit of the paper that includes the two additions.

No existing code, table or figure changed.

## 2.0.0 (2 October 2026), branch `v11-framework-ml`, release `v2.0.0`

Reorganised the repository around the framework described in the paper. The framework is named Wahaj and the repository `Wahaj-Framework`; NEGI remains the name of the index.

Added

- `code/gee`: Earth Engine export scripts and the saved cell panels (the two large panels are stored as `.gz`).
- `code/src`: framework modules (`fw_*.py`), matched-contrast and exposure analyses, figure scripts, tests.
- `code/negi_original/NEGI_Framework.py`: the index module.
- `tables/`, `tables_revision_v11/`, `figures_png/`, `figures_pdf/`: results.
- `docs/`: file guide, data notes, math check, NEGI logic audit, portability audit, numeric audit of the paper (`Wahaj_Manuscript_numeric_audit.md`).
- `REPRODUCE.md`, `MANIFEST_SHA256.txt`, new `README.md`, `CITATION.cff`, `requirements.txt`.

Removed from this branch (still on `main`)

- `src/`, `gee/`, `Data/`, `Results/`: the first version of the pipeline and its July 2026 outputs. Its scenario and NEGI values are withdrawn because the surrogate behind them was not tested against measured change.

Known gaps

- `tables/imagery_sampling_key_v7.csv` is withheld (blinded review key); two tests and one layout assertion are expected to fail for that reason and because of the added folders.
- The paper itself is not in the repository.
- The paper reference and DOI will be added to `README.md` and `CITATION.cff` after publication.

## 1.0.0 (July 2026), branch `main`

First public version: single-module surrogate pipeline with scenario analysis and NEGI.
