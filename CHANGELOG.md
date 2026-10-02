# Changelog

## 2.0.1 (2 October 2026), branch `v11-framework-ml`, release `v2.0.1`

- Spelling of the framework name corrected to Wahj in every file; the repository is renamed `Wahj-Framework`.
- `docs/Wahaj_Manuscript_numeric_audit.md` replaced by `docs/Wahj_Manuscript_numeric_audit.md`.
- No code, table or figure changed.

## 2.0.0 (2 October 2026), branch `v11-framework-ml`, release `v2.0.0`

Reorganised the repository around the framework described in the paper. The framework is named Wahj and the repository `Wahj-Framework`; NEGI remains the name of the index.

Added

- `code/gee`: Earth Engine export scripts and the saved cell panels (the two large panels are stored as `.gz`).
- `code/src`: framework modules (`fw_*.py`), matched-contrast and exposure analyses, figure scripts, tests.
- `code/negi_original/NEGI_Framework.py`: the index module.
- `tables/`, `tables_revision_v11/`, `figures_png/`, `figures_pdf/`: results.
- `docs/`: file guide, data notes, math check, NEGI logic audit, portability audit, numeric audit of the paper (`Wahj_Manuscript_numeric_audit.md`).
- `REPRODUCE.md`, `MANIFEST_SHA256.txt`, new `README.md`, `CITATION.cff`, `requirements.txt`.

Removed from this branch (still on `main`)

- `src/`, `gee/`, `Data/`, `Results/`: the first version of the pipeline and its July 2026 outputs. Its scenario and NEGI values are withdrawn because the surrogate behind them was not tested against measured change.

Known gaps

- `tables/imagery_sampling_key_v7.csv` is withheld (blinded review key); two tests and one layout assertion are expected to fail for that reason and because of the added folders.
- The paper itself is not in the repository.
- The paper reference and DOI will be added to `README.md` and `CITATION.cff` after publication.

## 1.0.0 (July 2026), branch `main`

First public version: single-module surrogate pipeline with scenario analysis and NEGI.
