# Changelog

## v11-framework-ml (2026-10-02)

Added, all under `v11_framework/`:

- `manuscript/MANUSCRIPT_FRAMEWORK_ML_V11B_TEMPLATE_E.docx`: the paper. It reports the v11 results only: the spatially validated XGBoost model, the matched-contrast benchmark, the counterfactual gates, the estimand diagnostics and a hypothetical water and energy ledger. NEGI is defined with its equations and limits; no NEGI value is reported.
- `manuscript/build_manuscript_framework_v11b.py`, `audit_numbers_framework_v11b.py`, `NUMERIC_REGISTRY_FRAMEWORK_V11B.json`, `NUMERIC_AUDIT_FRAMEWORK_V11B.md`: the paper is built from the tables, and every number is checked against its source.
- `manuscript/MATH_CHECK_V11.md`, `NEGI_LOGIC_AUDIT_V11.md`, `PORTABILITY_AUDIT_V11.md`.
- `tables_revision_v11/leverage_share_check_v11.csv`, `tables_revision_v11/own_only_vs_joint_identity_v11.csv`.
- `code/src`, `code/gee`, `tables`, `figures_png`, `figures_pdf`: the framework code, saved exports and results.
- `code/negi_original/NEGI_Framework.py`: the index module.
- `README.md`, `REPRODUCE.md`, `requirements.txt`, `MANIFEST_SHA256.txt`.

Not changed: everything outside `v11_framework/`.

Known gaps:

- `tables/imagery_sampling_key_v7.csv` is withheld; two tests are expected to fail without it.
- Scenario and index values from the earlier pipeline run (top-level `Data/` and `Results/`) are not used by the paper and are withdrawn there.
- Several author statements in the paper are placeholders.
