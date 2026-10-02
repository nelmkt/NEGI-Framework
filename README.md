# Remote sensing and machine learning for urban greening–energy trade-offs

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)

Code, saved Earth Engine exports, figures and result tables for the Jeddah case study accompanying **“A Remote Sensing and Machine Learning Framework for Evaluating Urban Greening–Energy Trade-Offs in Desalination-Dependent Cities: Spatial Validation and Metric Diagnostics in a Jeddah Case Study”** by Nelly F. Almaktoum.

The package asks whether an XGBoost model that predicts land surface temperature (LST) can also predict the *change* associated with greening. It compares model diagnostics with matched satellite contrasts, then illustrates the water and energy implied by assumed irrigation. **Jeddah is the only city analysed.** This is neither a measured irrigation ledger nor a validated model for selecting a new greening site.

## Results and their populations

| Result | Saved value | Source and limit |
| --- | --- | --- |
| XGBoost absolute-LST prediction | Spatial GroupKFold R² **0.795**, RMSE **1.184 °C**, MAE **0.903 °C**, on **19,650** non-greened cells | [`model_cv.csv`](tables/model_cv.csv), row 2. Skill on LST levels does **not** validate predicted greening effects. |
| Primary matched outside-area slope | **−1.181 °C per greened pixel** among **1,006** matched cells | [`headline_block_v8.csv`](tables/headline_block_v8.csv), row 2. A zero-intercept, dose-weighted summary in sampled, often contiguous blocks—not the marginal effect of a new isolated pixel. |
| Leverage concentration | One spatial block carries **55.7%** of pooled squared-dose leverage | Same table, row 2. Bootstrap and jackknife coverage is unproven. |
| Small-patch own-only slope | **−1.925 °C per own pixel** among **419** primary matched outside 1–2-pixel cells | Same table, row 3; [algebraic check](tables_revision_v11/own_only_vs_joint_identity_v11.csv). The own-only coefficient absorbs co-varying neighbour greening within the specified regression; it is not evidence of a larger physical effect per pixel. |
| Model B outside support | **14/744** cells pass the feature-space screen, including **0/75** fully greened cells | [`test_support_counts.csv`](tables/test_support_counts.csv), outside rows. This is the **narrower concurrent-change sensitivity**, not the 1,006-cell primary population. |
| Model B minus measured outside contrast | **+1.01 to +7.24 °C** across dose classes | [`test_by_dose.csv`](tables/test_by_dose.csv), outside Model B rows. The gap grows with dose; the model does not corroborate that outside gradient. |

The 1,006-cell matched estimate and 744-cell model diagnostics use different treated populations. They must not be read as one validation result. The small-patch exposure analyses are descriptive: the available data do not separate thermal-pixel blur, edge effects, neighbour cooling and neighbourhood confounding. The within-90 m ordering is **not resolved**.

## What the code contains

| Stage | Files | Role |
| --- | --- | --- |
| Satellite panel | [`code/gee/export_panel.py`](code/gee/export_panel.py), [`panel.csv`](code/gee/panel.csv), [`panel_meta.json`](code/gee/panel_meta.json) | Cloud-screened Landsat 8 Collection 2 Level-2 optical/LST composites on aligned 90 m cells for 2014, 2015, 2018, 2019, 2024 and 2025. [`panel_jun_sep.csv`](code/gee/panel_jun_sep.csv) is a common-month sensitivity. |
| Matched contrasts | [`fw_panel.py`](code/src/fw_panel.py), [`fw_matching.py`](code/src/fw_matching.py) | Pre-treatment-strata matched comparison with never-vegetated controls and spatial-block resampling; follow-up scripts inspect leverage, isolation and exposure. |
| ML diagnostic | [`fw_model.py`](code/src/fw_model.py), [`fw_validate.py`](code/src/fw_validate.py) | XGBoost absolute-LST prediction, spatial cross-validation, feature-support screening and effect comparison. Absolute-LST skill alone does not license a counterfactual effect. |
| Conditional resource accounting | [`fw_ledger.py`](code/src/fw_ledger.py), [`fw_decision.py`](code/src/fw_decision.py) | Reference ET, assumed landscape coefficient and irrigation efficiency, and assumed water-source energy intensities. The per-degree ratios are largely rescalings of cooling per area under shared irrigation depth—not a resource ranking or net-energy benefit. |
| Earlier index module | [`NEGI_Framework.py`](code/negi_original/NEGI_Framework.py) | Historical pathway/NEGI definitions. Its earlier scenario and index values are not reported as validated findings here. This module is separate from `fw_decision.py`. |

The full working package also contains `tables/`, `tables_revision_v11/`, `figures_png/`, `figures_pdf/` and `manuscript/`. The nested `github_release_v11*` folders are **publication staging copies**; their contents do not replace the source folders above. The two panels in this full package are already uncompressed. Do not run a decompression command over them.

## Interpretation and portability

- LST is **surface temperature**, not air temperature, thermal comfort, building electricity savings or population benefit. Satellite overpass cooling and annual irrigation have different time bases.
- NDVI-defined greening is **not verified managed irrigation**. Persistent vegetation may also draw on groundwater, wadi flow or discharge. No water-meter, planting-history or building-energy observations establish the resource ledger.
- The model's outside support is sparse, its dose-gradient error grows, and no practical effect-error tolerance was prespecified. This package presents no decision-ready model-based scenario or NEGI value.
- “Similar city” means a hot arid or semi-arid city with comparable optical and thermal satellite coverage, identifiable vegetation transitions, enough eligible bare controls and a relevant desalination or treated-wastewater supply context. That is a candidate for *adapting the workflow*, not for importing Jeddah's estimates. The seasonal windows, classification and matching thresholds, coastline, spatial blocks, ML model/support domain and water–energy assumptions all need local justification. **The Jeddah estimates, fitted model and hypothetical ledger do not transfer.**

## Reproduction and checks

Work from this package root **in a copy**: the main pipeline writes tables and figures, and several follow-up scripts refuse to overwrite saved outputs. Python 3.12 was used. The full source dependency list is [`code/src/requirements.txt`](code/src/requirements.txt); the [publication staging requirements](github_release_v11c/requirements.txt) additionally list Earth Engine and document-building packages and note a pandas-version discrepancy. Earth Engine exports require an authenticated project; the saved panels mean Earth Engine is not needed to inspect the current tables.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code/src/requirements.txt
python code/src/run_framework.py --panel code/gee/panel.csv --out .
```

The author reported **62 passing tests and a passing claims audit** on the then-current full v11 package on 2 October 2026. We did **not** rerun them while preparing this README, and those results do not verify the present expanded directory layout. The strict claims audit expects the earlier five-folder release root and may flag the nested publication staging folders. The blinded imagery key is present in this **local** working package but must remain unopened; it should be withheld from a public release. The [reproduction guide](github_release_v11c/REPRODUCE.md) records the commands and provenance for individual tables. No test result by itself establishes parallel trends, irrigation identity, interval coverage or a physical cooling mechanism.

## Citation, licence and contact

The staged [`CITATION.cff`](github_release_v11c/CITATION.cff) identifies the software and author; add the final paper reference and DOI only when available. **Licence needs reconciliation before publication:** the staged citation metadata says MIT, but the full package has no matching licence file at its root. Do not infer redistribution terms from the badge in an earlier README draft. Contact: [Nelly F. Almaktoum](mailto:nalmaktoum0001@stu.kau.edu.sa).
