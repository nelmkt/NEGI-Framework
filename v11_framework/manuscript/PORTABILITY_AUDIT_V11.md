# PORTABILITY_AUDIT_V11 (Section B9)

Written 2026-10-02. Read-only inspection of both code bases. No code was modified. Every line number below was read or grepped this session. An earlier, wider inventory of the validation package exists as `manuscript/PORTABILITY_CODE_AUDIT_V11.md` (not written by this session, and its scope excludes `code/negi_original`); where its line numbers differ from the ones here, the ones here are what this session saw.

The framework has been run on Jeddah only. "Change" means replace or re-justify before using the workflow for another city. It does not mean Jeddah's values carry over.

## A. Validation package (`code/src`, `code/gee`)

| Jeddah-specific element | File : line | In `fw_config.py`? | What a new city needs |
|---|---|---|---|
| Analysis rectangle `RECT = [39.0, 21.2, 39.4, 21.8]` | `code/gee/export_panel.py:23`; `code/gee/export_panel.js:7`; `code/gee/landsat_scenes.py:308` | no | new extent, in three separate places |
| Country land polygon `country_na == "Saudi Arabia"` | `code/gee/export_panel.py:48`; `code/gee/export_panel.js:12` | no | new land or domain polygon |
| Years `(2014, 2015, 2018, 2019, 2024, 2025)` and months May-September | `code/gee/export_panel.py:21-22`; `code/gee/export_panel.js:6`; `code/src/fw_config.py:17-19`; `code/gee/gee_monthly_isolation.py:24-25` | partly (analysis side only) | periods chosen for local treatment history and season; keep exporter and config in step |
| Sensor `LANDSAT/LC08/C02/T1_L2`, scene `CLOUD_COVER < 20` | `code/gee/export_panel.py:49`; `code/gee/export_panel.js:13`; `code/gee/export_pixel_centered_isolation.py:31-33`; `code/gee/export_external_pixel_counts_v8.py:228-230`; `code/gee/export_external_pixel_subrings_v8_followup5.py:243-245`; `code/gee/gee_monthly_isolation.py:32-34`; `code/gee/landsat_scenes.py:42` | no | check coverage and cloud rule locally |
| Water exclusion buffer 300 m | `code/gee/export_panel.py:79` | no | local shoreline and water bodies |
| Cell size `CELL_M = 90`; pixel area 0.09 ha; nine-pixel cell | `code/gee/export_panel.py:24`; `code/gee/export_panel.js:7`; `code/src/fw_panel.py:15-16` | no | rebuild if grid or projection differs |
| NDVI thresholds: bare `< 0.15`, greened `>= 0.30` | `code/gee/export_panel.py:82,85,108-109`; `code/gee/export_panel.js:38-39,52-53`; `code/gee/gee_monthly_isolation.py:26` | no | re-examine against local soil and vegetation |
| Ring distance 150 m; donor distance 300 m; control fraction 0.10 | `code/gee/export_panel.py:25,122-123,127`; `code/gee/export_panel.js:7` | no | re-justify against the local thermal footprint and donor supply |
| Coastline file `jeddah_coastline_ne10m.csv` | `code/src/fw_panel.py:13`; `code/src/analyze_isolation_reweighting_v8.py:22` | no | new coastline, or drop the coastal feature for an inland city |
| Distance approximation 111.32 cos(mean latitude) and 110.57 km per degree | `code/src/fw_panel.py:20-22`; `code/src/analyze_isolation_reweighting_v8.py:35-37` | no | check adequacy at the new latitude |
| Built-up split `builtup_min = 0.05`; region grid `region_deg = 0.10`; coast bins (2, 5, 10) km; own-cover bin 0.05; emissivity bin 0.96; `min_controls = 3`; stable-surface limit 0.20 | `code/src/fw_config.py:22,26-32`; applied at `code/src/fw_panel.py:71,97` | yes | reassess overlap and balance |
| Dose classes `((1, 2), (3, 4), (5, 8), (9, 9))` | `code/src/fw_config.py:35` | yes | tied to the nine-pixel cell |
| Bootstrap block `block_deg = 0.05`; 2,000 draws | `code/src/fw_config.py:39-40`; `code/src/fw_panel.py:76` | yes | choose from local spatial dependence |
| Block identifiers such as `786_426` | computed, not literal: `code/src/fw_panel.py:83` (`_grid_id`), `code/src/analyze_isolation_reweighting_v8.py:42-44`. Grep for `786_426` in `code/src/*.py` and `code/gee/*`: no match | n/a | the label is floor(lon/deg)_floor(lat/deg); a new city yields new labels, and the dominant block must be found again |
| Southern-belt proxy `arc_lat_bounds = (21.3, 21.4)` | `code/src/fw_config.py:44`; used at `code/src/fw_matching.py:124,172`, `code/src/analyze_isolation_reweighting_v8.py:63`, `code/src/reviewer_diagnostics.py:59` | yes | do not reuse; any analogue needs its own definition |
| Model features `ndvi, ndbi, elev, emis, coast_km, lon, lat`; XGBoost settings; 5 folds; 200 refits; support k = 5 and 95th percentile; 300 m strict exclusion; no default tolerance | `code/src/fw_config.py:51-64` | yes | refit locally; `lon`, `lat` and `coast_km` bind the fitted model to Jeddah |
| Irrigation and energy assumptions: crop coefficients, efficiencies, kWh per cubic metre bands; Saudi Irrigation Organization file | `code/src/fw_config.py:84-90,96-108` | yes | local, verified water and energy data |
| Expected count of 1,333 classified cells | `code/gee/export_external_pixel_counts_v8.py:169,210`; `code/gee/export_external_pixel_subrings_v8_followup5.py:172,225`; `code/src/analyze_isolation_reweighting_v8.py:77`; `code/src/analyze_pixel_subrings_v8_followup6.py:58`; `code/src/analyze_combined_exposure_v8.py:62-83`; `code/src/write_subring_results_v8_followup7.py:26` | no | these are checks for the saved Jeddah run; replace with the new count, do not delete the check |
| Expected populations 1,006 / 419 / 32 blocks; isolated counts 76 / 38 / 26 | `code/src/analyze_exposure_rings_v8_followup3.py:64,104`; `code/src/analyze_pixel_subrings_v8_followup6.py:93` | no | same as above |
| Projection `EPSG:32637` (UTM 37N) | `code/gee/export_external_pixel_counts_v8.py:234-235`; `code/gee/export_external_pixel_subrings_v8_followup5.py:249-250` | no | new UTM zone |
| Jeddah file names and paths: `Jeddah_points_for_gee.csv`, `Jeddah_landcover.csv`, `C:\Users\Nelly\Downloads\Jeddah_LST_Dataset_2023_raw.csv`, asset `users/YOUR_USER/Jeddah_LST_Dataset_2023_raw`, Drive folder `negi`, project id `satttt-500210` | `code/gee/fetch_landcover.py:27-28`; `code/gee/landsat_scenes.py:13,21,334,348`; `code/gee/attach_landcover.js:20,52`; `code/gee/export_external_pixel_counts_v8.py:4`; `code/gee/export_external_pixel_subrings_v8_followup5.py:4` | no | rename; the user path and project id are examples in docstrings |
| Report and docstring titles naming Jeddah | `code/src/run_framework.py:1`; `code/src/fw_report.py:38` | no | rewrite |
| Default panel path `code/gee/panel.csv` and saved raw exports | `code/src/fw_config.py:12`; `code/src/analyze_isolation_reweighting_v8.py:20-21` | partly | re-export under new names |

Does `fw_config.py` centralise them? **Partly.** It holds the analysis-side periods, matching, block, model, support and resource assumptions. It does not hold the Earth Engine rectangle, country, grid, thresholds, donor distances or control fraction, the coastline file, the projection, the pinned population counts, the user-specific example paths, or report text.

## B. Original NEGI pipeline (`code/negi_original/NEGI_Framework.py`)

| Jeddah-specific element | Line | In `Config`? | What a new city needs |
|---|---|---|---|
| Dataset file names `jeddah_lst_data.csv`, `Jeddah_LST_Dataset_2023_raw.csv`, `Jeddah_LST_Dataset_2023_v4_with_purity_flags.csv`, `Jeddah_LST_Dataset_2023.csv` | 298-305 | via `data_path` default (320) and the `NEGI_DATA_PATH` environment variable (278-280) | set `NEGI_DATA_PATH`; the default search list is Jeddah-only |
| Search in `Downloads` and `Desktop` folders beside the script's parent | 299-300, 302-305 | no | remove the dependence on a personal folder layout; pass the path explicitly |
| Required columns `NDVI, NDBI, Elevation, ST_EMIS, ST_EMSD, LST, block_id` | 206, 4971-4976 | `FEATURES` is module-level (206); `feature_names` mirrors it (557) | export the same columns |
| Purity flag and its origin script name `Jeddah_LST_Extraction_v4_purity_...` | 360-398 (flag setting), 368 (script name) | yes (`vegetation_purity_filter`, 394-398) | recompute the flag from local imagery |
| Scenario definitions: Scenario 2 "realistic municipal greening intervention for Jeddah"; archetype percentile 0.15 "representing Jeddah's existing lower-density, more vegetated residential/park districts" | 644-708 (comments and fields); 705 (`s2_archetype_ndbi_percentile`); 8063-8136; 8224-8288 | yes (705-708) | the percentile is an urban-form judgement for Jeddah; choose and justify a local one |
| NDVI target percentile 0.95; scenario step 0.0025; 40 trajectory bins | 615, 614, 708 | yes | review against the local NDVI distribution |
| Spatial block 0.09 degrees "(~10km at Jeddah's latitude/longitude)", chosen from the correlogram | 825-831 | yes (831) | re-derive from the local correlogram |
| Correlogram range 20 km | 849 | yes | review for city extent |
| NEGI weights and cost: alpha 1.0, beta 1.0, w0 200.0, intensity 3.5, exponents 0.5 and 1.0 | 745-750 | yes | uncalibrated defaults; not Jeddah measurements (stated at 9039-9042, 11602) |
| Interpretive text naming Jeddah in logs, summaries and figure notes | 2517, 2608, 8112, 14947, 15021, 15044, 16002 | no | rewrite |

Does `Config` centralise them? **Mostly for parameters, not for inputs.** Numeric scenario, validation and NEGI parameters are fields of one dataclass (311 onward). The dataset search list, the personal-folder search and the Jeddah wording are outside it.

## C. Required inputs for another city

1. Landsat Collection 2 Level-2 surface temperature and reflectance for the chosen periods, giving LST, NDVI and NDBI.
2. A land boundary and, for a coastal city, a coastline.
3. A built-up surface layer that predates the greening (GHSL 2015 here) and a land-cover layer.
4. Elevation.
5. For any decision use: verified irrigation, water-source and energy data. The package has none for Jeddah.

## D. What must be recalibrated

Analysis extent and periods; NDVI thresholds; ring and donor distances; matching strata and calipers; the belt or any sub-region definition; block sizes; model features, hyperparameters and support screen; scenario percentiles; cost shape and weights; every pinned count and expected value in the checks and tests.

## E. What must not be transferred

The Jeddah slopes and intervals, both fitted XGBoost models, the NEGI values, and the hypothetical water and energy ledger.
