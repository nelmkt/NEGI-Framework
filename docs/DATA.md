# Data notes

## Sources

| Layer | Earth Engine identifier | Used for | Where in the code |
|---|---|---|---|
| Landsat 8 Collection 2 Level-2 | `LANDSAT/LC08/C02/T1_L2`, scenes with `CLOUD_COVER < 20` | surface temperature (LST), NDVI, NDBI, emissivity | `code/gee/export_panel.py` line 49 |
| Country boundary | `USDOS/LSIB_SIMPLE/2017`, Saudi Arabia | land domain | line 47–48 |
| Dynamic World | `GOOGLE/DYNAMICWORLD/V1` | built-up probability, before and after | line 88 |
| SRTM elevation | `USGS/SRTMGL1_003` | elevation | line 103 |
| TerraClimate | `IDAHO_EPSCOR/TERRACLIMATE` | reference evapotranspiration for the ledger | line 171 |
| GHSL built-up surface 2015 | `JRC/GHSL/P2023A/GHS_BUILT_S` | building cover before greening | `code/src/fw_config.py` (SOURCES) |
| Coastline | Natural Earth 10 m | distance to coast | `code/src/jeddah_coastline_ne10m.csv` |
| Irrigation context | Saudi Irrigation Organization open data, 2020–2022 | context only; not Jeddah landscape use | `code/src/6.xlsx`, `8.xlsx`, `sio_irrigation_2020_2022_v8.csv` |

Analysis rectangle: longitude 39.0–39.4° E, latitude 21.2–21.8° N (`code/gee/export_panel.py` line 23). Composites are May–September means of monthly medians for 2014, 2015, 2018, 2019, 2024 and 2025 (lines 21–22).

## The cell panel

`code/gee/panel.csv` (stored as `panel.csv.gz`): one row per 90 m cell, each cell made of nine 30 m pixels. Export metadata is in `code/gee/panel_meta.json`.

| Column | Meaning |
|---|---|
| `lon`, `lat` | cell centre |
| `cls` | class: 1 greened; 2 never-vegetated ring cell within 150 m of greening; 3 never-vegetated control more than 300 m from greening (random 10% sample); 4 other land (random 10% sample) |
| `greened_frac` | share of the nine pixels that greened (bare in 2014 and 2015, NDVI at least 0.30 in 2024 and 2025) |
| `late_frac` | share that greened only after 2019 |
| `never_frac`, `pre_bare_frac`, `dry_frac` | shares of pixels never vegetated, bare at baseline, and land not within the water buffer |
| `dist_greened_m` | distance to the nearest greened pixel |
| `lst_YYYY`, `ndvi_YYYY`, `ndbi_YYYY`, `emis_YYYY` | summer composite LST (°C), NDVI, NDBI and emissivity for each exported year |
| `built_pre`, `built_post` | Dynamic World built-up probability of the cell, before and after |
| `nb_built_pre`, `nb_built_post` | the same for the surrounding neighbourhood |
| `ghsl_2015`, `ghsl_nb_2015` | GHSL building cover of the cell and of its surroundings in 2015 |
| `elev` | elevation (m) |
| `rnd` | random number used for the 10% samples |

Class counts in the export: 1,333 greened, 2,988 ring, 16,363 control, 4,892 other (`panel_meta.json`).

`code/gee/panel_jun_sep.csv` (stored as `.gz`) is the same panel built from June–September composites, used for the calendar sensitivity check.

## Other saved exports in `code/gee`

| File | What it is |
|---|---|
| `pixel_external_counts_v8_raw.csv` (+ `.json`) | number of greened 30 m pixels outside each greened cell within 90, 180 and 300 m |
| `pixel_external_subrings_v8_followup5_raw.csv` (+ `.json`) | the same at 30 and 60 m |
| `pixel_isolation_raw.csv` (+ `.json`) | pixel-centred isolation flags |
| `monthly_isolation_raw*.csv` (+ `.json`) | monthly NDVI persistence checks, several recipe versions |
| `Jeddah_points_for_gee.csv`, `Jeddah_landcover.csv` | point list and land-cover attributes used by auxiliary scripts |

Each `.json` beside an export records the recipe version, the hash of the panel it was built from and the result of its consistency checks.

## What is assumed, not measured

- Irrigation depth: reference evapotranspiration × landscape coefficient ÷ irrigation efficiency (`tables/water.csv`).
- Energy per cubic metre of water: literature ranges (`tables/energy.csv`).
- No irrigation record, water meter, building-energy or desalination-plant data is used anywhere.

## Withheld

`tables/imagery_sampling_key_v7.csv`: the blinded key of an imagery review that is not complete. The unlabelled review queue (`tables/imagery_review_queue_v7.csv`) is included.

## Checksums

`MANIFEST_SHA256.txt` lists the SHA256 and size of every file, and the hashes of the two panels after decompression.
