"""Summarize saved May–September endpoint NDVI by month, dose and latitude belt.

The saved export contains no October–April months, so this cannot compare
warm-season vegetation with wetter-season vegetation or prove irrigation.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from fw_config import Config
import fw_panel as panel
from analyze_monthly_isolation import link_export

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tables" / "gee_monthly_ndvi_by_month_belt_v6.csv"


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    df, _ = panel.load(Config(panel_path=ROOT / "code/gee/panel.csv"))
    df = link_export(df, pd.read_csv(ROOT / "code/gee/monthly_isolation_raw_v4.csv"))
    df = df.loc[df.group.eq("greened")].copy()
    df["dose_class"] = pd.cut(df.n_greened_px, bins=[0, 2, 4, 8, 9],
                              labels=["1–2", "3–4", "5–8", "9/9"])
    df["southern_belt"] = np.where(df.lat.ge(21.3) & df.lat.lt(21.4), "yes", "no")
    rows = []
    for (setting, dose, belt), group in df.groupby(["setting", "dose_class", "southern_belt"],
                                                   observed=True):
        for year in (2024, 2025):
            for month in (5, 6, 7, 8, 9):
                key = f"{year}{month:02d}"
                valid = group[f"ndvi_valid_green_count_{key}"].to_numpy(float)
                high = group[f"ndvi_ge30_green_count_{key}"].to_numpy(float)
                required = np.ceil(group.n_greened_px.to_numpy(float) / 2)
                adequate = valid + .25 >= required
                cell_share = np.divide(high, valid, out=np.full_like(high, np.nan), where=valid > 0)
                reportable = len(group) >= 10
                rows.append({
                    "setting": setting, "dose_class": str(dose), "southern_belt": belt,
                    "year": year, "month": month, "n_cells": len(group),
                    "n_adequately_observed": int(adequate.sum()), "reportable": reportable,
                    "fraction_observed_green_pixels_ndvi_ge_0_30":
                        float(high.sum() / valid.sum()) if reportable and valid.sum() else np.nan,
                    "median_cell_fraction_ndvi_ge_0_30":
                        float(np.nanmedian(cell_share[adequate])) if reportable and adequate.any() else np.nan,
                    "share_cells_ge80pct_green_pixels":
                        float(np.mean(cell_share[adequate] >= .8)) if reportable and adequate.any() else np.nan,
                })
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("saved",len(rows),"month-setting-dose-belt rows;")
    print("outside 1–2 warm-season monthly fractions:")
    print(pd.DataFrame(rows).query("setting == 'outside the built-up area' and dose_class == '1–2'")
          [["southern_belt", "year", "month", "n_cells",
            "fraction_observed_green_pixels_ndvi_ge_0_30"]].to_string(index=False))


if __name__ == "__main__":
    main()
