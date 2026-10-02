"""Describe which NDVI-defined greened cells change under June–September export.

The already-saved common-month match is not rerun here. This table only
decomposes the change in classified cells by geography and dose.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from fw_config import Config
import fw_panel as panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tables" / "gee_common_month_composition_v6.csv"


def dose_class(values: pd.Series) -> pd.Series:
    return pd.cut(values, bins=[0, 2, 4, 8, 9], labels=["1–2", "3–4", "5–8", "9/9"])


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    old, _ = panel.load(Config(panel_path=ROOT / "code/gee/panel.csv"))
    new, _ = panel.load(Config(panel_path=ROOT / "code/gee/panel_jun_sep.csv"))
    cols = ["lon", "lat", "setting", "group", "n_greened_px", "coast_km", "lst_pre", "block"]
    linked = old[cols].merge(new[cols], on=["lon", "lat"], how="outer", validate="one_to_one",
                                 suffixes=("_old", "_new"), indicator=True)
    linked["group_old"] = linked.group_old.fillna("absent")
    linked["group_new"] = linked.group_new.fillna("absent")
    linked["setting"] = linked.setting_old.fillna(linked.setting_new)
    linked["old_dose"] = dose_class(linked.n_greened_px_old).astype(str).replace("nan", "none")
    linked["new_dose"] = dose_class(linked.n_greened_px_new).astype(str).replace("nan", "none")
    linked["southern_belt"] = np.where(linked.lat.ge(21.3) & linked.lat.lt(21.4), "yes", "no")
    out = (linked.groupby(["setting", "group_old", "group_new", "old_dose", "new_dose", "southern_belt"],
                          dropna=False, observed=True).size().reset_index(name="n_cells"))
    assert int(out.n_cells.sum()) == len(linked)
    assert int((linked.group_old == "greened").sum()) == 1333
    assert int((linked.group_new == "greened").sum()) == 1273
    out.to_csv(OUT, index=False)
    print("classified old/new", int((linked.group_old == "greened").sum()),
          int((linked.group_new == "greened").sum()))
    print(linked.loc[linked.group_old.eq("greened") | linked.group_new.eq("greened")]
          .groupby(["group_old", "group_new"]).size().to_string())
    print("old-greened exits by setting, dose, belt")
    print(linked.loc[linked.group_old.eq("greened") & ~linked.group_new.eq("greened")]
          .groupby(["setting", "old_dose", "southern_belt"]).size().to_string())


if __name__ == "__main__":
    main()
