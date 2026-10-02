"""Record annual LST differences on cells present in both saved GEE exports."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "code/gee/panel.csv"
NEW = ROOT / "code/gee/panel_jun_sep.csv"
OUT = ROOT / "tables/gee_month_composite_changes.csv"


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    years = (2014, 2015, 2018, 2019, 2024, 2025)
    cols = ["lon", "lat"] + [f"lst_{year}" for year in years]
    old = pd.read_csv(OLD, usecols=cols)
    new = pd.read_csv(NEW, usecols=cols)
    both = old.merge(new, on=["lon", "lat"], validate="one_to_one", suffixes=("_old", "_new"))
    rows = []
    for year in years:
        difference = (both[f"lst_{year}_new"] - both[f"lst_{year}_old"]).dropna()
        rows.append(dict(year=year, n_shared_cells=len(both), n_with_both_lsts=len(difference),
                         n_equal_1e_minus_6=int(np.isclose(difference, 0, rtol=0, atol=1e-6).sum()),
                         mean_new_minus_old_C=float(difference.mean()),
                         median_new_minus_old_C=float(difference.median()),
                         max_abs_change_C=float(difference.abs().max())))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
