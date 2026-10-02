"""Cross-check own-cell, monthly, and nested-neighborhood counts in a GEE export."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MONTHS = [f"{year}{month:02d}" for year in (2024, 2025) for month in (5, 6, 7, 8, 9)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    if not args.raw.is_file():
        raise SystemExit(f"Missing raw export {args.raw}")
    output = ROOT / "tables" / f"gee_monthly_raw_diagnostics_{args.raw.stem.removeprefix('monthly_isolation_raw_')}.csv"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    raw = pd.read_csv(args.raw)
    source = pd.read_csv(ROOT / "code/gee/panel.csv")
    source = source.loc[source.cls.eq(1), ["lon", "lat", "greened_frac"]].copy()
    for frame in (raw, source):
        frame["lon_key"] = frame.lon.round(6)
        frame["lat_key"] = frame.lat.round(6)
    linked = raw.merge(source[["lon_key", "lat_key", "greened_frac"]],
                       on=["lon_key", "lat_key"], how="left", validate="one_to_one")
    own = linked.green_count_cell
    expected = (linked.greened_frac * 9).round()
    neighbor = linked.green_count_8_neighbor_cells
    within300 = linked.green_count_other_within_300m_center
    bad_monthly = 0
    full_valid_months = 0
    for month in MONTHS:
        valid = linked[f"ndvi_valid_green_count_{month}"]
        high = linked[f"ndvi_ge30_green_count_{month}"]
        bad_monthly += int((valid.lt(-.25) | high.lt(-.25) |
                            valid.gt(own + .25) | high.gt(valid + .25)).sum())
        full_valid_months += int((valid - own).abs().le(.25).sum())
    row = dict(raw_file=args.raw.name, n_exported=len(raw), n_source_joined=int(linked.greened_frac.notna().sum()),
               n_own_cell_dose_disagreements=int((own - expected).abs().gt(.25).sum()),
               n_neighbor_order_violations=int((neighbor - within300).gt(.25).sum()),
               n_monthly_count_violations=bad_monthly,
               n_full_valid_cell_months=full_valid_months,
               n_possible_cell_months=len(raw) * len(MONTHS))
    pd.DataFrame([row]).to_csv(output, index=False)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
