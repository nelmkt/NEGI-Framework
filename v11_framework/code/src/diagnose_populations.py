"""Audit treatment retention and spatial support for the reported specifications."""
from pathlib import Path
import numpy as np
import pandas as pd
from fw_config import Config, SETTINGS
import fw_followup as followup
import fw_panel as panel


def compute(df, cfg):
    setups = followup.setup(df, cfg)
    retention = []
    coast = []
    for setting in SETTINGS:
        rs = setups[setting]
        treated = rs["treated"]
        matched = treated & (rs["weights"] > 0)
        controls = rs["controls"]
        sd = ((df.loc[treated, followup.BALANCE].var() + df.loc[controls, followup.BALANCE].var()) / 2) ** .5
        calipers = {}
        for width in (.5, .2):
            ti, _ = followup.caliper_matrix(df, matched, controls, rs["key"], sd, width, cfg)
            calipers[width] = pd.Series(df.index.isin(ti), index=df.index)
        for lo, hi in cfg.dose_bins:
            band = df.n_greened_px.between(lo, hi)
            all_count = int((treated & band).sum())
            match_count = int((matched & band).sum())
            row = dict(setting=setting, pixels=f"{lo}–{hi}" if lo != hi else str(lo),
                       classified=all_count, matched=match_count, dropped_matching=all_count-match_count,
                       matching_retention=match_count/all_count if all_count else np.nan,
                       matched_blocks=int(df.loc[matched & band, "block"].nunique()))
            for width, keep in calipers.items():
                label = f"caliper_{width:g}_sd"
                row[label] = int((keep & band).sum())
                row[label + "_blocks"] = int(df.loc[keep & band, "block"].nunique())
            retention.append(row)
        coastband = panel.coast_band(df.coast_km, cfg)
        for name in coastband.cat.categories:
            mask = matched & (coastband.to_numpy() == name)
            coast.append(dict(setting=setting, coast_band=name, n_matched=int(mask.sum()),
                              n_spatial_blocks=int(df.loc[mask, "block"].nunique()),
                              n_matching_regions=int(panel._grid_id(df.loc[mask], cfg.region_deg).nunique())))
    return pd.DataFrame(retention), pd.DataFrame(coast)


def main():
    cfg = Config()
    df, _ = panel.load(cfg)
    retention, coast = compute(df, cfg)
    out = Path("tables")
    retention.to_csv(out / "population_retention.csv", index=False)
    coast.to_csv(out / "coastal_block_counts.csv", index=False)
    print("Saved dose retention and coast block tables.")


if __name__ == "__main__":
    main()
