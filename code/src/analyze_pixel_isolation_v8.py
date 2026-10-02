"""Dose-specific primary-match sensitivity using the verified 300 m pixel screen.

The existing raw export has no 90/180 m counts and no combined exposure.
Those radii are written as unavailable, never approximated from cell centres.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
import fw_panel as panel
import fw_matching as matching
from analyze_pixel_centered_isolation import SCREEN, validate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tables/pixel_isolation_by_dose_v8.csv"


def make_rows(df, cfg):
    key = panel.strata(df, cfg, baseline_only=True)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    rows = []
    for setting in SETTINGS:
        controls = (df.group.eq("control") & df.setting.eq(setting) &
                    df.d_own_built.abs().lt(cfg.stable_surface_max))
        for dose_label, lower, upper in (("exactly one", 1, 1), ("1–2", 1, 2), ("3–4", 3, 4),
                                         ("5–8", 5, 8), ("9/9", 9, 9)):
            treated = df.group.eq("greened") & df.setting.eq(setting) & df.n_greened_px.between(lower, upper)
            if not treated.any():
                continue
            matched = treated & matching.control_weights(df, treated, controls, key, cfg).gt(0)
            full = matching.att(df, treated, controls, "d_lst", key, cfg, boot,
                                dose="n_greened_px", keep_reps=True)
            for radius in (90, 180, 300):
                if radius < 300:
                    rows.append({"setting": setting, "dose_class": dose_label, "radius_m": radius,
                                 "status": "not_exported", "n_classified_full": int(treated.sum()),
                                 "n_matched_full": int(matched.sum())})
                    continue
                isolated = treated & df[SCREEN].lt(.5)
                matched_isolated = isolated & matching.control_weights(df, isolated, controls, key, cfg).gt(0)
                n, blocks = int(matched_isolated.sum()), int(df.loc[matched_isolated, "block"].nunique())
                diagnostic = n < 20 or blocks < 5
                result = (matching.att(df, isolated, controls, "d_lst", key, cfg,
                                       boot if not diagnostic else None,
                                       dose="n_greened_px", keep_reps=True) if n else None)
                paired = (result["_reps"] - full["_reps"]) if result is not None and not diagnostic else np.array([])
                paired = paired[np.isfinite(paired)]
                rows.append({
                    "setting": setting, "dose_class": dose_label, "radius_m": radius,
                    "status": "diagnostic_only" if diagnostic else "selected_site_sensitivity",
                    "n_classified_full": int(treated.sum()), "n_matched_full": int(matched.sum()),
                    "n_blocks_full": int(df.loc[matched, "block"].nunique()),
                    "full_slope_C_per_own_pixel": full["estimate_C"],
                    "full_lo_C": full["lo_C"], "full_hi_C": full["hi_C"],
                    "n_classified_isolated": int(isolated.sum()), "n_matched_isolated": n,
                    "n_blocks_isolated": blocks,
                    "isolated_share_of_classified": float(isolated.sum() / treated.sum()),
                    "isolated_slope_C_per_own_pixel": result["estimate_C"] if result is not None else np.nan,
                    "isolated_lo_C": result["lo_C"] if result is not None and not diagnostic else np.nan,
                    "isolated_hi_C": result["hi_C"] if result is not None and not diagnostic else np.nan,
                    "isolated_minus_full_C": result["estimate_C"] - full["estimate_C"] if result is not None else np.nan,
                    "difference_lo_C": float(np.percentile(paired, 2.5)) if len(paired) else np.nan,
                    "difference_hi_C": float(np.percentile(paired, 97.5)) if len(paired) else np.nan,
                    "isolated_baseline_lst_C": float(df.loc[matched_isolated, "lst_pre"].mean()) if n else np.nan,
                    "isolated_coast_km": float(df.loc[matched_isolated, "coast_km"].mean()) if n else np.nan,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    df, _ = panel.load(cfg)
    df = validate(ROOT / "code/gee/pixel_isolation_raw.csv", df)
    table = make_rows(df, cfg)
    table.to_csv(OUT, index=False)
    print("Wrote", OUT, len(table), "rows; 90/180 m unavailable; 300 m counts verified")
    print(table.loc[(table.setting == SETTINGS[0]) & table.radius_m.eq(300),
                    ["dose_class", "n_matched_full", "n_matched_isolated", "n_blocks_isolated",
                     "isolated_slope_C_per_own_pixel", "status"]].to_string(index=False))
