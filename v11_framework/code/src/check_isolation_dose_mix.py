"""Check whether the 300 m screen's dose mix alone explains attenuation.

This is a selected-site sensitivity on the same endpoint panel, not a
contamination or irrigation-identity estimator. Never overwrite an output.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from fw_config import Config
import fw_panel as panel
import fw_matching as matching
from analyze_monthly_isolation import link_export

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tables" / "gee_isolation_dose1_comparison_v5_corrected.csv"


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    cfg = Config(panel_path=ROOT / "code" / "gee" / "panel.csv")
    df, _ = panel.load(cfg)
    df = link_export(df, pd.read_csv(ROOT / "code" / "gee" / "monthly_isolation_raw_v4.csv"))
    setting = "outside the built-up area"
    treated = df.group.eq("greened") & df.setting.eq(setting) & df.n_greened_px.eq(1)
    controls = df.group.eq("control") & df.setting.eq(setting) & df.d_own_built.abs().lt(cfg.stable_surface_max)
    key = panel.strata(df, cfg, baseline_only=True)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    full = matching.att(df, treated, controls, "d_lst", key, cfg, boot,
                        dose="n_greened_px", keep_reps=True)
    screened = treated & df.green_count_other_within_300m_center.lt(.5)
    part = matching.att(df, screened, controls, "d_lst", key, cfg, boot,
                        dose="n_greened_px", keep_reps=True)
    reps = part["_reps"] - full["_reps"]
    reps = reps[np.isfinite(reps)]
    rows = []
    for label, selected, result in (("all one-pixel cells", treated, full),
                                    ("300 m centre-distance screen", screened, part)):
        matched = selected & matching.control_weights(df, selected, controls, key, cfg).gt(0)
        rows.append({
            "setting": setting, "screen": label, "n_classified": int(selected.sum()),
            "n_matched": int(matched.sum()), "n_matched_blocks": int(df.loc[matched, "block"].nunique()),
            "per_pixel_C": result["estimate_C"], "lo_C": result["lo_C"], "hi_C": result["hi_C"],
            "difference_from_full_C": result["estimate_C"] - full["estimate_C"],
            "difference_lo_C": (float(np.percentile(reps, 2.5)) if len(reps) else np.nan)
                               if label != "all one-pixel cells" else 0.0,
            "difference_hi_C": (float(np.percentile(reps, 97.5)) if len(reps) else np.nan)
                               if label != "all one-pixel cells" else 0.0,
            "baseline_lst_C": float(df.loc[matched, "lst_pre"].mean()),
            "coast_km": float(df.loc[matched, "coast_km"].mean()),
            "n_boot_valid": result["n_boot_valid"],
        })
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
