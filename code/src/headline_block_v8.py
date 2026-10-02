"""Primary outside class slopes, bootstrap medians and dominant-block deletion."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from fw_config import Config
import fw_panel as panel
import fw_matching as matching

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tables/headline_block_v8.csv"


def build(df, cfg):
    key = panel.strata(df, cfg, baseline_only=True)
    treated = df.group.eq("greened") & df.setting.eq("outside the built-up area")
    controls = (df.group.eq("control") & df.setting.eq("outside the built-up area") &
                df.d_own_built.abs().lt(cfg.stable_surface_max))
    matched = treated & matching.control_weights(df, treated, controls, key, cfg).gt(0)
    dose2 = np.square(df.loc[matched, "n_greened_px"].to_numpy(float))
    per_block = pd.Series(dose2, index=df.index[matched]).groupby(df.loc[matched, "block"]).sum()
    dominant = per_block.idxmax()
    total = float(per_block.sum())
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    rows = []
    for label, lo, hi in (("all", 1, 9), ("1–2", 1, 2), ("3–4", 3, 4),
                          ("5–8", 5, 8), ("9/9", 9, 9)):
        selected = treated & df.n_greened_px.between(lo, hi)
        selected_matched = selected & matched
        fit = matching.att(df, selected, controls, "d_lst", key, cfg, boot,
                           dose="n_greened_px", keep_reps=True)
        reps = fit["_reps"]
        reps = reps[np.isfinite(reps)]
        drops = []
        for block in sorted(df.loc[selected_matched, "block"].unique()):
            cut = df.block.eq(block)
            value = matching.att(df, selected & ~cut, controls & ~cut,
                                 "d_lst", key, cfg, None, dose="n_greened_px")["estimate_C"]
            drops.append(value)
        vals = np.asarray(drops, float)
        if len(vals) > 2 and np.isfinite(vals).all():
            se = np.sqrt((len(vals) - 1) / len(vals) * np.square(vals - vals.mean()).sum())
            crit = student_t.ppf(.975, len(vals) - 1)
            jk_lo, jk_hi = fit["estimate_C"] - crit * se, fit["estimate_C"] + crit * se
        else:
            jk_lo = jk_hi = np.nan
        cut = df.block.eq(dominant)
        without = matching.att(df, selected & ~cut, controls & ~cut, "d_lst", key, cfg,
                               None, dose="n_greened_px")
        leverage = float(np.square(df.loc[selected_matched, "n_greened_px"]).sum() / total)
        rows.append({"dose_class": label, "n_matched": int(selected_matched.sum()),
                     "n_blocks": int(df.loc[selected_matched, "block"].nunique()),
                     "dose_squared_leverage_share": leverage,
                     "point_C_per_pixel": fit["estimate_C"], "bootstrap_median_C": float(np.median(reps)),
                     "bootstrap_lo_C": fit["lo_C"], "bootstrap_hi_C": fit["hi_C"],
                     "jackknife_lo_C": jk_lo, "jackknife_hi_C": jk_hi,
                     "dominant_pooled_block": dominant,
                     "dominant_pooled_block_leverage_share": float(per_block.max() / total),
                     "dominant_block_removed_C": without["estimate_C"],
                     "n_matched_after_removal": without["n_treated_matched"],
                     "n_boot_valid": len(reps)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    df, _ = panel.load(cfg)
    result = build(df, cfg)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))
