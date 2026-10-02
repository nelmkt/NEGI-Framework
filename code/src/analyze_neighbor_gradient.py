"""Pre-specified greening-count categories around outside one-pixel cells.

These categories describe neighboring greening intensity at two saved radii,
not distance rings or a causal spillover dose. No thresholds depend on LST.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from fw_config import Config
import fw_panel as panel
import fw_matching as matching
from analyze_monthly_isolation import link_export

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tables" / "gee_neighbor_count_gradient_v7.csv"
BINS = (("0", 0, 0), ("1–2", 1, 2), ("3–8", 3, 8), ("≥9", 9, np.inf))
MEASURES = (("eight adjacent 90 m cells", "green_count_8_neighbor_cells"),
            ("within 300 m of 90 m cell centre", "green_count_other_within_300m_center"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--baseline-lst-tercile", action="store_true",
                        help="Add fixed baseline-LST terciles to the primary strata")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    df, _ = panel.load(cfg)
    df = link_export(df, pd.read_csv(ROOT / "code/gee/monthly_isolation_raw_v4.csv"))
    treated = df.group.eq("greened") & df.setting.eq("outside the built-up area") & df.n_greened_px.eq(1)
    controls = df.group.eq("control") & df.setting.eq("outside the built-up area") & df.d_own_built.abs().lt(cfg.stable_surface_max)
    key = panel.strata(df, cfg, baseline_only=True)
    if args.baseline_lst_tercile:
        cuts = np.quantile(df.loc[treated, "lst_pre"], [1 / 3, 2 / 3])
        key = key + "|lst" + np.digitize(df.lst_pre, cuts).astype(str)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    rows = []
    for measure, col in MEASURES:
        reference = None
        for label, low, high in BINS:
            selected = treated & df[col].ge(low - .25) & df[col].le(high + .25)
            result = matching.att(df, selected, controls, "d_lst", key, cfg, boot,
                                  dose="n_greened_px", keep_reps=True)
            matched = selected & matching.control_weights(df, selected, controls, key, cfg).gt(0)
            if reference is None:
                reference = result
            paired = result["_reps"] - reference["_reps"]
            paired = paired[np.isfinite(paired)]
            rows.append({
                "setting": "outside the built-up area", "dose": "one greened 30 m pixel",
                "neighbor_measure": measure, "neighbor_count_bin": label,
                "n_classified": int(selected.sum()), "n_matched": int(matched.sum()),
                "n_blocks": int(df.loc[matched, "block"].nunique()),
                "per_pixel_C": result["estimate_C"], "lo_C": result["lo_C"], "hi_C": result["hi_C"],
                "difference_from_zero_count_C": result["estimate_C"] - reference["estimate_C"],
                "difference_lo_C": float(np.percentile(paired, 2.5)) if label != "0" and len(paired) else 0.0,
                "difference_hi_C": float(np.percentile(paired, 97.5)) if label != "0" and len(paired) else 0.0,
                "baseline_lst_C": float(df.loc[matched, "lst_pre"].mean()) if matched.any() else np.nan,
                "coast_km": float(df.loc[matched, "coast_km"].mean()) if matched.any() else np.nan,
                "n_boot_valid": result["n_boot_valid"],
            })
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(rows)} neighbor-count rows")


if __name__ == "__main__":
    main()
