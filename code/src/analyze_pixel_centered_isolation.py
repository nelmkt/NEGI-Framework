"""Check a new pixel-centred Earth Engine export and compare selected sites.

This is a site-selection sensitivity, not an estimate of thermal spillover.
Run only after export_pixel_centered_isolation.py has completed successfully.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config
import fw_panel as panel
import fw_matching as matching

ROOT = Path(__file__).resolve().parents[2]
REVISION = "pixel-centered-native30-v7"
SCREEN = "max_external_green_within300_any_own30"


def validate(raw_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    meta = json.loads(raw_path.with_suffix(".json").read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256((ROOT / "code/gee/panel.csv").read_bytes()).hexdigest()
    if meta.get("recipe_revision") != REVISION or meta.get("source_panel_sha256") != expected_sha:
        raise SystemExit("Recipe or source-panel SHA mismatch; do not interpret export.")
    if any(meta.get(name) != 0 for name in ("dose_count_disagreements", "unjoined_cells",
                                            "negative_external_flags", "missing_pixel_centered_screens")):
        raise SystemExit("Exporter reported a failed geometry/count check; do not interpret export.")
    raw = pd.read_csv(raw_path)
    needed = {"lon", "lat", "green_count_cell", SCREEN, "negative_external_flag"}
    if not needed.issubset(raw.columns):
        raise SystemExit(f"Missing columns: {sorted(needed - set(raw.columns))}")
    if len(raw) != meta.get("n_exported_cells"):
        raise SystemExit("Row count does not match export metadata.")
    key = ["lon_key", "lat_key"]
    left = df.assign(lon_key=df.lon.round(6), lat_key=df.lat.round(6))
    right = raw.assign(lon_key=raw.lon.round(6), lat_key=raw.lat.round(6))
    if right.duplicated(key).any():
        raise SystemExit("Duplicate cell centres in new export.")
    joined = left.merge(right.drop(columns=["lon", "lat"]), on=key, how="left",
                        validate="one_to_one", sort=False)
    green = joined.group.eq("greened")
    if int(green.sum()) != len(raw) or joined.loc[green, SCREEN].isna().any():
        raise SystemExit("New export does not cover precisely the saved greened population.")
    if (joined.loc[green, "green_count_cell"] - joined.loc[green, "n_greened_px"]).abs().gt(.25).any():
        raise SystemExit("Own-cell dose disagrees with source panel.")
    if (joined.loc[green, SCREEN].lt(-.25) | joined.loc[green, "negative_external_flag"].gt(.25)).any():
        raise SystemExit("Impossible negative external-neighbor count.")
    return joined


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=ROOT / "code/gee/pixel_centered_isolation_raw_v7.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "tables/gee_pixel_centered_isolation_effects_v7.csv")
    a = ap.parse_args()
    if not a.raw.is_file() or not a.raw.with_suffix(".json").is_file():
        raise SystemExit("Missing new export and/or metadata; run the Earth Engine exporter first.")
    if a.out.exists():
        raise SystemExit(f"Refusing to overwrite {a.out}")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    df, _ = panel.load(cfg)
    df = validate(a.raw, df)
    controls = (df.group.eq("control") & df.setting.eq("outside the built-up area") &
                df.d_own_built.abs().lt(cfg.stable_surface_max))
    key = panel.strata(df, cfg, baseline_only=True)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    rows = []
    for dose, mask in (("1–2", df.n_greened_px.between(1, 2)),
                       ("exactly one", df.n_greened_px.eq(1))):
        treated = df.group.eq("greened") & df.setting.eq("outside the built-up area") & mask
        full = matching.att(df, treated, controls, "d_lst", key, cfg, boot,
                            dose="n_greened_px", keep_reps=True)
        for label, selected in (("all", treated),
                                ("no external greened 30 m pixel within 300 m of any own greened pixel",
                                 treated & df[SCREEN].lt(.5))):
            result = matching.att(df, selected, controls, "d_lst", key, cfg, boot,
                                  dose="n_greened_px", keep_reps=True)
            matched = selected & matching.control_weights(df, selected, controls, key, cfg).gt(0)
            paired = result["_reps"] - full["_reps"]
            paired = paired[np.isfinite(paired)]
            rows.append({
                "dose": dose, "screen": label, "n_classified": int(selected.sum()),
                "n_matched": int(matched.sum()), "n_blocks": int(df.loc[matched, "block"].nunique()),
                "per_pixel_C": result["estimate_C"], "lo_C": result["lo_C"], "hi_C": result["hi_C"],
                "difference_from_full_C": result["estimate_C"] - full["estimate_C"],
                "difference_lo_C": float(np.percentile(paired, 2.5)) if len(paired) else np.nan,
                "difference_hi_C": float(np.percentile(paired, 97.5)) if len(paired) else np.nan,
                "baseline_lst_C": float(df.loc[matched, "lst_pre"].mean()) if matched.any() else np.nan,
                "coast_km": float(df.loc[matched, "coast_km"].mean()) if matched.any() else np.nan,
                "n_boot_valid": result["n_boot_valid"],
            })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"Wrote {a.out}: {len(rows)} rows (site-selection sensitivity only)")


if __name__ == "__main__":
    main()
