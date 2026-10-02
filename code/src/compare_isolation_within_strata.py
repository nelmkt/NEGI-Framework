"""Feasibility and descriptive within-stratum isolation comparison.

One-pixel outside treated cells are compared within the primary pre-treatment
matching key augmented by baseline-LST tercile. Coast band is already part
of that key. This does not remove unmeasured site selection.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from fw_config import Config
import fw_panel as panel
from analyze_monthly_isolation import link_export

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tables" / "gee_isolation_within_strata_v6.csv"


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    df, _ = panel.load(cfg)
    df = link_export(df, pd.read_csv(ROOT / "code/gee/monthly_isolation_raw_v4.csv"))
    treated = df.group.eq("greened") & df.setting.eq("outside the built-up area") & df.n_greened_px.eq(1)
    controls = df.group.eq("control") & df.setting.eq("outside the built-up area") & df.d_own_built.abs().lt(cfg.stable_surface_max)
    cut = np.quantile(df.loc[treated, "lst_pre"], [1 / 3, 2 / 3])
    key = panel.strata(df, cfg, baseline_only=True) + "|lst" + np.digitize(df.lst_pre, cut).astype(str)
    isolated = treated & df.green_count_other_within_300m_center.lt(.5)
    other = treated & ~isolated
    grouping = pd.DataFrame({"key": key, "isolated": isolated, "other": other, "controls": controls})
    counts = grouping.groupby("key")[["isolated", "other", "controls"]].sum()
    common = counts.index[(counts.isolated > 0) & (counts.other > 0) & (counts.controls >= cfg.min_controls)]
    rows = []
    for stratum in common:
        iso = df.loc[isolated & key.eq(stratum)]
        non = df.loc[other & key.eq(stratum)]
        rows.append({"stratum": stratum, "n_isolated": len(iso), "n_nonisolated": len(non),
                     "n_controls": int(counts.loc[stratum, "controls"]),
                     "isolated_mean_d_lst_C": float(iso.d_lst.mean()),
                     "nonisolated_mean_d_lst_C": float(non.d_lst.mean()),
                     "isolated_minus_nonisolated_C": float(iso.d_lst.mean() - non.d_lst.mean()),
                     "isolated_mean_baseline_lst_C": float(iso.lst_pre.mean()),
                     "nonisolated_mean_baseline_lst_C": float(non.lst_pre.mean()),
                     "isolated_mean_coast_km": float(iso.coast_km.mean()),
                     "nonisolated_mean_coast_km": float(non.coast_km.mean()),
                     "baseline_lst_tercile_lower_C": float(cut[0]),
                     "baseline_lst_tercile_upper_C": float(cut[1])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    if len(out):
        point = float(np.average(out.isolated_minus_nonisolated_C, weights=out.n_isolated))
        print("common strata",len(out),"isolated",int(out.n_isolated.sum()),
              "nonisolated",int(out.n_nonisolated.sum()),
              "weighted within-stratum difference C",point)
        print("isolation status not comparable for",int(isolated.sum() - out.n_isolated.sum()),
              "of",int(isolated.sum()),"classified isolated one-pixel cells")
    else:
        print("No common strata; comparison withheld")


if __name__ == "__main__":
    main()
