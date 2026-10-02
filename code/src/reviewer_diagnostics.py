"""Reproduce added reviewer diagnostics from the observed panel and saved run."""
from pathlib import Path
import numpy as np
import pandas as pd
from fw_config import Config, SETTINGS
import fw_panel as panel
import fw_matching as matching
import fw_followup as followup


def main():
    cfg = Config()
    df, _ = panel.load(cfg)
    rs = followup.setup(df, cfg)[SETTINGS[0]]
    treated, controls, key = rs["treated"], rs["controls"], rs["key"]
    matched = treated & (rs["weights"] > 0)
    dose = df.n_greened_px.to_numpy(float)
    y = df.d_lst.to_numpy(float)
    sd = ((df.loc[treated, followup.BALANCE].var() + df.loc[controls, followup.BALANCE].var()) / 2) ** .5
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)

    caliper = []
    profiles = []
    retained = {}
    for width in (.5, .2):
        ti, graph = followup.caliper_matrix(df, matched, controls, key, sd, width, cfg)
        keep = np.zeros(len(df), dtype=bool)
        keep[ti] = True
        retained[width] = pd.Series(keep, index=df.index)
        for lo, hi in cfg.dose_bins:
            pos = np.asarray(df.iloc[ti].n_greened_px.between(lo, hi))
            tclass = ti[pos]
            subgraph = graph[pos]
            cell, slope = followup.caliper_effect(y, dose, tclass, subgraph, np.ones(len(df)))
            draws = np.array([followup.caliper_effect(y, dose, tclass, subgraph, boot.weights(i))[0]
                              for i in range(cfg.n_boot)])
            interval = followup.interval(draws)
            label = f"{lo}–{hi}" if lo != hi else str(lo)
            main_same = matching.att(df, retained[width] & df.n_greened_px.between(lo, hi),
                                     controls, "d_lst", key, cfg, None, dose="n_greened_px")
            caliper.append(dict(caliper_sd=width, pixels=label, n_cells=len(tclass),
                                n_blocks=int(df.iloc[tclass].block.nunique()), estimate_cell_C=cell,
                                lo_cell_C=interval["lo_C"], hi_cell_C=interval["hi_C"],
                                per_pixel_C=slope, n_valid=interval["n_valid"],
                                main_on_same_cells_per_pixel_C=main_same["estimate_C"]))
    for label, mask in [
        ("9/9 all classified", treated & (df.n_greened_px == 9)),
        ("9/9 main matched", matched & (df.n_greened_px == 9)),
        ("9/9 dropped by main matching", treated & ~matched & (df.n_greened_px == 9)),
        ("9/9 retained by 0.5 SD", retained[.5] & (df.n_greened_px == 9)),
        ("9/9 dropped from matched by 0.5 SD", matched & ~retained[.5] & (df.n_greened_px == 9)),
    ]:
        d = df[mask]
        profiles.append(dict(group=label, n_cells=len(d), n_blocks=d.block.nunique(),
                             mean_lon=d.lon.mean(), mean_lat=d.lat.mean(),
                             lat_min=d.lat.min(), lat_max=d.lat.max(),
                             baseline_lst_C=d.lst_pre.mean(), baseline_ndvi=d.ndvi_pre.mean(),
                             coast_km=d.coast_km.mean(), coast_km_median=d.coast_km.median(),
                             southern_belt_share=d.lat.between(*cfg.arc_lat_bounds, inclusive="left").mean()))
    pd.DataFrame(caliper).to_csv("tables/reviewer_caliper_by_dose.csv", index=False)
    pd.DataFrame(profiles).to_csv("tables/reviewer_9pixel_profiles.csv", index=False)

    total = float(np.square(dose[matched.to_numpy()]).sum())
    leverage = []
    for lo, hi in cfg.dose_bins:
        mask = matched & df.n_greened_px.between(lo, hi)
        e = matching.att(df, mask, controls, "d_lst", key, cfg, None, dose="n_greened_px")
        leverage.append(dict(pixels=f"{lo}–{hi}" if lo != hi else str(lo),
                             n_cells=int(mask.sum()), dose_squared_share=float(np.square(dose[mask.to_numpy()]).sum()/total),
                             class_per_pixel_C=e["estimate_C"], class_mean_cell_C=matching.att(
                                 df, mask, controls, "d_lst", key, cfg, None)["estimate_C"]))
    pd.DataFrame(leverage).to_csv("tables/reviewer_dose_leverage.csv", index=False)

    holds = pd.read_csv("tables/test_region_holdouts.csv")
    covered = set(holds.loc[holds.setting == SETTINGS[0], "region"].astype(str))
    region = panel._grid_id(df, cfg.region_deg)
    counts = region[matched].value_counts()
    coverage = []
    for name, n in counts.items():
        coverage.append(dict(region=name, n_matched=int(n), in_regional_holdout=name in covered,
                             reason="reported" if name in covered else "fewer than reporting minimum of 10 cells"))
    c = pd.DataFrame(coverage)
    assert int(c.loc[c.in_regional_holdout, "n_matched"].sum()) == int(
        holds.loc[holds.setting == SETTINGS[0], "n_cells"].sum())
    assert int(c.loc[~c.in_regional_holdout, "n_matched"].sum()) == 28
    c.to_csv("tables/reviewer_holdout_coverage.csv", index=False)

    basekey = panel.strata(df, cfg, baseline_only=True)
    baseweights = matching.control_weights(df, treated, controls, basekey, cfg)
    baseline = treated & (baseweights > 0)
    mixes = []
    for name, mask in [("main", matched), ("baseline-only strata", baseline)]:
        total_d2 = float(np.square(dose[mask.to_numpy()]).sum())
        for lo, hi in cfg.dose_bins:
            sub = mask & df.n_greened_px.between(lo, hi)
            mixes.append(dict(specification=name, pixels=f"{lo}–{hi}" if lo != hi else str(lo),
                              n_cells=int(sub.sum()), mean_dose=float(df.loc[sub, "n_greened_px"].mean()),
                              dose_squared_share=float(np.square(dose[sub.to_numpy()]).sum()/total_d2)))
    pd.DataFrame(mixes).to_csv("tables/reviewer_baseline_only_mix.csv", index=False)
    print("Saved reviewer caliper, 9-pixel profile, leverage, holdout and baseline-only diagnostics.")


if __name__ == "__main__":
    main()
