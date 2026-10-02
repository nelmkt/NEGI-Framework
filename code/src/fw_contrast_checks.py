"""Exploratory paired spatial contrasts; these do not identify physical mechanisms."""
from __future__ import annotations
import numpy as np
import pandas as pd
from fw_matching import Bootstrap, _point
from fw_panel import strata
from fw_config import SETTINGS


def evaluate(df, key, treated, controls, cfg):
    codes, levels = pd.factorize(key)
    t, c = treated.to_numpy(bool), controls.to_numpy(bool)
    y, dose = df.d_lst.to_numpy(float), df.n_greened_px.to_numpy(float)
    eligible = np.bincount(codes, weights=c.astype(float), minlength=len(levels)) >= cfg.min_controls
    masks = [t & (dose >= lo) & (dose <= hi) for lo, hi in cfg.dose_bins]
    coast = df.coast_km.to_numpy(float)
    coast_masks = [t & (coast >= 5) & (coast < 10), t & (coast >= 10)]

    def one(w):
        nc = np.bincount(codes, weights=w*c, minlength=len(levels))
        available = eligible[codes] & (nc[codes] > 0)
        means = []
        for mask in masks:
            used = mask & available
            means.append(float(np.average(dose[used], weights=w[used])) if w[used].sum() > 0 else np.nan)
        effects = [_point(y, mask, c, codes, eligible, w) for mask in masks]
        coastal = [_point(y, mask, c, codes, eligible, w, dose) for mask in coast_masks]
        return np.r_[effects, means, coastal]

    point = one(np.ones(len(df)))
    boot = Bootstrap(df.block, cfg.n_boot, cfg.seed)
    draws = np.array([one(boot.weights(r)) for r in range(cfg.n_boot)])
    counts = [int((m & eligible[codes]).sum()) for m in masks]

    def interval(a):
        valid = np.isfinite(a)
        lo, hi = np.percentile(a[valid], [2.5, 97.5]) if valid.any() else (np.nan, np.nan)
        return dict(lo_C=float(lo), hi_C=float(hi), valid_draws=int(valid.sum()), requested_draws=cfg.n_boot)

    labels = [f"{lo}–{hi}" if lo != hi else str(lo) for lo, hi in cfg.dose_bins]
    reproduced = pd.DataFrame([dict(pixels=labels[i], n_matched=counts[i], mean_pixels=point[4+i],
                                    estimate_C=point[i], **interval(draws[:, i])) for i in range(4)])
    mixing = []
    for i in range(3):
        benchmark = point[3]*point[4+i]/9
        benchmark_draws = draws[:, 3]*draws[:, 4+i]/9
        delta = draws[:, i]-benchmark_draws
        mixing.append(dict(pixels=labels[i], n_matched=counts[i], mean_pixels=point[4+i],
                           measured_C=point[i], benchmark_C=benchmark, difference_C=point[i]-benchmark,
                           **interval(delta), interpretation="measured minus area-scaled 9-pixel contrast; exploratory"))
    coastal = pd.DataFrame([dict(contrast="at least 10 km minus 5–10 km", estimate_C=point[9]-point[8],
                                n_near=int((coast_masks[0] & eligible[codes]).sum()),
                                n_far=int((coast_masks[1] & eligible[codes]).sum()),
                                **interval(draws[:, 9]-draws[:, 8]),
                                interpretation="difference of dose-weighted slopes, not a causal coast effect")])
    return dict(mixing=pd.DataFrame(mixing), coastal=coastal, dose_reproduction=reproduced)


def run(df, cfg):
    outside = df.setting == SETTINGS[0]
    treated = outside & (df.group == "greened")
    controls = outside & (df.group == "control") & (df.d_own_built.abs() < cfg.stable_surface_max)
    return evaluate(df, strata(df, cfg), treated, controls, cfg)
