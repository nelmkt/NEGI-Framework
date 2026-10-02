"""Paired inner-subring coefficient contrasts on the saved primary match."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
from analyze_isolation_reweighting_v8 import ROOT, PANEL, interval
from analyze_exposure_rings_v8_followup3 import wls_no_intercept
from analyze_pixel_subrings_v8_followup6 import DEFAULT_RAW, load_new, subring_design, TERMS_FINE

PRIOR = ROOT / "tables/pixel_subring_regression_v8_followup6.csv"
OUT = ROOT / "tables/pixel_subring_pair_differences_v8_followup7.csv"
PAIRS = (("external_0_30m", "external_30_60m"),
         ("external_0_30m", "external_60_90m"),
         ("external_30_60m", "external_60_90m"))


def analyze(df, cfg):
    controls = (df.cls.eq(3) & df.setting.eq(SETTINGS[0]) &
                df.d_own_built.abs().lt(cfg.stable_surface_max)).to_numpy()
    key, labels = pd.factorize(df.key)
    n_ctrl = np.bincount(key, controls.astype(float), minlength=len(labels))
    own = df.n_greened_px.to_numpy(float)
    treated = ((df.cls.eq(1) & df.setting.eq(SETTINGS[0])).to_numpy() &
               (n_ctrl >= cfg.min_controls)[key] & (own <= 2))
    if int(treated.sum()) != 419 or int(df.loc[treated, "block"].nunique()) != 32:
        raise SystemExit("Primary outside 1–2-pixel population changed")
    small_ix = np.flatnonzero(treated)
    columns = [own[treated]] + [df.loc[treated, f"external_green_px_{r}m"].to_numpy(float)
                                for r in (30, 60, 90, 180, 300)]
    x = subring_design(*columns)["fine_three_inner_rings"]
    y = df.d_lst.to_numpy(float)
    block_code, blocks = pd.factorize(df.block)
    draws = np.random.default_rng(cfg.seed).multinomial(
        len(blocks), np.full(len(blocks), 1 / len(blocks)), size=cfg.n_boot)

    def contrast(weights):
        cw = weights * controls
        count = np.bincount(key, cw, minlength=len(labels))
        total = np.bincount(key, cw * y, minlength=len(labels))
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        valid = treated & (weights > 0) & (count[key] > 0)
        return y - mean[key], valid

    delta0, valid0 = contrast(np.ones(len(df)))
    if not valid0[treated].all():
        raise SystemExit("Primary treated cell lacks matched control")
    beta = wls_no_intercept(x, delta0[small_ix], np.ones(len(small_ix)))
    prior = pd.read_csv(PRIOR).query("model == 'fine_three_inner_rings'").set_index("term")
    if set(prior.index) != set(TERMS_FINE) or not np.allclose(
            beta, prior.loc[list(TERMS_FINE), "beta_C_per_pixel"].to_numpy(float), atol=1e-12):
        raise SystemExit("Prior fine subring coefficients do not reproduce")
    reps = []
    for i, draw in enumerate(draws, 1):
        bw = draw[block_code].astype(float)
        delta, valid = contrast(bw)
        take = valid[small_ix]
        reps.append(wls_no_intercept(x[take], delta[small_ix][take], bw[small_ix][take]))
        if i % 250 == 0:
            print(f"  matched spatial-block draws {i}/{cfg.n_boot}", flush=True)
    reps = np.asarray(reps, float)
    rows = []
    for a, b in PAIRS:
        j, k = TERMS_FINE.index(a), TERMS_FINE.index(b)
        diff = reps[:, j] - reps[:, k]
        lo, hi, n = interval(diff)
        rows.append(dict(term_a=a, term_b=b, n_cells=len(small_ix), n_blocks=32,
                         beta_a_minus_beta_b_C_per_pixel=float(beta[j]-beta[k]),
                         lo_C=lo, hi_C=hi, valid_paired_block_draws=n,
                         total_block_draws=cfg.n_boot))
    return pd.DataFrame(rows)


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite earlier output: {OUT}")
    cfg = Config(panel_path=PANEL)
    df = load_new(DEFAULT_RAW, cfg)
    result = analyze(df, cfg)
    result.to_csv(OUT, index=False)
    print(f"Wrote {OUT.name}: {len(result)} paired inner-subring contrasts")


if __name__ == "__main__":
    main()
