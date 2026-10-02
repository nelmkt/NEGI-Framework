"""Analyze a future pixel-centred 30/60 m export; never calls Earth Engine.

Fits both a fine three-subring 0–90 m model and a prespecified coarser
0–30/30–90 m model on the unchanged primary outside 1–2-pixel population.
Reports zero-count shares, pairwise uncentred cosines, ranks and block draws.
No output is written unless the new raw CSV/JSON and all reconciliation checks
pass. The exporter has not run in the agent's environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
from analyze_isolation_reweighting_v8 import ROOT, PANEL, RAW as PRIOR_RAW, load_saved, interval
from analyze_exposure_rings_v8_followup3 import wls_no_intercept

DEFAULT_RAW = ROOT / "code/gee/pixel_external_subrings_v8_followup5_raw.csv"
OUT_COEFFICIENTS = ROOT / "tables/pixel_subring_regression_v8_followup6.csv"
OUT_COLLINEARITY = ROOT / "tables/pixel_subring_collinearity_v8_followup6.csv"
EXPECTED_RECIPE = "unique-external-native30-subrings-v8-followup5"
TERMS_FINE = ("own", "external_0_30m", "external_30_60m", "external_60_90m",
              "external_90_180m", "external_180_300m")
TERMS_COARSE = ("own", "external_0_30m", "external_30_90m",
                "external_90_180m", "external_180_300m")


def subring_design(own, e30, e60, e90, e180, e300):
    own, e30, e60, e90, e180, e300 = [np.asarray(v, float) for v in
                                      (own, e30, e60, e90, e180, e300)]
    fine = np.column_stack((own, e30, e60-e30, e90-e60, e180-e90, e300-e180))
    if not np.isfinite(fine).all() or (fine[:, 1:] < 0).any():
        raise ValueError("Subring counts are missing, non-finite or non-nested")
    if not np.array_equal(fine[:, 1:4].sum(axis=1), e90):
        raise ValueError("The three inner rings do not sum to the existing 90 m count")
    coarse = np.column_stack((own, e30, e90-e30, e180-e90, e300-e180))
    return {"fine_three_inner_rings": fine, "coarse_two_inner_rings": coarse}


def load_new(raw_path: Path, cfg: Config):
    meta_path = raw_path.with_suffix(".json")
    if not raw_path.is_file() or not meta_path.is_file():
        raise SystemExit("New subring CSV/JSON are missing; run the new Earth Engine exporter first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("recipe_revision") != EXPECTED_RECIPE or any(meta.get("checks", {}).values()):
        raise SystemExit("New export recipe or validation checks do not match the prepared version")
    for label, path in (("source_panel_sha256", PANEL),
                        ("source_prior_external_counts_sha256", PRIOR_RAW)):
        if meta.get(label) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise SystemExit(f"New export's {label} differs from the saved source")
    raw = pd.read_csv(raw_path)
    prior = pd.read_csv(PRIOR_RAW)
    if len(raw) != 1333 or len(prior) != 1333:
        raise SystemExit("Expected 1,333 classified cells in both count exports")
    for frame in (raw, prior):
        frame["lon_key"], frame["lat_key"] = frame.lon.round(6), frame.lat.round(6)
        if frame[["lon_key", "lat_key"]].duplicated().any():
            raise SystemExit("Duplicate coordinate ID in pixel-count export")
    check = raw.merge(prior, on=["lon_key", "lat_key"], how="outer",
                      suffixes=("_new", "_prior"), validate="one_to_one", indicator=True)
    if not check._merge.eq("both").all():
        raise SystemExit("New and prior classified-cell IDs disagree")
    for r in (90, 180, 300):
        if not check[f"external_green_px_{r}m_new"].eq(
                check[f"external_green_px_{r}m_prior"]).all():
            raise SystemExit(f"New and prior {r} m pixel counts disagree")
    if not check.own_green_px_new.eq(check.own_green_px_prior).all():
        raise SystemExit("Own greened-pixel counts disagree")
    if not (raw.external_ring_0_30m + raw.external_ring_30_60m +
            raw.external_ring_60_90m).eq(raw.external_green_px_90m).all():
        raise SystemExit("Subring partition does not sum to the 90 m count")
    df = load_saved(cfg)
    added = raw[["lon_key", "lat_key", "external_green_px_30m",
                 "external_green_px_60m"]]
    df = df.merge(added, on=["lon_key", "lat_key"], how="left",
                  validate="one_to_one", sort=False)
    return df


def analyze(df: pd.DataFrame, cfg: Config):
    controls = (df.cls.eq(3) & df.setting.eq(SETTINGS[0]) &
                df.d_own_built.abs().lt(cfg.stable_surface_max)).to_numpy()
    key, labels = pd.factorize(df.key)
    n_ctrl = np.bincount(key, controls.astype(float), minlength=len(labels))
    own = df.n_greened_px.to_numpy(float)
    treated = ((df.cls.eq(1) & df.setting.eq(SETTINGS[0])).to_numpy() &
               (n_ctrl >= cfg.min_controls)[key] & (own <= 2))
    if int(treated.sum()) != 419 or int(df.loc[treated, "block"].nunique()) != 32:
        raise SystemExit("Primary outside 1–2-pixel population changed")
    cols = [own[treated]] + [df.loc[treated, f"external_green_px_{r}m"].to_numpy(float)
                             for r in (30, 60, 90, 180, 300)]
    designs = subring_design(*cols)
    small_ix = np.flatnonzero(treated)
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
        raise SystemExit("A primary treated cell lacks its matched control")
    point = {name: wls_no_intercept(x, delta0[small_ix], np.ones(len(small_ix)))
             for name, x in designs.items()}
    reps = {name: [] for name in designs}
    for i, draw in enumerate(draws, 1):
        bw = draw[block_code].astype(float)
        delta, valid = contrast(bw)
        take = valid[small_ix]
        for name, x in designs.items():
            reps[name].append(wls_no_intercept(x[take], delta[small_ix][take], bw[small_ix][take]))
        if i % 250 == 0:
            print(f"  matched spatial-block draws {i}/{cfg.n_boot}", flush=True)
    coef_rows = []
    for name, x in designs.items():
        terms = TERMS_FINE if name == "fine_three_inner_rings" else TERMS_COARSE
        rep = np.asarray(reps[name], float)
        rank = int(np.linalg.matrix_rank(x))
        cond = float(np.linalg.cond(x))
        for j, term in enumerate(terms):
            lo, hi, n = interval(rep[:, j])
            coef_rows.append(dict(model=name, term=term, n_cells=len(small_ix), n_blocks=32,
                                  beta_C_per_pixel=float(point[name][j]), lo_C=lo, hi_C=hi,
                                  valid_block_draws=n, total_block_draws=cfg.n_boot,
                                  design_rank=rank, design_condition=cond,
                                  zero_count_share=float(np.mean(x[:, j] == 0)),
                                  status="diagnostic_only_subring" if j else "descriptive_own"))
    fine = designs["fine_three_inner_rings"]
    pair_rows = []
    for j in range(1, fine.shape[1]):
        for k in range(j+1, fine.shape[1]):
            a, b = fine[:, j], fine[:, k]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            pair_rows.append(dict(term_a=TERMS_FINE[j], term_b=TERMS_FINE[k],
                                  n_cells=len(small_ix), n_blocks=32,
                                  uncentred_cosine=float(a @ b / denom) if denom else np.nan,
                                  pearson_correlation=float(np.corrcoef(a, b)[0, 1])
                                  if a.std() and b.std() else np.nan,
                                  zero_share_a=float(np.mean(a == 0)),
                                  zero_share_b=float(np.mean(b == 0)),
                                  design_rank=int(np.linalg.matrix_rank(fine)),
                                  design_condition=float(np.linalg.cond(fine))))
    return pd.DataFrame(coef_rows), pd.DataFrame(pair_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    for path in (OUT_COEFFICIENTS, OUT_COLLINEARITY):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite earlier output: {path}")
    cfg = Config(panel_path=PANEL)
    df = load_new(args.raw.resolve(), cfg)
    coefficients, collinearity = analyze(df, cfg)
    coefficients.to_csv(OUT_COEFFICIENTS, index=False)
    collinearity.to_csv(OUT_COLLINEARITY, index=False)
    print("Wrote two versioned subring tables; interpret zero shares and design rank before coefficients")


if __name__ == "__main__":
    main()
