"""Versioned paired middle-ring change for primary outside 1–2-pixel cells.

Compare the original four-term model with models that omit the outermost ring
or merge the two outer rings. All models use the same matched controls and
paired spatial-block bootstrap draws. Earlier outputs are never overwritten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
from analyze_isolation_reweighting_v8 import ROOT, load_saved, interval
from analyze_exposure_rings_v8_followup3 import wls_no_intercept, RING_NAMES

OUT_COEFFICIENTS = ROOT / "tables/ring_reduced_coefficients_v8_followup6.csv"
OUT_COMPARISONS = ROOT / "tables/ring_reduced_comparisons_v8_followup6.csv"
PRIOR_COEFFICIENTS = ROOT / "tables/ring_reduced_coefficients_v8_followup5.csv"
PRIOR_COMPARISONS = ROOT / "tables/ring_reduced_comparisons_v8_followup5.csv"
PRIOR_RING = ROOT / "tables/ring_exposure_regression_v8_followup3.csv"

MODELS = {
    "full_four_terms": ("own", "external_0_90m", "external_90_180m", "external_180_300m"),
    "drop_180_300m": ("own", "external_0_90m", "external_90_180m"),
    "merge_90_300m": ("own", "external_0_90m", "external_90_300m"),
}


def model_design(own: np.ndarray, e90: np.ndarray, e180: np.ndarray,
                 e300: np.ndarray) -> dict[str, np.ndarray]:
    """Non-overlapping rings; merged outer count is the same pixel union."""
    full = np.column_stack((own, e90, e180-e90, e300-e180))
    if (full[:, 1:] < 0).any():
        raise ValueError("Cumulative external counts are not nondecreasing")
    return {"full_four_terms": full,
            "drop_180_300m": full[:, :3],
            "merge_90_300m": np.column_stack((own, e90, e300-e90))}


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
    e90, e180, e300 = (df[f"external_green_px_{r}m"].to_numpy(float)
                        for r in (90, 180, 300))
    designs = model_design(own[treated], e90[treated], e180[treated], e300[treated])
    if any(np.linalg.matrix_rank(x) != x.shape[1] for x in designs.values()):
        raise SystemExit("A point-estimate design is algebraically rank deficient")
    y = df.d_lst.to_numpy(float)
    block_code, blocks = pd.factorize(df.block)
    draws = np.random.default_rng(cfg.seed).multinomial(
        len(blocks), np.full(len(blocks), 1 / len(blocks)), size=cfg.n_boot)

    def contrast(weights):
        wc = weights * controls
        count = np.bincount(key, wc, minlength=len(labels))
        total = np.bincount(key, wc * y, minlength=len(labels))
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        valid = treated & (weights > 0) & (count[key] > 0)
        return y - mean[key], valid

    zero, valid0 = contrast(np.ones(len(df)))
    if not valid0[treated].all():
        raise SystemExit("A primary treated cell lacks its matched control")
    small_ix = np.flatnonzero(treated)
    point = {name: wls_no_intercept(x, zero[small_ix], np.ones(len(small_ix)))
             for name, x in designs.items()}
    old = pd.read_csv(PRIOR_RING).query("group == '1–2'").set_index("term")
    if set(old.index) != set(RING_NAMES) or not np.allclose(
            point["full_four_terms"], old.loc[list(RING_NAMES), "beta_C_per_pixel"], atol=1e-12):
        raise SystemExit("Full four-term point estimates do not reproduce the prior table")

    reps = {name: [] for name in MODELS}
    for i, draw in enumerate(draws, 1):
        bw = draw[block_code].astype(float)
        delta, valid = contrast(bw)
        take = valid[small_ix]
        for name, x in designs.items():
            reps[name].append(wls_no_intercept(x[take], delta[small_ix][take], bw[small_ix][take]))
        if i % 250 == 0:
            print(f"  matched spatial-block draws {i}/{cfg.n_boot}", flush=True)
    reps = {name: np.asarray(values, float) for name, values in reps.items()}

    coef_rows = []
    for name, terms in MODELS.items():
        x = designs[name]
        for j, term in enumerate(terms):
            lo, hi, n = interval(reps[name][:, j])
            coef_rows.append(dict(model=name, term=term, n_cells=len(small_ix), n_blocks=32,
                                  beta_C_per_pixel=float(point[name][j]), lo_C=lo, hi_C=hi,
                                  valid_block_draws=n, total_block_draws=cfg.n_boot,
                                  design_rank=int(np.linalg.matrix_rank(x)),
                                  design_condition=float(np.linalg.cond(x))))

    comparison_rows = []
    for name in ("drop_180_300m", "merge_90_300m"):
        shared_terms = ("own", "external_0_90m", "external_90_180m") if name == "drop_180_300m" else ("own", "external_0_90m")
        for term in shared_terms:
            j = MODELS[name].index(term)
            k = MODELS["full_four_terms"].index(term)
            diff = reps[name][:, j] - reps["full_four_terms"][:, k]
            lo, hi, n = interval(diff)
            before, after = point["full_four_terms"][k], point[name][j]
            comparison_rows.append(dict(model=name, term=term, n_cells=len(small_ix), n_blocks=32,
                                        full_beta_C_per_pixel=float(before),
                                        reduced_beta_C_per_pixel=float(after),
                                        reduced_minus_full_C_per_pixel=float(after-before),
                                        difference_lo_C=lo, difference_hi_C=hi,
                                        valid_paired_block_draws=n, total_block_draws=cfg.n_boot))
    coefficients = pd.DataFrame(coef_rows)
    comparisons = pd.DataFrame(comparison_rows)
    old_c = pd.read_csv(PRIOR_COEFFICIENTS)
    old_d = pd.read_csv(PRIOR_COMPARISONS)
    if len(coefficients) != len(old_c) or not np.allclose(
            coefficients.beta_C_per_pixel, old_c.beta_C_per_pixel, atol=1e-12):
        raise SystemExit("Follow-up 5 reduced-ring point estimates did not reproduce")
    repeated = comparisons.loc[~comparisons.term.eq("external_90_180m")].reset_index(drop=True)
    if not repeated[["model", "term"]].equals(old_d[["model", "term"]]):
        raise SystemExit("Follow-up 5 paired-comparison identities did not reproduce")
    for column in old_d.columns:
        if column in ("model", "term"):
            continue
        if not np.allclose(repeated[column].to_numpy(float),
                           old_d[column].to_numpy(float), equal_nan=True, atol=1e-12):
            raise SystemExit(f"Follow-up 5 paired comparison did not reproduce: {column}")
    return coefficients, comparisons


def main():
    for path in (OUT_COEFFICIENTS, OUT_COMPARISONS):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite earlier output: {path}")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    df = load_saved(cfg)
    coefficients, comparisons = analyze(df, cfg)
    coefficients.to_csv(OUT_COEFFICIENTS, index=False)
    comparisons.to_csv(OUT_COMPARISONS, index=False)
    print("Wrote two versioned reduced-ring tables", flush=True)


if __name__ == "__main__":
    main()
