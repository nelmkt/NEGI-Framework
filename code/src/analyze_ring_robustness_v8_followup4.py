"""Saved-panel ring collinearity and leave-one-spatial-block-out follow-up.

Uses the unchanged primary pre-treatment match, saved v8d external counts and
original zero-intercept ring regression. It writes only new versioned files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
from analyze_isolation_reweighting_v8 import ROOT, load_saved
from analyze_exposure_rings_v8_followup3 import RING_NAMES, wls_no_intercept

OUT_PAIR = ROOT / "tables/ring_pair_collinearity_v8_followup4.csv"
OUT_LOO = ROOT / "tables/ring_leave_one_block_out_v8_followup4.csv"
OUT_SUMMARY = ROOT / "tables/ring_leave_one_block_summary_v8_followup4.csv"
PRIOR_RING = ROOT / "tables/ring_exposure_regression_v8_followup3.csv"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.asarray(a, float), np.asarray(b, float)
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(aa @ bb / denom) if denom > 0 else np.nan


def normalized_condition(x: np.ndarray) -> float:
    matrix = np.asarray(x, float)
    norms = np.linalg.norm(matrix, axis=0)
    return float(np.linalg.cond(matrix / norms)) if (norms > 0).all() else np.nan


def analyze(df: pd.DataFrame, cfg: Config):
    controls = (df.cls.eq(3) & df.setting.eq(SETTINGS[0]) &
                df.d_own_built.abs().lt(cfg.stable_surface_max)).to_numpy()
    key, labels = pd.factorize(df.key)
    counts = np.bincount(key, controls.astype(float), minlength=len(labels))
    treated = ((df.cls.eq(1) & df.setting.eq(SETTINGS[0])).to_numpy() &
               (counts >= cfg.min_controls)[key])
    own = df.n_greened_px.to_numpy(float)
    small = treated & (own <= 2)
    belt = df.southern_belt.to_numpy(bool)
    if int(treated.sum()) != 1006 or int(small.sum()) != 419:
        raise SystemExit("The primary outside matched population changed")
    e90, e180, e300 = [df[f"external_green_px_{r}m"].to_numpy(float)
                        for r in (90, 180, 300)]
    ring = np.column_stack((own, e90, e180-e90, e300-e180))
    if (ring[treated, 1:] < 0).any() or not np.allclose(ring[treated, 1:].sum(axis=1), e300[treated]):
        raise SystemExit("External counts do not partition into non-overlapping rings")
    groups = {"all outside": treated, "1–2": small,
              "southern belt": treated & belt, "outside belt": treated & ~belt}

    pair_rows = []
    for name, mask in groups.items():
        x = ring[mask]
        external = x[:, 1:]
        raw3 = float(np.linalg.cond(external))
        norm3 = normalized_condition(external)
        raw4 = float(np.linalg.cond(x))
        norm4 = normalized_condition(x)
        for left in range(1, 4):
            for right in range(left+1, 4):
                a, b = x[:, left], x[:, right]
                corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
                pair_rows.append(dict(group=name, ring_a=RING_NAMES[left], ring_b=RING_NAMES[right],
                                      n_cells=int(mask.sum()), n_blocks=int(df.loc[mask, "block"].nunique()),
                                      uncentred_cosine=cosine(a, b), pearson_correlation=corr,
                                      zero_share_a=float(np.mean(a == 0)), zero_share_b=float(np.mean(b == 0)),
                                      external_three_rank=int(np.linalg.matrix_rank(external)),
                                      external_three_condition_raw=raw3,
                                      external_three_condition_normalized=norm3,
                                      full_four_rank=int(np.linalg.matrix_rank(x)),
                                      full_four_condition_raw=raw4,
                                      full_four_condition_normalized=norm4))

    y = df.d_lst.to_numpy(float)
    block = df.block.to_numpy(str)

    def matched_contrast(weights: np.ndarray):
        cw = weights * controls
        count = np.bincount(key, cw, minlength=len(labels))
        total = np.bincount(key, cw * y, minlength=len(labels))
        means = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        valid = small & (weights > 0) & (count[key] > 0)
        return y-means[key], valid

    full_delta, full_valid = matched_contrast(np.ones(len(df)))
    if not full_valid[small].all():
        raise SystemExit("A primary 1–2-pixel cell lacks a matched control")
    full_beta = wls_no_intercept(ring[small], full_delta[small], np.ones(small.sum()))
    old = pd.read_csv(PRIOR_RING)
    old_small = old.loc[old.group.eq("1–2")].set_index("term")
    if set(old_small.index) != set(RING_NAMES) or not np.allclose(
            full_beta, old_small.loc[list(RING_NAMES), "beta_C_per_pixel"].to_numpy(float), atol=1e-12):
        raise SystemExit("The original 1–2-pixel four-term regression does not reproduce")

    small_blocks = sorted(set(block[small]))
    if len(small_blocks) != 32:
        raise SystemExit("Expected 32 primary 1–2-pixel treated blocks")
    loo_rows = []
    for removed in small_blocks:
        weights = (block != removed).astype(float)
        delta, valid = matched_contrast(weights)
        beta = wls_no_intercept(ring[valid], delta[valid], weights[valid])
        lost_support = int((small & (block != removed) & ~valid).sum())
        deleted_small = small & (block == removed)
        row = dict(deleted_block=removed, deleted_1_2_cells=int(deleted_small.sum()),
                   deleted_controls=int((controls & (block == removed)).sum()),
                   deleted_1_2_belt_share=float(belt[deleted_small].mean()),
                   retained_1_2_cells=int(valid.sum()),
                   retained_1_2_blocks=int(df.loc[valid, "block"].nunique()),
                   lost_other_treated_for_no_controls=lost_support,
                   design_rank=int(np.linalg.matrix_rank(ring[valid])),
                   design_condition=float(np.linalg.cond(ring[valid])))
        for k, term in enumerate(RING_NAMES):
            row[f"beta_{term}_C_per_pixel"] = float(beta[k])
            row[f"change_{term}_C_per_pixel"] = float(beta[k]-full_beta[k])
        loo_rows.append(row)

    loo = pd.DataFrame(loo_rows)
    near = loo["beta_external_0_90m_C_per_pixel"].to_numpy(float)
    change = loo["change_external_0_90m_C_per_pixel"].to_numpy(float)
    largest = loo.iloc[int(np.nanargmax(np.abs(change)))]
    summary = pd.DataFrame([dict(n_deleted_block_runs=len(loo), full_1_2_cells=int(small.sum()),
                                 full_1_2_blocks=len(small_blocks),
                                 full_beta_external_0_90m_C_per_pixel=float(full_beta[1]),
                                 minimum_leave_one_out_beta_C_per_pixel=float(np.nanmin(near)),
                                 maximum_leave_one_out_beta_C_per_pixel=float(np.nanmax(near)),
                                 maximum_absolute_change_C_per_pixel=float(np.nanmax(np.abs(change))),
                                 most_influential_deleted_block=largest.deleted_block,
                                 most_influential_block_deleted_cells=int(largest.deleted_1_2_cells),
                                 most_influential_block_belt_share=float(largest.deleted_1_2_belt_share),
                                 sign_reversals=int(np.sum(near >= 0)),
                                 invalid_design_runs=int(np.isnan(near).sum()),
                                 runs_losing_other_treated_support=int((loo.lost_other_treated_for_no_controls > 0).sum()))])
    return pd.DataFrame(pair_rows), loo, summary


def main():
    for path in (OUT_PAIR, OUT_LOO, OUT_SUMMARY):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite earlier output: {path}")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    print("Loading saved primary panel and v8d external counts...", flush=True)
    df = load_saved(cfg)
    pair, loo, summary = analyze(df, cfg)
    for data, path in ((pair, OUT_PAIR), (loo, OUT_LOO), (summary, OUT_SUMMARY)):
        data.to_csv(path, index=False)
    print("Wrote ring-pair collinearity, 32 block deletions, and summary", flush=True)


if __name__ == "__main__":
    main()
