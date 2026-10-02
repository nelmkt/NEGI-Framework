"""Versioned saved-data ring regression and collinearity audit.

The four ring columns are own, external 0–90 m, 90–180 m, and 180–300 m.
The matched treated-minus-control contrast and 2,000 spatial-block draws use
the same unchanged primary pre-treatment strata as the prior class slopes.
No Earth Engine call or earlier output replacement occurs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
from analyze_isolation_reweighting_v8 import ROOT, RADII, interval, load_saved, slope

OUT_RING = ROOT / "tables/ring_exposure_regression_v8_followup3.csv"
OUT_COLLIN = ROOT / "tables/joint_exposure_collinearity_v8_followup3.csv"
OUT_SUBGROUP = ROOT / "tables/isolation_subgroups_v8_followup3.csv"
PREVIOUS_JOINT = ROOT / "tables/joint_exposure_regression_v8_followup2.csv"
RING_NAMES = ("own", "external_0_90m", "external_90_180m", "external_180_300m")


def wls_no_intercept(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return NaNs where the sampled four-column design loses algebraic rank."""
    matrix = np.asarray(x, float)
    target = np.asarray(y, float)
    w = np.asarray(weights, float)
    good = (w > 0) & np.isfinite(target) & np.isfinite(matrix).all(axis=1)
    if good.sum() <= matrix.shape[1]:
        return np.full(matrix.shape[1], np.nan)
    sw = np.sqrt(w[good])
    weighted = matrix[good] * sw[:, None]
    if np.linalg.matrix_rank(weighted) < matrix.shape[1]:
        return np.full(matrix.shape[1], np.nan)
    return np.linalg.lstsq(weighted, target[good] * sw, rcond=None)[0]


def two_column_diagnostics(own: np.ndarray, external: np.ndarray) -> dict:
    """Uncentred geometry; condition number is for X, not X'X."""
    a, b = np.asarray(own, float), np.asarray(external, float)
    x = np.column_stack((a, b))
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    cosine = float(a @ b / (an * bn)) if an > 0 and bn > 0 else np.nan
    return dict(uncentred_cosine=cosine,
                design_condition_number=float(np.linalg.cond(x)),
                column_normalized_condition_number=float(np.linalg.cond(x / np.array([an, bn])))
                if an > 0 and bn > 0 else np.nan,
                algebraic_rank=int(np.linalg.matrix_rank(x)),
                zero_external_share=float(np.mean(b == 0)))


def analyze(df: pd.DataFrame, cfg: Config):
    controls = (df.cls.eq(3) & df.setting.eq(SETTINGS[0]) &
                df.d_own_built.abs().lt(cfg.stable_surface_max)).to_numpy()
    key, labels = pd.factorize(df.key)
    n_ctrl = np.bincount(key, controls.astype(float), minlength=len(labels))
    treated = ((df.cls.eq(1) & df.setting.eq(SETTINGS[0])).to_numpy() &
               (n_ctrl >= cfg.min_controls)[key])
    own = df.n_greened_px.to_numpy(float)
    y = df.d_lst.to_numpy(float)
    belt = df.southern_belt.to_numpy(bool)
    if int(treated.sum()) != 1006 or int((treated & (own <= 2)).sum()) != 419:
        raise SystemExit("The saved primary outside population does not reconcile")
    ext = {r: df[f"external_green_px_{r}m"].to_numpy(float) for r in RADII}
    ring = np.column_stack((own, ext[90], ext[180] - ext[90], ext[300] - ext[180]))
    if (ring[treated, 1:] < 0).any() or not np.allclose(ring[treated, 1:].sum(axis=1), ext[300][treated]):
        raise SystemExit("External ring counts are not a non-overlapping partition")
    groups = {"all outside": treated, "1–2": treated & (own <= 2),
              "southern belt": treated & belt, "outside belt": treated & ~belt}
    col_groups = {"all outside": treated, "1–2": treated & (own <= 2),
                  "5–8": treated & (own >= 5) & (own <= 8),
                  "9/9": treated & (own == 9),
                  "5–9 combined": treated & (own >= 5) & (own <= 9)}
    for r in RADII:
        if not np.allclose(own[treated] + ext[r][treated],
                           df.loc[treated, f"combined_green_px_{r}m"].to_numpy(float)):
            raise SystemExit(f"Combined saved pixel counts disagree at {r} m")
    block_code, blocks = pd.factorize(df.block)
    rng = np.random.default_rng(cfg.seed)
    draws = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=cfg.n_boot)

    def contrasts(bw):
        wc = bw * controls
        count = np.bincount(key, wc, minlength=len(labels))
        total = np.bincount(key, wc * y, minlength=len(labels))
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        valid = treated & (count[key] > 0) & (bw > 0)
        return y - mean[key], valid

    delta0, valid0 = contrasts(np.ones(len(df)))
    if not valid0[treated].all():
        raise SystemExit("A primary treated cell lacks its matched control")
    point = {g: wls_no_intercept(ring[m], delta0[m], np.ones(m.sum()))
             for g, m in groups.items()}
    reps = {g: [] for g in groups}
    small_ix = np.flatnonzero(groups["1–2"])
    small_own = own[small_ix]
    small_belt = belt[small_ix]
    subgroups = {"exactly one": small_own == 1, "exactly two": small_own == 2,
                 "southern belt": small_belt, "outside belt": ~small_belt}
    iso = {r: ext[r][small_ix] == 0 for r in RADII}
    if [int(iso[r].sum()) for r in RADII] != [76, 38, 26]:
        raise SystemExit("The saved isolated 1–2 cell counts do not reconcile")
    sub_point = {(r, g, flag): slope(own[small_ix[sel]], delta0[small_ix[sel]],
                                   np.ones(sel.sum()))
                 for r in RADII for g, subgroup in subgroups.items()
                 for flag, sel in (("isolated", iso[r] & subgroup),
                                   ("non-isolated", ~iso[r] & subgroup))}
    sub_reps = {(r, g, flag): [] for r in RADII for g in subgroups
                for flag in ("isolated", "non-isolated")}

    for j, draw in enumerate(draws, 1):
        bw = draw[block_code].astype(float)
        delta, valid = contrasts(bw)
        for g, mask in groups.items():
            take = mask & valid
            reps[g].append(wls_no_intercept(ring[take], delta[take], bw[take]))
        for r in RADII:
            for g, subgroup in subgroups.items():
                for flag, sel in (("isolated", iso[r] & subgroup),
                                  ("non-isolated", ~iso[r] & subgroup)):
                    ix = small_ix[sel & valid[small_ix]]
                    sub_reps[r, g, flag].append(slope(own[ix], delta[ix], bw[ix]))
        if j % 250 == 0:
            print(f"  matched spatial-block draws {j}/{cfg.n_boot}", flush=True)

    ring_rows = []
    for g, mask in groups.items():
        array = np.asarray(reps[g], float)
        rank = int(np.linalg.matrix_rank(ring[mask]))
        condition = float(np.linalg.cond(ring[mask]))
        for k, name in enumerate(RING_NAMES):
            lo, hi, n = interval(array[:, k])
            ring_rows.append(dict(group=g, term=name, n_cells=int(mask.sum()),
                                  n_blocks=int(df.loc[mask, "block"].nunique()),
                                  status="diagnostic_only_under_20_cells" if mask.sum() < 20 else "estimate",
                                  beta_C_per_pixel=float(point[g][k]), lo_C=lo, hi_C=hi,
                                  valid_block_draws=n, algebraic_rank=rank,
                                  design_condition_number=condition,
                                  zero_count_share=float(np.mean(ring[mask, k] == 0))))

    old = pd.read_csv(PREVIOUS_JOINT)
    col_rows = []
    for row in old.itertuples(index=False):
        mask = col_groups[row.group]
        if int(mask.sum()) != int(row.n_cells):
            raise SystemExit("Earlier joint-model cell counts do not reconcile")
        a, b = own[mask], ext[int(row.radius_m)][mask]
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else np.nan
        if not (np.isnan(corr) and np.isnan(row.own_external_correlation)) and not np.isclose(
                corr, row.own_external_correlation, atol=1e-12):
            raise SystemExit("Earlier joint-model correlation does not reproduce")
        diag = two_column_diagnostics(a, b)
        note = ("own marginal coefficient unavailable: own dose fixed at nine"
                if row.group == "9/9" else
                "both columns full rank; causal own/external split not identified")
        col_rows.append(dict(radius_m=int(row.radius_m), group=row.group,
                             n_cells=int(mask.sum()), n_blocks=int(row.n_blocks),
                             prior_correlation=row.own_external_correlation,
                             identification_note=note, **diag))

    sub_rows = []
    for r in RADII:
        for g, subgroup in subgroups.items():
            isol, non = iso[r] & subgroup, ~iso[r] & subgroup
            ni, nn = int(isol.sum()), int(non.sum())
            ib = np.asarray(sub_reps[r, g, "isolated"], float)
            nb = np.asarray(sub_reps[r, g, "non-isolated"], float)
            li, ui, _ = interval(ib)
            ln, un, _ = interval(nb)
            ld, ud, nd = interval(nb - ib)
            diagnostic = ni < 20 or nn < 20
            sub_rows.append(dict(radius_m=r, subgroup=g, isolated_cells=ni,
                                 nonisolated_cells=nn,
                                 isolated_blocks=int(df.iloc[small_ix[isol]].block.nunique()),
                                 nonisolated_blocks=int(df.iloc[small_ix[non]].block.nunique()),
                                 status="diagnostic_only_under_20_cells" if diagnostic else "estimate",
                                 isolated_slope_C= sub_point[r, g, "isolated"],
                                 isolated_lo_C=np.nan if diagnostic else li,
                                 isolated_hi_C=np.nan if diagnostic else ui,
                                 nonisolated_slope_C=sub_point[r, g, "non-isolated"],
                                 nonisolated_lo_C=np.nan if diagnostic else ln,
                                 nonisolated_hi_C=np.nan if diagnostic else un,
                                 nonisolated_minus_isolated_C=(sub_point[r,g,"non-isolated"]-
                                                               sub_point[r,g,"isolated"]),
                                 difference_lo_C=np.nan if diagnostic else ld,
                                 difference_hi_C=np.nan if diagnostic else ud,
                                 valid_paired_draws=nd))
    return pd.DataFrame(ring_rows), pd.DataFrame(col_rows), pd.DataFrame(sub_rows)


def main():
    for path in (OUT_RING, OUT_COLLIN, OUT_SUBGROUP):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite earlier output: {path}")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    print("Loading saved primary panel and v8d pixel counts...", flush=True)
    df = load_saved(cfg)
    ring, collin, subgroup = analyze(df, cfg)
    for data, path in ((ring, OUT_RING), (collin, OUT_COLLIN), (subgroup, OUT_SUBGROUP)):
        data.to_csv(path, index=False)
    print("Wrote three new versioned tables", flush=True)


if __name__ == "__main__":
    main()
