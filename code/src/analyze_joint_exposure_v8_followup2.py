"""Saved-data primary-match joint exposure and isolation follow-up.

No Earth Engine calls, changed thresholds, or replacement of earlier output.
Coast distance is reconstructed with the package's saved coastline reference,
which is needed for the original pre-treatment matching key.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
from analyze_isolation_reweighting_v8 import (
    ROOT, RADII, CLASSES, entropy_weights, interval, load_saved, moments, slope,
)

OUT_JOINT = ROOT / "tables/joint_exposure_regression_v8_followup2.csv"
OUT_BALANCE = ROOT / "tables/isolation_balance_v8_followup2.csv"
OUT_SUBGROUP = ROOT / "tables/isolation_subgroups_v8_followup2.csv"
OUT_RELATIVE = ROOT / "tables/combined_class_relative_spread_v8_followup2.csv"
OUT_MANUSCRIPT = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP2.md"


def joint_fit(own: np.ndarray, external: np.ndarray, contrast: np.ndarray,
              weights: np.ndarray) -> tuple[float, float]:
    """Zero-intercept weighted least squares on the matched cell contrast."""
    x = np.column_stack((own, external)).astype(float)
    w = np.asarray(weights, float)
    good = (w > 0) & np.isfinite(contrast) & np.isfinite(x).all(axis=1)
    if good.sum() < 3:
        return np.nan, np.nan
    sw = np.sqrt(w[good])
    xw = x[good] * sw[:, None]
    if np.linalg.matrix_rank(xw) < 2:
        return np.nan, np.nan
    coef = np.linalg.lstsq(xw, contrast[good] * sw, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def relative_spread(slopes: np.ndarray) -> float:
    """Positive class range divided by the absolute four-class mean slope."""
    vals = np.asarray(slopes, float)
    if not np.isfinite(vals).all() or abs(vals.mean()) < 1e-12:
        return np.nan
    return float(np.ptp(vals) / abs(vals.mean()))


def calibration_matrix(covariates: np.ndarray, belt: np.ndarray,
                       is_two: np.ndarray) -> np.ndarray:
    """Nine original continuous moments plus two prespecified binary means."""
    centre = covariates.mean(axis=0)
    scale = covariates.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("A calibration covariate has no variation")
    return np.column_stack((moments(covariates, centre, scale),
                            belt.astype(float), is_two.astype(float)))


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
    class_masks = {label: treated & (own >= low) & (own <= high)
                   for label, low, high in CLASSES}
    if [int(class_masks[label].sum()) for label, _, _ in CLASSES] != [419, 183, 244, 160]:
        raise SystemExit("The primary outside matched population has changed")
    groups = {"all outside": treated, "1–2": class_masks["1–2"],
              "5–8": class_masks["5–8"], "9/9": class_masks["9/9"],
              "5–9 combined": class_masks["5–8"] | class_masks["9/9"]}
    external = {r: df[f"external_green_px_{r}m"].to_numpy(float) for r in RADII}
    block_code, blocks = pd.factorize(df.block)
    rng = np.random.default_rng(cfg.seed)
    draws = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=cfg.n_boot)

    def contrast_for(bw):
        cw = bw * controls
        count = np.bincount(key, cw, minlength=len(labels))
        total = np.bincount(key, cw * y, minlength=len(labels))
        means = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        valid = treated & (count[key] > 0) & (bw > 0)
        return y - means[key], valid

    contrast0, valid0 = contrast_for(np.ones(len(df)))
    if not valid0[treated].all():
        raise SystemExit("A primary treated cell lacks a matched control")
    small_ix = np.flatnonzero(class_masks["1–2"])
    x = df.loc[class_masks["1–2"], ["lst_pre", "coast_km", "ghsl_2015"]].to_numpy(float)
    is_two = own[small_ix] == 2
    design = calibration_matrix(x, belt[small_ix], is_two)
    isolated = {r: external[r][small_ix] == 0 for r in RADII}
    if [int(isolated[r].sum()) for r in RADII] != [76, 38, 26]:
        raise SystemExit("Saved 90/180/300 m isolated counts disagree")

    point_joint = {(r, g): joint_fit(own[m], external[r][m], contrast0[m],
                                    np.ones(m.sum()))
                   for r in RADII for g, m in groups.items()}
    joint_boot = {(r, g): [] for r in RADII for g in groups}

    point_cal = {}
    weight_info = {}
    cal_boot = {r: [] for r in RADII}
    iso_boot = {r: [] for r in RADII}
    cal_diff_boot = {r: [] for r in RADII}
    subgroup_names = {"exactly one": ~is_two, "exactly two": is_two,
                      "southern belt": belt[small_ix], "outside belt": ~belt[small_ix]}
    sub_point = {}
    sub_boot = {(r, name, exposure): [] for r in RADII
                for name in subgroup_names for exposure in ("isolated", "non-isolated")}
    for r in RADII:
        iso = isolated[r]
        w = entropy_weights(design[~iso], design[iso],
                            np.ones((~iso).sum()), np.ones(iso.sum()))
        if w is None:
            raise SystemExit(f"Expanded calibration infeasible for point estimate at {r} m")
        weight_info[r] = w
        nix, iix = small_ix[~iso], small_ix[iso]
        point_cal[r] = (slope(own[nix], contrast0[nix], w),
                        slope(own[iix], contrast0[iix], np.ones(len(iix))))
        for name, subgroup in subgroup_names.items():
            for exposure, selection in (("isolated", iso & subgroup),
                                        ("non-isolated", ~iso & subgroup)):
                ix = small_ix[selection]
                sub_point[r, name, exposure] = slope(own[ix], contrast0[ix], np.ones(len(ix)))

    dose_fields = {"own": own, **{f"combined_{r}m": own + external[r] for r in RADII}}
    rel_point = {}
    rel_boot = {exposure: [] for exposure in dose_fields}
    for exposure, field in dose_fields.items():
        slopes = [slope(field[mask], contrast0[mask], np.ones(mask.sum()))
                  for _, mask in class_masks.items()]
        rel_point[exposure] = relative_spread(np.asarray(slopes))

    for j, draw in enumerate(draws, 1):
        bw = draw[block_code].astype(float)
        contrast, valid = contrast_for(bw)
        for r in RADII:
            for name, mask in groups.items():
                select = mask & valid
                joint_boot[r, name].append(joint_fit(own[select], external[r][select],
                                                     contrast[select], bw[select]))
            iso = isolated[r]
            i_local = iso & valid[small_ix]
            n_local = ~iso & valid[small_ix]
            iix, nix = small_ix[i_local], small_ix[n_local]
            ip = slope(own[iix], contrast[iix], bw[iix])
            iso_boot[r].append(ip)
            fit = entropy_weights(design[n_local], design[i_local], bw[nix], bw[iix])
            cp = slope(own[nix], contrast[nix], fit) if fit is not None else np.nan
            cal_boot[r].append(cp)
            cal_diff_boot[r].append(cp - ip)
            for name, subgroup in subgroup_names.items():
                for exposure, selection in (("isolated", iso & subgroup),
                                            ("non-isolated", ~iso & subgroup)):
                    ix = small_ix[selection & valid[small_ix]]
                    sub_boot[r, name, exposure].append(slope(own[ix], contrast[ix], bw[ix]))
        for exposure, field in dose_fields.items():
            slopes = [slope(field[mask & valid], contrast[mask & valid], bw[mask & valid])
                      for mask in class_masks.values()]
            rel_boot[exposure].append(relative_spread(np.asarray(slopes)))
        if j % 250 == 0:
            print(f"  block bootstrap {j}/{cfg.n_boot}", flush=True)

    joint_rows = []
    for r in RADII:
        for name, mask in groups.items():
            a = own[mask]
            b = external[r][mask]
            corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
            reps = np.asarray(joint_boot[r, name], float)
            lo_a, hi_a, valid_a = interval(reps[:, 0])
            lo_b, hi_b, valid_b = interval(reps[:, 1])
            joint_rows.append(dict(radius_m=r, group=name, n_cells=int(mask.sum()),
                                   n_blocks=int(df.loc[mask, "block"].nunique()),
                                   status="diagnostic_only_under_20_cells" if mask.sum() < 20 else "estimate",
                                   beta_own_C_per_pixel=point_joint[r, name][0], own_lo_C=lo_a, own_hi_C=hi_a,
                                   beta_external_C_per_pixel=point_joint[r, name][1], ext_lo_C=lo_b, ext_hi_C=hi_b,
                                   own_external_correlation=corr, valid_block_draws=min(valid_a, valid_b),
                                   method="zero-intercept WLS on treated-minus-stratum-control change"))

    balance_rows = []
    for r in RADII:
        iso = isolated[r]
        w = weight_info[r]
        cp, ip = point_cal[r]
        cp_lo, cp_hi, cp_n = interval(cal_boot[r])
        ip_lo, ip_hi, ip_n = interval(iso_boot[r])
        dp_lo, dp_hi, dp_n = interval(cal_diff_boot[r])
        balance_rows.append(dict(radius_m=r, isolated_cells=int(iso.sum()),
                                 nonisolated_cells=int((~iso).sum()),
                                 isolated_blocks=int(df.iloc[small_ix[iso]].block.nunique()),
                                 isolated_slope_C_per_own_pixel=ip, isolated_lo_C=ip_lo,
                                 isolated_hi_C=ip_hi, calibrated_nonisolated_slope_C_per_own_pixel=cp,
                                 calibrated_lo_C=cp_lo, calibrated_hi_C=cp_hi,
                                 calibrated_minus_isolated_C=cp-ip, difference_lo_C=dp_lo,
                                 difference_hi_C=dp_hi, weight_ess=float(1/np.sum(w*w)),
                                 largest_weight_share=float(w.max()),
                                 max_abs_moment_error=float(np.max(np.abs(w @ design[~iso] - design[iso].mean(axis=0)))),
                                 feasible_block_draws=dp_n, feasible_share=dp_n/cfg.n_boot,
                                 isolated_draws=ip_n, calibrated_draws=cp_n,
                                 status="diagnostic_only_under_20_cells" if iso.sum() < 20 else "estimate"))

    subgroup_rows = []
    for r in RADII:
        for name, subgroup in subgroup_names.items():
            point_i = sub_point[r, name, "isolated"]
            point_n = sub_point[r, name, "non-isolated"]
            count_i = int((isolated[r] & subgroup).sum())
            count_n = int((~isolated[r] & subgroup).sum())
            diagnostic = count_i < 20 or count_n < 20
            ib = np.asarray(sub_boot[r, name, "isolated"], float)
            nb = np.asarray(sub_boot[r, name, "non-isolated"], float)
            ilo, ihi, inv = interval(ib)
            nlo, nhi, nnv = interval(nb)
            dlo, dhi, dnv = interval(nb - ib)
            subgroup_rows.append(dict(radius_m=r, subgroup=name, isolated_cells=count_i,
                                      nonisolated_cells=count_n,
                                      isolated_blocks=int(df.iloc[small_ix[isolated[r] & subgroup]].block.nunique()),
                                      nonisolated_blocks=int(df.iloc[small_ix[~isolated[r] & subgroup]].block.nunique()),
                                      status="diagnostic_only_under_20_cells" if diagnostic else "estimate",
                                      isolated_slope_C_per_own_pixel=point_i,
                                      isolated_lo_C=np.nan if diagnostic else ilo,
                                      isolated_hi_C=np.nan if diagnostic else ihi,
                                      nonisolated_slope_C_per_own_pixel=point_n,
                                      nonisolated_lo_C=np.nan if diagnostic else nlo,
                                      nonisolated_hi_C=np.nan if diagnostic else nhi,
                                      nonisolated_minus_isolated_C=point_n-point_i,
                                      difference_lo_C=np.nan if diagnostic else dlo,
                                      difference_hi_C=np.nan if diagnostic else dhi,
                                      valid_paired_draws=dnv, isolated_valid_draws=inv,
                                      nonisolated_valid_draws=nnv))

    relative_rows = []
    for exposure in dose_fields:
        lo, hi, valid = interval(rel_boot[exposure])
        relative_rows.append(dict(exposure=exposure, radius_m=0 if exposure=="own" else
                                  int(exposure.split("_")[1][:-1]),
                                  relative_spread_range_over_abs_mean=rel_point[exposure],
                                  lo=lo, hi=hi, valid_block_draws=valid,
                                  note="descriptive; external pixels may enter multiple focal-cell exposures"))
    return (pd.DataFrame(joint_rows), pd.DataFrame(balance_rows),
            pd.DataFrame(subgroup_rows), pd.DataFrame(relative_rows))


def write_manuscript(joint, balance, subgroup, relative):
    source = (ROOT / "manuscript/MANUSCRIPT_V8.md").read_text(encoding="utf-8")
    old_abstract = "This is consistent with neighbour contamination, with site selection not ruled out."
    new_abstract = ("The isolation comparison is a selected-site sensitivity: differences in location, "
                    "pixel mix, land use and unmeasured characteristics remain possible explanations; "
                    "it does not identify neighbour contamination.")
    if source.count(old_abstract) != 1:
        raise SystemExit("Could not safely update the abstract in the versioned manuscript")
    source = source.replace(old_abstract, new_abstract)
    obsolete = ("The export lacks external-pixel counts at 90 and 180 m and lacks unique neighbouring "
                "greened-pixel counts at any radius, so a combined-exposure slope or revised Figure 5a "
                "ratio cannot be computed from it.")
    replacement = ("The later saved v8d export counts unique external 30 m greened pixels at 90, 180 and "
                   "300 m. The joint-exposure coefficients and expanded isolation calibration are reported "
                   "below as descriptive matched contrasts; they do not identify spillover or irrigation.")
    if source.count(obsolete) != 1:
        raise SystemExit("Could not safely replace the obsolete exposure statement")
    source = source.replace(obsolete, replacement)
    old_mechanism = ("These patterns are consistent with neighbour contamination, with site selection "
                     "and thermal retrieval artefacts not ruled out.")
    new_mechanism = ("The isolation comparison does not distinguish neighbour contamination, "
                     "site selection, or thermal retrieval artefacts.")
    if source.count(old_mechanism) != 1:
        raise SystemExit("Could not safely narrow the mechanism statement")
    source = source.replace(old_mechanism, new_mechanism)
    limit_old = ("The saved pixel-centred export lacks unique external 30 m greened-pixel counts at 90, "
                 "180 and 300 m needed to redefine exposure.")
    limit_new = ("The v8d export supplies unique external 30 m greened-pixel counts, but overlapping focal "
                 "footprints and selected isolated sites prevent causal attribution of the external-pixel coefficient.")
    if source.count(limit_old) != 1:
        raise SystemExit("Could not safely update the limitations statement")
    source = source.replace(limit_old, limit_new)
    source = source.replace("# Jeddah vegetation transitions and summer surface temperature: v8 text",
                            "# Jeddah vegetation transitions and summer surface temperature: v8 follow-up 2")
    anchor = "The common-month June–September sensitivity gives"
    if source.count(anchor) != 1:
        raise SystemExit("Could not locate the new manuscript section")
    b90 = balance.loc[balance.radius_m.eq(90)].iloc[0]
    b180 = balance.loc[balance.radius_m.eq(180)].iloc[0]
    b300 = balance.loc[balance.radius_m.eq(300)].iloc[0]
    rel = {row.exposure: row.relative_spread_range_over_abs_mean for row in relative.itertuples()}
    insert = ("### Saved-data exposure and isolation follow-up\n\n"
              "In the prior three-continuous-covariate calibration, the non-isolated minus isolated gap "
              "persisted at 90 and 180 m. Adding southern-belt membership and the exactly-one-versus-two-pixel "
              "indicator to the same moment calibration gives paired differences of "
              f"{b90.calibrated_minus_isolated_C:+.3f} [{b90.difference_lo_C:+.3f}, {b90.difference_hi_C:+.3f}] "
              f"at 90 m and {b180.calibrated_minus_isolated_C:+.3f} "
              f"[{b180.difference_lo_C:+.3f}, {b180.difference_hi_C:+.3f}] °C per own greened pixel at 180 m. "
              f"At 300 m it is {b300.calibrated_minus_isolated_C:+.3f} "
              f"[{b300.difference_lo_C:+.3f}, {b300.difference_hi_C:+.3f}]. "
              "Intervals use only bootstrap draws with feasible calibration; the feasible shares and weight "
              "effective sample sizes are in [the balance table](../tables/isolation_balance_v8_followup2.csv). "
              "Balancing means, squares and pairwise products of three baseline covariates plus two binary "
              "means does not exclude residual belt or pixel-mix heterogeneity, land use, or unmeasured differences. "
              "The isolated slopes are roughly level at 180 and 300 m after a less negative point estimate "
              "from 90 to 180 m; the three points do not form a monotone gradient. "
              "Rows with fewer than 20 cells are diagnostic only "
              "([within-group table](../tables/isolation_subgroups_v8_followup2.csv)).\n\n"
              "The [joint-exposure table](../tables/joint_exposure_regression_v8_followup2.csv) fits own and "
              "unique external greened pixels simultaneously to the matched temperature-change contrast, "
              "with zero intercept and the unchanged spatial-block bootstrap. These coefficients are conditional "
              "associations, not separate causal effects. In 9/9 cells own dose is constant, so its coefficient "
              "is not a marginal own-pixel effect. Isolation comparisons remain untestable for the 5–8 and 9/9 "
              "classes because almost none has zero external exposure. The positive [relative class-slope "
              "spread](../tables/combined_class_relative_spread_v8_followup2.csv), defined as range divided "
              "by the absolute four-class mean, is "
              f"{rel['own']:.3f} for own pixels, then {rel['combined_90m']:.3f}, "
              f"{rel['combined_180m']:.3f} and {rel['combined_300m']:.3f} for combined exposures. "
              "Absolute spread shrinks mechanically as the counted denominator grows, so neither measure "
              "establishes physical convergence or a resource ranking.\n\n")
    source = source.replace(anchor, insert + anchor)
    OUT_MANUSCRIPT.write_text(source, encoding="utf-8")


def main():
    outputs = (OUT_JOINT, OUT_BALANCE, OUT_SUBGROUP, OUT_RELATIVE, OUT_MANUSCRIPT)
    for path in outputs:
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    print("Loading saved panel and v8d external counts...", flush=True)
    df = load_saved(cfg)
    print("Fitting primary matched contrasts and 2,000 spatial-block draws...", flush=True)
    joint, balance, subgroup, relative = analyze(df, cfg)
    write_manuscript(joint, balance, subgroup, relative)
    for data, path in zip((joint, balance, subgroup, relative), outputs):
        data.to_csv(path, index=False)
    print("Wrote four versioned tables and versioned manuscript", flush=True)


if __name__ == "__main__":
    main()
