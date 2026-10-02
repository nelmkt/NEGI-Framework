"""Create a versioned manuscript and supplementary spread note from saved tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP2.md"
TARGET = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP3.md"
SUPPLEMENT = ROOT / "manuscript/SUPPLEMENT_RELATIVE_SPREAD_V8_FOLLOWUP3.md"
TABLES = ROOT / "tables"


def find(df, **fields):
    subset = df
    for key, value in fields.items():
        subset = subset[subset[key].eq(value)]
    if len(subset) != 1:
        raise ValueError(f"Expected exactly one table row for {fields}")
    return subset.iloc[0]


def estimate(row, field="beta_C_per_pixel", lo="lo_C", hi="hi_C"):
    return f"{row[field]:+.3f} [{row[lo]:+.3f}, {row[hi]:+.3f}]"


def between(text, start, end, replacement):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Could not safely replace section between {start!r} and {end!r}")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + replacement + end + after


def main():
    for path in (TARGET, SUPPLEMENT):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite earlier manuscript output: {path}")
    ring = pd.read_csv(TABLES / "ring_exposure_regression_v8_followup3.csv")
    col = pd.read_csv(TABLES / "joint_exposure_collinearity_v8_followup3.csv")
    sub = pd.read_csv(TABLES / "isolation_subgroups_v8_followup3.csv")
    balance = pd.read_csv(TABLES / "isolation_balance_v8_followup2.csv")
    rel = pd.read_csv(TABLES / "combined_class_relative_spread_v8_followup2.csv")
    source = SOURCE.read_text(encoding="utf-8")

    small_own = find(ring, group="1–2", term="own")
    all_own = find(ring, group="all outside", term="own")
    all_near = find(ring, group="all outside", term="external_0_90m")
    all_mid = find(ring, group="all outside", term="external_90_180m")
    all_far = find(ring, group="all outside", term="external_180_300m")
    b90, b180, b300 = [find(balance, radius_m=r) for r in (90, 180, 300)]
    sb = find(sub, radius_m=90, subgroup="southern belt")
    nb = find(sub, radius_m=90, subgroup="outside belt")
    c90 = find(col, radius_m=90, group="all outside")
    high = find(col, radius_m=90, group="9/9")

    abstract = ("We compare Landsat-derived summer daytime land-surface temperature (LST) changes after "
                "NDVI-defined vegetation transitions in Jeddah with matched, never-vegetated controls. "
                "In the primary outside-built-up population, the own-pixel coefficient for 1–2-pixel cells "
                f"is {small_own.beta_C_per_pixel:+.3f} °C/pixel when non-overlapping neighbouring rings "
                "are included; the selected isolated 1–2-pixel slopes are −1.136, −0.879 and −0.891 "
                "°C per own pixel at 90, 180 and 300 m. The 90 and 180 m isolation gaps remain after "
                "balancing observed baseline temperature, coast distance, built fraction, southern-belt "
                "membership and one-versus-two-pixel dose. The gaps attenuate in point estimates; the "
                "isolated slopes level off from 180 to 300 m. At 90 m the gap is larger in the southern "
                "belt, while isolated belt samples at longer distances are diagnostic only. Stronger "
                "near-ring associations are consistent with thermal-pixel blur or edge effects, but site "
                "selection and other mechanisms remain possible. The pooled −1.181 °C/pixel slope is a "
                "dose-weighted description of the sampled contiguous greening, with 55.7% of its leverage "
                "in one spatial block; it is not the marginal effect of a new small patch. The high-dose "
                "own-versus-external causal split is unidentified, and own dose does not vary at all among "
                "9/9 cells. Irrigation identity remains unverified; water and energy quantities are "
                "hypothetical scenarios, not investment rankings. These LST contrasts do not measure "
                "air temperature, thermal comfort or population benefit.\n")
    source = between(source, "## Abstract\n\n", "\n## Design and population", abstract)
    source = source.replace("# Jeddah vegetation transitions and summer surface temperature: v8 follow-up 2",
                            "# Jeddah vegetation transitions and summer surface temperature: v8 follow-up 3")

    result = ("\nThe [non-overlapping ring regression](../tables/ring_exposure_regression_v8_followup3.csv) "
              "fits own pixels and external pixels at 0–90, 90–180 and 180–300 m simultaneously to each "
              "primary matched cell's temperature-change contrast, with zero intercept and the unchanged "
              "spatial-block bootstrap. For all 1,006 outside cells, own and the three ring coefficients "
              f"are {estimate(all_own)}, {estimate(all_near)}, {estimate(all_mid)}, and "
              f"{estimate(all_far)} °C per counted pixel. Among 419 small-patch cells, the own coefficient "
              f"is {estimate(small_own)}. The nearer external coefficients are more negative in point "
              "estimates; this is consistent with thermal-pixel blur or edge effects but does not identify "
              "either mechanism. External pixels can be counted in multiple focal cells, and the "
              "regressors inherit site selection. The southern-belt regression uses 636 cells in only "
              "eight blocks; the non-belt regression uses 370 cells in 26 blocks.\n\n"
              "The [collinearity audit](../tables/joint_exposure_collinearity_v8_followup3.csv) adds the "
              "uncentred own/external cosine, condition number of the raw two-column design, normalized "
              "condition number, and zero-external share to every earlier joint regression. All 15 "
              "designs have algebraic rank two, but this does not identify separate causal own and "
              "neighbour effects. For all outside cells at 90 m, the uncentred cosine is "
              f"{c90.uncentred_cosine:.3f} and the raw-design condition number is "
              f"{c90.design_condition_number:.1f}; for 9/9 cells they are "
              f"{high.uncentred_cosine:.3f} and {high.design_condition_number:.1f}. "
              "The 9/9 own count is fixed at nine, so its reported zero-intercept coefficient is a "
              "conditional level term, not an identified marginal own-pixel response. The high-dose "
              "causal own/external split is not identified.\n\n"
              "The 1–2-pixel isolated slopes are −1.136, −0.879 and −0.891 °C per own pixel at 90, "
              "180 and 300 m. They level off from 180 to 300 m rather than trace a monotone gradient. "
              "After expanded observed-covariate balancing, non-isolated minus isolated differences "
              f"are {b90.calibrated_minus_isolated_C:+.3f} "
              f"[{b90.difference_lo_C:+.3f}, {b90.difference_hi_C:+.3f}], "
              f"{b180.calibrated_minus_isolated_C:+.3f} "
              f"[{b180.difference_lo_C:+.3f}, {b180.difference_hi_C:+.3f}], and "
              f"{b300.calibrated_minus_isolated_C:+.3f} "
              f"[{b300.difference_lo_C:+.3f}, {b300.difference_hi_C:+.3f}] °C per own pixel. "
              "The point gaps attenuate with radius; the 300 m interval includes zero. The intervals "
              "condition on feasible calibration draws and do not prove bootstrap coverage. At 90 m "
              f"the unweighted belt gap is {sb.nonisolated_minus_isolated_C:+.3f} on "
              f"{int(sb.isolated_cells)} isolated and {int(sb.nonisolated_cells)} non-isolated cells, "
              f"versus {nb.nonisolated_minus_isolated_C:+.3f} on "
              f"{int(nb.isolated_cells)} and {int(nb.nonisolated_cells)} outside the belt. "
              "Thus the 90 m gap is larger in the belt; concentration across all radii is untested "
              "because only 13 and seven isolated belt cells remain at 180 and 300 m. "
              "[The subgroup table](../tables/isolation_subgroups_v8_followup3.csv) gives every "
              "belt/non-belt and exactly-one/exactly-two count and slope. Exactly-two isolated cells "
              "number 13, seven and four at 90, 180 and 300 m; all those rows are diagnostic only, "
              "as is any other row below 20 cells. Belt, pixel mix, land use and unmeasured differences "
              "remain possible explanations despite balancing selected observed moments. High-dose "
              "isolation comparisons are untestable.\n\n"
              "The relative-spread convergence calculation has been moved to "
              "[the supplement](SUPPLEMENT_RELATIVE_SPREAD_V8_FOLLOWUP3.md). Its shrinking absolute "
              "class spread is partly mechanical because the combined-exposure denominator grows; "
              "it is not evidence of a physical convergence or a water ranking.\n\n")
    source = between(source, "### Saved-data exposure and isolation follow-up\n\n",
                     "The common-month June–September sensitivity gives", result)
    TARGET.write_text(source, encoding="utf-8")

    lines = ["# Supplement: descriptive class-slope convergence", "",
             "The earlier relative-spread analysis is supplementary. It divides the range of four "
             "class slopes by the absolute value of their equally weighted mean. Because external-pixel "
             "counts enlarge the denominator, absolute slope spread shrinks partly mechanically. "
             "An external pixel can appear in several focal-cell exposures. Neither relative nor "
             "absolute convergence identifies spillover, an independent planted-pixel response, or a "
             "water/energy ranking.", "",
             "| Exposure | Relative spread [95% spatial-block interval] |", "|---|---:|"]
    for row in rel.itertuples():
        lines.append(f"| {row.exposure} | {row.relative_spread_range_over_abs_mean:.3f} "
                     f"[{row.lo:.3f}, {row.hi:.3f}] |")
    lines.extend(["", "Source: [relative-spread table](../tables/combined_class_relative_spread_v8_followup2.csv) "
                  "and the [saved-data script](../code/src/analyze_joint_exposure_v8_followup2.py).", ""])
    SUPPLEMENT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {TARGET.name} and {SUPPLEMENT.name}")


if __name__ == "__main__":
    main()
