"""Create a versioned manuscript with saved-data ring robustness limits."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP3.md"
TARGET = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP4.md"
PAIR = ROOT / "tables/ring_pair_collinearity_v8_followup4.csv"
SUMMARY = ROOT / "tables/ring_leave_one_block_summary_v8_followup4.csv"
SUBGROUP = ROOT / "tables/isolation_subgroups_v8_followup3.csv"


def one(frame: pd.DataFrame, **fields):
    part = frame
    for col, value in fields.items():
        part = part[part[col].eq(value)]
    if len(part) != 1:
        raise ValueError(f"Expected exactly one row: {fields}")
    return part.iloc[0]


def main():
    if TARGET.exists():
        raise SystemExit(f"Refusing to overwrite earlier manuscript: {TARGET}")
    pairs = pd.read_csv(PAIR)
    summary = pd.read_csv(SUMMARY)
    sub = pd.read_csv(SUBGROUP)
    text = SOURCE.read_text(encoding="utf-8")
    small = pairs.loc[pairs.group.eq("1–2")]
    if len(small) != 3 or len(summary) != 1:
        raise SystemExit("Saved ring follow-up tables have unexpected rows")
    s = summary.iloc[0]
    one_gap = [one(sub, radius_m=r, subgroup="exactly one").nonisolated_minus_isolated_C
               for r in (90, 180, 300)]
    text = text.replace("# Jeddah vegetation transitions and summer surface temperature: v8 follow-up 3",
                        "# Jeddah vegetation transitions and summer surface temperature: v8 follow-up 4", 1)
    anchor = ("The southern-belt regression uses 636 cells in only eight blocks; "
              "the non-belt regression uses 370 cells in 26 blocks.\n\n")
    if text.count(anchor) != 1:
        raise SystemExit("Could not locate the ring result paragraph")
    addition = ("The three external-ring counts are not statistically independent despite their "
                "non-overlapping geometry. Among 1–2-pixel cells, their pairwise uncentred cosines "
                "are " + ", ".join(f"{v:.3f}" for v in small.uncentred_cosine) +
                " (0–90 with 90–180, 0–90 with 180–300, then 90–180 with 180–300 m)"
                +
                f"; the three-column normalized design condition number is "
                f"{small.external_three_condition_normalized.iloc[0]:.2f} "
                "([ring collinearity table](../tables/ring_pair_collinearity_v8_followup4.csv)). "
                "A leave-one-treated-block-out calculation refits the matched control contrast after "
                "removing each block. Across "
                f"{int(s.n_deleted_block_runs)} deletions, the 1–2-pixel 0–90 m coefficient ranges "
                f"from {s.minimum_leave_one_out_beta_C_per_pixel:+.3f} to "
                f"{s.maximum_leave_one_out_beta_C_per_pixel:+.3f} °C/pixel, versus "
                f"{s.full_beta_external_0_90m_C_per_pixel:+.3f} with all blocks; "
                f"{int(s.sign_reversals)} deletions reverse the sign. The largest absolute change "
                f"is {s.maximum_absolute_change_C_per_pixel:.3f} °C/pixel. "
                "This checks one-block leverage for that coefficient, not the coverage of its "
                "bootstrap interval or confounding by several blocks. The belt has only eight "
                "blocks and several percentile intervals are asymmetric; an estimate close to an "
                "interval endpoint alone does not prove bootstrap bias or valid coverage "
                "([all deletions](../tables/ring_leave_one_block_out_v8_followup4.csv); "
                "[summary](../tables/ring_leave_one_block_summary_v8_followup4.csv)).\n\n")
    text = text.replace(anchor, anchor + addition)
    contrast_anchor = ("They level off from 180 to 300 m rather than trace a monotone gradient. "
                       "After expanded observed-covariate balancing")
    replacement = ("They level off from 180 to 300 m rather than trace a monotone gradient. "
                   "Within exactly-one-pixel cells, the unweighted non-isolated-minus-isolated "
                   "gaps at 90, 180 and 300 m are " + ", ".join(f"{v:+.3f}" for v in one_gap) +
                   " °C per own pixel. Their near-flat point pattern coexists with the negative "
                   "90–180 and 180–300 m ring coefficients: isolation and joint ring regression "
                   "compare different selected populations and estimands. Neither supports a claim "
                   "of zero association beyond 90 m. After expanded observed-covariate balancing")
    if text.count(contrast_anchor) != 1:
        raise SystemExit("Could not locate the isolation-distance sentence")
    text = text.replace(contrast_anchor, replacement)
    limit_anchor = ("The high-dose causal own/external split is not identified.\n\n")
    limit_add = ("The saved v8d export has only cumulative unique-external counts at 90, 180 and "
                 "300 m. It has no unique counts at 30 or 60 m, so 0–30, 30–60 and 60–90 m "
                 "regressors cannot be recovered from this panel or approximated by cell-centre "
                 "distances. That finer test requires a new pixel-centred export with the same "
                 "greening definition, grid and unique-pixel rule.\n\n")
    if text.count(limit_anchor) != 1:
        raise SystemExit("Could not locate the spatial-resolution limit")
    text = text.replace(limit_anchor, limit_anchor + limit_add)
    TARGET.write_text(text, encoding="utf-8")
    print(f"Wrote {TARGET.name}")


if __name__ == "__main__":
    main()
