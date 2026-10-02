"""Versioned manuscript update with paired inner-subring comparisons."""
from __future__ import annotations

import pandas as pd

from analyze_isolation_reweighting_v8 import ROOT

SOURCE = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP7.md"
TARGET = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP8.md"
TABLE = ROOT / "tables/pixel_subring_pair_differences_v8_followup7.csv"


def main():
    if TARGET.exists():
        raise SystemExit(f"Refusing to overwrite {TARGET}")
    table = pd.read_csv(TABLE)
    if len(table) != 3 or not table.n_cells.eq(419).all() or not table.n_blocks.eq(32).all():
        raise SystemExit("Paired subring comparison population changed")
    if not table.valid_paired_block_draws.eq(2000).all():
        raise SystemExit("A paired subring interval lacks valid draws")
    if not ((table.lo_C < 0) & (table.hi_C > 0)).all():
        raise SystemExit("At least one within-90 m paired interval no longer includes zero")
    r = table.loc[table.term_a.eq("external_0_30m") &
                  table.term_b.eq("external_30_60m")]
    if len(r) != 1:
        raise SystemExit("First-to-second subring contrast missing")
    r = r.iloc[0]
    text = SOURCE.read_text(encoding="utf-8").replace("v8 follow-up 7", "v8 follow-up 8", 1)
    anchor = "All three inner-ring intervals include zero; their point estimates do not show a sharp first-pixel drop."
    if text.count(anchor) != 1:
        raise SystemExit("Subring interpretation anchor missing")
    new = (anchor + " A direct paired comparison gives 0–30 minus 30–60 m "
           f"{r.beta_a_minus_beta_b_C_per_pixel:+.3f} "
           f"[{r.lo_C:+.3f}, {r.hi_C:+.3f}] °C/pixel; "
           "all three pairwise inner-ring difference intervals include zero "
           "([paired differences](../tables/pixel_subring_pair_differences_v8_followup7.csv)). "
           "These wide intervals do not establish equality; they show this sample cannot "
           "resolve a sharper first-pixel decline.")
    text = text.replace(anchor, new)
    TARGET.write_text(text, encoding="utf-8")
    print(f"Wrote {TARGET.name}")


if __name__ == "__main__":
    main()
