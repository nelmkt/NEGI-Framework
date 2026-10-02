"""Create a versioned manuscript update from the saved paired middle-ring result."""
from __future__ import annotations

import pandas as pd

from analyze_isolation_reweighting_v8 import ROOT

SOURCE = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP5.md"
TARGET = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP6.md"
COMPARISON = ROOT / "tables/ring_reduced_comparisons_v8_followup6.csv"


def main():
    if TARGET.exists():
        raise SystemExit(f"Refusing to overwrite {TARGET}")
    rows = pd.read_csv(COMPARISON)
    row = rows.loc[rows.model.eq("drop_180_300m") & rows.term.eq("external_90_180m")]
    if len(row) != 1 or int(row.iloc[0].valid_paired_block_draws) != 2000:
        raise SystemExit("Paired middle-ring row is missing or has incomplete block draws")
    r = row.iloc[0]
    text = SOURCE.read_text(encoding="utf-8").replace("v8 follow-up 5", "v8 follow-up 6", 1)
    lead = "the own-pixel coefficient for 1–2-pixel cells is -1.002 °C/pixel when non-overlapping neighbouring rings are included;"
    if text.count(lead) != 1:
        raise SystemExit("Own-coefficient abstract anchor missing")
    text = text.replace(lead,
        "the outside 1–2-pixel own coefficient is about −1 °C/pixel across the full, far-ring-omitted and outer-ring-merged fits (−1.002, −1.036, −0.988);",
        1)
    anchor = "Neither the ring ordering nor a particular ring coefficient identifies a physical mechanism."
    if text.count(anchor) != 1:
        raise SystemExit("Reduced-ring interpretation anchor missing")
    addition = (f" Dropping the far ring changes the 90–180 m coefficient by "
                f"{r.reduced_minus_full_C_per_pixel:+.3f} °C/pixel "
                f"[{r.difference_lo_C:+.3f}, {r.difference_hi_C:+.3f}] on the same "
                "2,000 block draws; this interval excludes zero. Thus its allocation is plainly "
                "specification sensitive, rather than a separately stable middle-ring effect. "
                "Across the three models, all estimated external terms remain negative and "
                "roughly 0.018–0.104 °C per counted pixel in magnitude, compared with about "
                "1 °C per own pixel. These are conditional associations, not causal exchange "
                "rates. The merged 90–300 m model imposes one coefficient on two previously "
                "separate rings, so its decline cannot confirm the finer distance profile "
                "([five paired rows](../tables/ring_reduced_comparisons_v8_followup6.csv)).")
    text = text.replace(anchor, anchor + addition, 1)
    TARGET.write_text(text, encoding="utf-8")
    print(f"Wrote {TARGET.name}")


if __name__ == "__main__":
    main()
