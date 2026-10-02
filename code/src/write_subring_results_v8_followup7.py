"""Create versioned manuscript and provenance notes for observed subring results."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from analyze_isolation_reweighting_v8 import ROOT

SOURCE = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP6.md"
TARGET = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP7.md"
COEF = ROOT / "tables/pixel_subring_regression_v8_followup6.csv"
COLLIN = ROOT / "tables/pixel_subring_collinearity_v8_followup6.csv"
RAW = ROOT / "code/gee/pixel_external_subrings_v8_followup5_raw.csv"
PANEL = ROOT / "code/gee/panel.csv"
PRIOR = ROOT / "code/gee/pixel_external_counts_v8_raw.csv"
EARLIER_RING = ROOT / "tables/ring_exposure_regression_v8_followup3.csv"


def main():
    if TARGET.exists():
        raise SystemExit(f"Refusing to overwrite {TARGET}")
    meta = json.loads(RAW.with_suffix(".json").read_text(encoding="utf-8"))
    if (meta.get("recipe_revision") != "unique-external-native30-subrings-v8-followup5" or
            meta.get("n_exported_cells") != 1333 or any(meta.get("checks", {}).values()) or
            meta.get("source_panel_sha256") != hashlib.sha256(PANEL.read_bytes()).hexdigest() or
            meta.get("source_prior_external_counts_sha256") != hashlib.sha256(PRIOR.read_bytes()).hexdigest()):
        raise SystemExit("New export provenance or zero-violation checks fail")
    c = pd.read_csv(COEF)
    d = pd.read_csv(COLLIN)
    if (len(c) != 11 or len(d) != 10 or not c.n_cells.eq(419).all() or
            not c.n_blocks.eq(32).all() or not c.valid_block_draws.eq(2000).all()):
        raise SystemExit("Subring table populations or bootstrap draws do not reconcile")
    fine = c.loc[c.model.eq("fine_three_inner_rings")].set_index("term")
    coarse = c.loc[c.model.eq("coarse_two_inner_rings")].set_index("term")
    earlier = pd.read_csv(EARLIER_RING)
    earlier_cond = float(earlier.loc[(earlier.group.eq("1–2")) &
                                     earlier.term.eq("own"), "design_condition_number"].iloc[0])
    pairs = d.set_index(["term_a", "term_b"])
    terms = ("external_0_30m", "external_30_60m", "external_60_90m")
    if not all(float(fine.loc[t, "lo_C"]) < 0 < float(fine.loc[t, "hi_C"]) for t in terms):
        raise SystemExit("A fine inner-ring interval no longer includes zero")
    if fine.design_rank.nunique() != 1 or int(fine.design_rank.iloc[0]) != 6:
        raise SystemExit("Fine-ring design is not full rank")

    source = SOURCE.read_text(encoding="utf-8")
    text = source.replace("v8 follow-up 6", "v8 follow-up 7", 1)
    old = ("The saved v8d export has only cumulative unique-external counts at 90, 180 and 300 m. "
           "It has no unique counts at 30 or 60 m, so 0–30, 30–60 and 60–90 m regressors cannot "
           "be recovered from this panel or approximated by cell-centre distances. The prepared "
           "but unrun [versioned exporter](../code/gee/export_external_pixel_subrings_v8_followup5.py) "
           "would use the same native-pixel grid and unique-pixel rule, check that the three "
           "subrings sum to the previous 90 m count in every cell, and report zero-count shares. "
           "A mostly-zero 0–30 m ring would have little identifying variation; no subring effect is estimated yet.")
    if text.count(old) != 1:
        raise SystemExit("Unrun-export manuscript text not found exactly once")
    def fmt(term):
        r = fine.loc[term]
        return f"{r.beta_C_per_pixel:.3f} [{r.lo_C:.3f}, {r.hi_C:+.3f}]"
    def cosine(a, b):
        return float(pairs.loc[(a, b), "uncentred_cosine"])
    paragraph = (
        "The [pixel-centred 30/60 m export](../code/gee/pixel_external_subrings_v8_followup5_raw.csv) "
        "returned 1,333 classified greened cells. Its saved metadata reports zero violations "
        "for cell IDs, own-pixel counts, monotonic radii, the old isolation flags, and the "
        "previous 90/180/300 m counts; the three inner rings sum to the prior 90 m count "
        "in every cell. Those checks establish consistency of this export, not physical "
        "validity of the LST retrieval or a causal mechanism. Among the 419 primary matched "
        "outside 1–2-pixel cells, the [fine joint model](../tables/pixel_subring_regression_v8_followup6.csv) "
        f"estimates own {fine.loc['own','beta_C_per_pixel']:.3f} °C/pixel and external "
        f"0–30 m {fmt('external_0_30m')}, 30–60 m {fmt('external_30_60m')}, "
        f"60–90 m {fmt('external_60_90m')} °C per counted external pixel. "
        "All three inner-ring intervals include zero; their point estimates do not show "
        "a sharp first-pixel drop. The same cells have zero external counts in "
        f"{100*fine.loc['external_0_30m','zero_count_share']:.1f}%, "
        f"{100*fine.loc['external_30_60m','zero_count_share']:.1f}% and "
        f"{100*fine.loc['external_60_90m','zero_count_share']:.1f}% of those subrings, "
        "respectively. That is appreciable variation, but the counts are highly correlated: "
        f"pairwise uncentred cosines are {cosine('external_0_30m','external_30_60m'):.3f} "
        f"(0–30 with 30–60), {cosine('external_0_30m','external_60_90m'):.3f} "
        f"(0–30 with 60–90), and {cosine('external_30_60m','external_60_90m'):.3f} "
        "(30–60 with 60–90 m) "
        "([all pairs](../tables/pixel_subring_collinearity_v8_followup6.csv)). The six-column "
        f"design has full algebraic rank but condition number {float(fine.iloc[0].design_condition):.1f}, "
        f"versus {earlier_cond:.1f} for the earlier four-term design. "
        "Coarsening to 0–30/30–90 m gives own "
        f"{coarse.loc['own','beta_C_per_pixel']:.3f}, inner "
        f"{coarse.loc['external_0_30m','beta_C_per_pixel']:.3f} and outer-inner "
        f"{coarse.loc['external_30_90m','beta_C_per_pixel']:.3f} °C/pixel; both "
        "external inner-ring intervals still include zero. Thus the own-pixel association "
        "remains about −1 °C/pixel, while the subring data cannot distinguish thermal-pixel "
        "blur from edge or neighbourhood effects, or locate a sharp decay within 90 m. "
        "None of these external coefficients is a causal spillover estimate."
    )
    text = text.replace(old, paragraph)
    abstract_old = "External-pixel associations are model dependent and consistent with thermal-pixel blur or edge effects, but site selection and other mechanisms remain possible."
    abstract_new = ("External-pixel associations are model dependent; a new 30/60 m split "
                    "does not resolve a within-90 m gradient. Thermal-pixel blur, edge "
                    "effects, site selection and other mechanisms remain possible.")
    if text.count(abstract_old) != 1:
        raise SystemExit("Abstract sentence not found exactly once")
    text = text.replace(abstract_old, abstract_new)
    TARGET.write_text(text, encoding="utf-8")
    print(f"Wrote {TARGET.name}")


if __name__ == "__main__":
    main()
