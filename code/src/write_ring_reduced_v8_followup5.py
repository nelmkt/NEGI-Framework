"""Create a versioned manuscript and validation note for reduced ring models."""
from __future__ import annotations

import pandas as pd

from analyze_isolation_reweighting_v8 import ROOT
from analyze_ring_reduced_v8_followup5 import OUT_COEFFICIENTS, OUT_COMPARISONS

SOURCE = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP4.md"
TARGET = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP5.md"
VALIDATION = ROOT / "manuscript/VALIDATION_V8_FOLLOWUP5.md"


def main():
    if TARGET.exists() or VALIDATION.exists():
        raise SystemExit("Refusing to overwrite prior follow-up 5 output")
    c = pd.read_csv(OUT_COEFFICIENTS).set_index(["model", "term"])
    d = pd.read_csv(OUT_COMPARISONS).set_index(["model", "term"])
    needed = [(m, t) for m, terms in {
        "full_four_terms": ("own", "external_0_90m", "external_90_180m", "external_180_300m"),
        "drop_180_300m": ("own", "external_0_90m", "external_90_180m"),
        "merge_90_300m": ("own", "external_0_90m", "external_90_300m"),
    }.items() for t in terms]
    if set(c.index) != set(needed) or len(d) != 4:
        raise SystemExit("Reduced-ring tables do not have the expected model terms")
    if not c.n_cells.eq(419).all() or not c.n_blocks.eq(32).all():
        raise SystemExit("Reduced-ring population differs from the 419-cell primary subset")

    old = SOURCE.read_text(encoding="utf-8")
    old = old.replace("v8 follow-up 4", "v8 follow-up 5", 1)
    old = old.replace(
        "Stronger near-ring associations are consistent with thermal-pixel blur or edge effects, but site selection and other mechanisms remain possible.",
        "External-pixel associations are model dependent and consistent with thermal-pixel blur or edge effects, but site selection and other mechanisms remain possible.")
    old = old.replace(
        "The nearer external coefficients are more negative in point estimates; this is consistent with thermal-pixel blur or edge effects but does not identify either mechanism.",
        "In the full four-term fit, nearer external coefficients are more negative in point estimates; reduced models below test how strongly that ordering depends on the correlated rings. This association does not identify thermal-pixel blur or edge effects.")
    anchor = "The [collinearity audit](../tables/joint_exposure_collinearity_v8_followup3.csv)"
    if old.count(anchor) != 1:
        raise SystemExit("Manuscript insertion anchor not found exactly once")
    a = lambda model, term: float(c.loc[(model, term), "beta_C_per_pixel"])
    delta = lambda model, term: float(d.loc[(model, term), "reduced_minus_full_C_per_pixel"])
    paragraph = (
        "A [paired reduced-model check](../tables/ring_reduced_coefficients_v8_followup5.csv) "
        "refits the same 419 outside 1–2-pixel cells and matched controls on the same 2,000 "
        "spatial-block draws. Removing the 180–300 m term changes the own coefficient from "
        f"{a('full_four_terms','own'):.3f} to {a('drop_180_300m','own'):.3f} and the 0–90 m "
        f"coefficient from {a('full_four_terms','external_0_90m'):.3f} to "
        f"{a('drop_180_300m','external_0_90m'):.3f} °C/pixel. The remaining 90–180 m "
        f"coefficient becomes {a('drop_180_300m','external_90_180m'):.3f}, slightly more "
        "negative than 0–90 m: the monotone ring ordering is **not** stable to dropping the "
        "outermost term. Merging the two outer rings into one 90–300 m count gives own "
        f"{a('merge_90_300m','own'):.3f}, 0–90 m "
        f"{a('merge_90_300m','external_0_90m'):.3f}, and merged outer "
        f"{a('merge_90_300m','external_90_300m'):.3f} °C/pixel. The own changes are "
        f"{delta('drop_180_300m','own'):+.3f} and {delta('merge_90_300m','own'):+.3f}; "
        f"the near-ring changes are {delta('drop_180_300m','external_0_90m'):+.3f} and "
        f"{delta('merge_90_300m','external_0_90m'):+.3f} °C/pixel "
        "([paired differences and intervals](../tables/ring_reduced_comparisons_v8_followup5.csv)). "
        "Thus own-pixel magnitude changes little, while the allocation of external association "
        "among correlated rings is model dependent. Neither the ring ordering nor a particular "
        "ring coefficient identifies a physical mechanism.\n\n"
    )
    old = old.replace(anchor, paragraph + anchor, 1)
    old = old.replace(
        "A leave-one-treated-block-out calculation refits the matched control contrast after removing each block.",
        "A leave-one-treated-block-out calculation tests sensitivity to individual blocks by refitting the matched control contrast after each deletion; it does not validate interval coverage.")
    old = old.replace(
        "They level off from 180 to 300 m rather than trace a monotone gradient.",
        "Their point estimates level off from 180 to 300 m rather than trace a monotone gradient; the smaller isolated samples limit power to distinguish the latter two distances.")
    old = old.replace(
        "Their near-flat point pattern coexists with the negative 90–180 and 180–300 m ring coefficients:",
        "Their near-flat point pattern is not proof of zero difference across isolation distances: exactly-one isolated counts fall from 63 to 31 and 22 cells, and some exactly-two and belt strata have only 4–13 cells. It coexists with negative farther-ring coefficients because")
    old = old.replace(
        "The saved v8d export has only cumulative unique-external counts at 90, 180 and 300 m. It has no unique counts at 30 or 60 m, so 0–30, 30–60 and 60–90 m regressors cannot be recovered from this panel or approximated by cell-centre distances. That finer test requires a new pixel-centred export with the same greening definition, grid and unique-pixel rule.",
        "The saved v8d export has only cumulative unique-external counts at 90, 180 and 300 m. It has no unique counts at 30 or 60 m, so 0–30, 30–60 and 60–90 m regressors cannot be recovered from this panel or approximated by cell-centre distances. The prepared but unrun [versioned exporter](../code/gee/export_external_pixel_subrings_v8_followup5.py) would use the same native-pixel grid and unique-pixel rule, check that the three subrings sum to the previous 90 m count in every cell, and report zero-count shares. A mostly-zero 0–30 m ring would have little identifying variation; no subring effect is estimated yet.")
    if "The point gaps attenuate with radius" not in old:
        raise SystemExit("Earlier claim unexpectedly absent")
    TARGET.write_text(old, encoding="utf-8")
    VALIDATION.write_text(
        "# Saved-data ring follow-up 5: observed checks\n\n"
        "The saved-panel script reproduced the earlier four-term 1–2-pixel point estimates, "
        "then fit the two reduced specifications on the same 419 cells, 32 treated blocks, "
        "matched controls and 2,000 paired spatial-block draws. All 2,000 draws yielded "
        "valid paired own/near-ring differences in both comparisons. See the versioned "
        "coefficient and difference CSVs for all endpoints. The subring exporter was written "
        "and tested on a synthetic pixel grid, but Earth Engine was not run and no 30/60 m "
        "counts or coefficients were produced. The new table registry covers 116 of 116 table "
        "CSV files, with no missing or duplicate entries; the root still has the five expected "
        "sibling folders.\n\n"
        "This session's targeted unittest command passed 20 tests, including three new "
        "reduced-model/subring tests and 17 earlier saved-data tests. `code/src/audit_claims.py` "
        "passed. An attempted complete unittest discovery ran 25 tests with five import "
        "errors (`test_combined_exposure_v8.py`, `test_core.py`, `test_external_pixel_counts_v8.py`, "
        "`test_followup.py`, `test_framework.py`), caused by missing pytest or scipy. "
        "The full pytest command could not start in the bundled runtime (`No module named pytest`). "
        "The user's Python 3.12 executable exists but could not be launched from this sandbox "
        "(`Access is denied`), including after a read-permission grant. The full suite is "
        "therefore not recorded as passing.\n", encoding="utf-8")
    print(f"Wrote {TARGET.name} and {VALIDATION.name}")


if __name__ == "__main__":
    main()
