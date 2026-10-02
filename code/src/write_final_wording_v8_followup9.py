"""Versioned final wording pass; no scientific output or estimate changes."""
from __future__ import annotations

from analyze_isolation_reweighting_v8 import ROOT

SOURCE = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP8.md"
TARGET = ROOT / "manuscript/MANUSCRIPT_V8_FOLLOWUP9.md"


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise SystemExit(f"Expected one wording anchor, found {text.count(before)}: {before[:80]}")
    return text.replace(before, after, 1)


def main():
    if TARGET.exists():
        raise SystemExit(f"Refusing to overwrite {TARGET}")
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_once(text, "v8 follow-up 8", "v8 follow-up 9")

    start = text.index("## Abstract\n\n") + len("## Abstract\n\n")
    end = text.index("\n\n## Design and population", start)
    abstract = (
        "We compare Landsat-derived summer daytime land-surface temperature (LST) changes "
        "after NDVI-defined vegetation transitions in Jeddah with matched, never-vegetated "
        "controls. Among 419 primary matched outside-built-up cells with 1–2 greened pixels, "
        "the own-only slope is −1.925 °C per greened pixel, while joint regressions that "
        "include neighbouring greened-pixel counts give an own coefficient near −1 °C/pixel "
        "across four ring parameterisations, including −1.014 [−1.226, −0.761] in the "
        "fine-ring model. These are distinct descriptive estimands. Selected isolated cells "
        "show less negative own-only slopes; the 90 m gap is larger in the southern belt, "
        "and isolated samples shrink at longer distances. The 30/60 m split does **not "
        "resolve** a within-90 m gradient: every inner-ring interval and every paired "
        "inner-ring difference interval includes zero. Strong correlation among ring counts "
        "prevents a reliable allocation of the external association by distance. Thermal-pixel "
        "blur, edge effects, neighbourhood confounding and site selection remain possible. "
        "The pooled −1.181 °C/pixel slope describes sampled contiguous greening, with 55.7% "
        "of its dose-squared leverage in one block; it is not the marginal effect of a new "
        "small patch. Irrigation identity is unverified, and the water and energy figures are "
        "hypothetical scenarios rather than investment rankings. These LST contrasts do not "
        "measure air temperature, thermal comfort or population benefit."
    )
    text = text[:start] + abstract + text[end:]

    old = ("The [non-overlapping ring regression](../tables/ring_exposure_regression_v8_followup3.csv) "
           "fits own pixels and external pixels at 0–90, 90–180 and 180–300 m simultaneously "
           "to each primary matched cell's temperature-change contrast, with zero intercept "
           "and the unchanged spatial-block bootstrap. For all 1,006 outside cells, own and "
           "the three ring coefficients are -0.603 [-0.661, -0.439], -0.068 [-0.116, -0.053], "
           "-0.044 [-0.065, -0.030], and -0.005 [-0.042, -0.002] °C per counted pixel. "
           "Among 419 small-patch cells, the own coefficient is -1.002 [-1.172, -0.783]. "
           "In the full four-term fit, nearer external coefficients are more negative in point "
           "estimates; reduced models below test how strongly that ordering depends on the "
           "correlated rings. This association does not identify thermal-pixel blur or edge "
           "effects. External pixels can be counted in multiple focal cells, and the regressors "
           "inherit site selection. The southern-belt regression uses 636 cells in only eight "
           "blocks; the non-belt regression uses 370 cells in 26 blocks.")
    new = ("The [non-overlapping ring regression](../tables/ring_exposure_regression_v8_followup3.csv) "
           "fits own pixels and external counts at 0–90, 90–180 and 180–300 m jointly to "
           "the matched temperature-change contrast, with zero intercept and the unchanged "
           "spatial-block bootstrap. The all-outside fit uses 1,006 treated cells across dose "
           "classes; its own coefficient is −0.603 [−0.661, −0.439] °C/pixel. The separate "
           "1–2-pixel fit uses 419 treated cells and gives own −1.002 [−1.172, −0.783] "
           "°C/pixel. Their dose compositions differ, so the coefficients should not be "
           "compared as an effect change. External terms have negative point estimates in "
           "these fits, but their allocation among correlated rings is specification "
           "sensitive; they do not identify thermal blur, edge effects or causal spillover. "
           "External pixels can enter multiple focal-cell records, and site selection "
           "remains possible. The southern-belt fit uses 636 cells in only eight blocks; "
           "the non-belt fit uses 370 cells in 26 blocks.")
    text = replace_once(text, old, new)

    start = text.index("The [pixel-centred 30/60 m export]")
    end = text.index("\n\nThe 1–2-pixel isolated slopes", start)
    subring = (
        "The [pixel-centred 30/60 m export](../code/gee/pixel_external_subrings_v8_followup5_raw.csv) "
        "returned 1,333 classified greened cells and zero recorded discrepancies for IDs, "
        "own counts, nested radii, prior 90/180/300 m counts and the old isolation flags. "
        "The three inner rings sum to the previous 90 m count for every cell. These are "
        "pipeline-consistency checks, not validation of LST retrieval or a mechanism. "
        "Among 419 primary matched outside 1–2-pixel cells, the [six-term joint fit]"
        "(../tables/pixel_subring_regression_v8_followup6.csv) gives an own coefficient "
        "of −1.014 [−1.226, −0.761] °C/pixel. All three inner-ring point estimates are "
        "negative, but each interval includes zero; their individual values remain in the "
        "table rather than being presented as a distance profile. The 0–30 m point estimate "
        "is the least negative, opposite the prespecified blur-like ordering, but the "
        "0–30 minus 30–60 m paired difference is +0.061 [−0.377, +0.445] °C/pixel and "
        "all three paired inner-ring intervals include zero ([paired table]"
        "(../tables/pixel_subring_pair_differences_v8_followup7.csv)). The data therefore "
        "do **not resolve** the within-90 m ordering; they neither establish nor rule out "
        "a sharp first-pixel decline. Zero external counts occur in 29–37% of these matched "
        "cells across the three inner rings, and pairwise uncentred cosines range from "
        "0.776 to 0.879 ([collinearity table]"
        "(../tables/pixel_subring_collinearity_v8_followup6.csv)). The six-column design "
        "has full algebraic rank but condition number 71.0, compared with 38.4 for the "
        "earlier four-term design. A coarser 0–30/30–90 m fit leaves the own coefficient "
        "near −1 and both inner external intervals spanning zero. The blur-versus-edge "
        "question remains open at this sample size, including the possibility of "
        "neighbourhood confounding; none of these external coefficients is a causal "
        "spillover estimate."
    )
    text = text[:start] + subring + text[end:]

    text = replace_once(
        text,
        "The 1–2-pixel isolated slopes are −1.136, −0.879 and −0.891 °C per own pixel at 90, 180 and 300 m. Their point estimates level off from 180 to 300 m rather than trace a monotone gradient; the smaller isolated samples limit power to distinguish the latter two distances.",
        "The 1–2-pixel isolated slopes are −1.136, −0.879 and −0.891 °C per own pixel at 90, 180 and 300 m. These selected samples shrink with radius, so the apparent plateau after 180 m is not a detected absence of further change.")
    text = replace_once(
        text,
        "The blinded imagery queue remains unlabelled; no land-use inference is drawn from it.",
        "The blinded imagery queue remains unlabelled; no land-use inference is drawn from it. Resolving blur versus edge and neighbourhood effects would require more isolated small patches or an independent site with comparable treatment histories and LST retrieval.")

    TARGET.write_text(text, encoding="utf-8")
    print(f"Wrote {TARGET.name}; tables and figures unchanged")


if __name__ == "__main__":
    main()
