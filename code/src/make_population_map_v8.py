"""Inventory table populations from provenance families; never read the blinded key."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tables/table_population_v8.csv"


def population(name: str) -> tuple[str, str]:
    if name == "imagery_sampling_key_v7.csv":
        return "unopened blinded key", "Filename inventoried only; contents not inspected"
    if name == "imagery_review_queue_v7.csv":
        return "unlabelled stratified review queue, 200 cells", "No land-use labels available"
    if name in {"headline_block_v8.csv", "pixel_isolation_by_dose_v8.csv",
                "figure1_population_comparison_v8.csv"}:
        return "primary pre-treatment-only matched panel", "Setting and dose are identified by row"
    if name.startswith("logic_primary_") or name == "followup_control_selection_primary.csv":
        return "primary pre-treatment-only matched panel", "Use row's setting/dose/subset"
    if name.startswith("gee_pixel_centered_") or name.startswith("gee_isolation_within_strata"):
        return "primary matched panel, pixel-screen sensitivity", "Selected isolated sites are not the full class"
    if name.startswith("gee_neighbor_count_gradient") or name.startswith("gee_isolation_dose1"):
        return "primary matched panel, neighbor-screen sensitivity", "Filename marks preserved invalid variants"
    if name.startswith("gee_common_month_") or name == "gee_month_composite_changes.csv":
        return "May–September versus June–September panel", "Row identifies full, shared or paired subset"
    if name.startswith("gee_monthly_") or name == "gee_raw_version_comparison.csv":
        return "all NDVI-classified cells, monthly export", "Filename marks superseded or old-recipe variants"
    if name.startswith("gee_exact_isolation_"):
        return "primary and narrower matched panels, cell-centre screen", "Use row specification; INVALID file is preserved, not evidence"
    if name.startswith("sio_") or name == "logic_sio_integrity.csv":
        return "agricultural SIO branch-years, not greening cells", "Original join checked in v8 audit"
    if name in {"water.csv", "energy.csv", "structural_ratio_sensitivity.csv"}:
        return "hypothetical resource factors or narrower-sensitivity conversion", "Not measured irrigation or net energy"
    if name.startswith("decision_") or name == "allocation_selected.csv":
        return "hypothetical scenario on narrower matched panel or optional allocation", "No municipal ranking established"
    if name.startswith("model_"):
        return "held-out non-greened model sample", "Absolute-LST skill; not effect validation"
    if name.startswith("test_"):
        return "concurrent-change matched sensitivity: 744 outside / 310 built-up", "Model support not computed for primary population"
    if name.startswith("measured_") or name in {"coastal_block_counts.csv", "population_retention.csv"}:
        return "concurrent-change matched sensitivity, with all-classified counts where labelled", "Read n_treated versus n_treated_matched columns"
    if name.startswith("reviewer_") or name.startswith("followup_") or name.startswith("logic_block_"):
        return "mixed matching sensitivities", "Read specification, setting and subset columns; not interchangeable"
    if name in {"cells.csv", "logic_panel_proximity_proxy.csv"}:
        return "exported cell panel or primary proximity subset", "Read class and matching fields"
    raise ValueError(f"Population not assigned: {name}")


def main() -> None:
    names = sorted(p.name for p in (ROOT / "tables").glob("*.csv") if p.name != OUT.name)
    rows = [{"table": name, "population": population(name)[0], "interpretation": population(name)[1]}
            for name in names]
    rows.append({"table": OUT.name, "population": "all table filenames in v8",
                 "interpretation": "Inventory metadata only; blinded key not opened"})
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
        if old != rows:
            raise SystemExit("Existing population map differs; version a replacement")
    else:
        with OUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["table", "population", "interpretation"])
            writer.writeheader()
            writer.writerows(rows)
    print(f"Population map covers {len(rows)} CSVs; key contents not read")


if __name__ == "__main__":
    main()
