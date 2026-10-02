"""Cross-artifact checks for the common-month and monthly-NDVI follow-up."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tables"


def rows(name):
    with (TABLES / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_common_month_rerun_and_population_accounting():
    for stem in ("primary", "paired", "overlap"):
        assert (TABLES / f"gee_common_month_{stem}_v6.csv").read_bytes() == (
            TABLES / f"gee_common_month_{stem}_v2.csv").read_bytes()
    x = rows("gee_common_month_composition_v6.csv")
    total = lambda pred: sum(int(r["n_cells"]) for r in x if pred(r))
    old = total(lambda r: r["group_old"] == "greened")
    new = total(lambda r: r["group_new"] == "greened")
    exits = total(lambda r: r["group_old"] == "greened" and r["group_new"] != "greened")
    entries = total(lambda r: r["group_old"] != "greened" and r["group_new"] == "greened")
    shifts = total(lambda r: r["group_old"] == r["group_new"] == "greened"
                   and r["old_dose"] != r["new_dose"])
    assert (old, new, exits, entries, shifts) == (1333, 1273, 94, 34, 67)
    assert old - exits + entries == new


def test_monthly_rows_have_coverage_and_bounded_shares():
    x = rows("gee_monthly_ndvi_by_month_belt_v6.csv")
    assert len(x) == 150
    assert {(int(r["year"]), int(r["month"])) for r in x} == {
        (y, m) for y in (2024, 2025) for m in (5, 6, 7, 8, 9)}
    for r in x:
        assert int(r["n_adequately_observed"]) == int(r["n_cells"])
        if r["reportable"] == "True":
            assert 0 <= float(r["fraction_observed_green_pixels_ndvi_ge_0_30"]) <= 1
    outside_small = [r for r in x if r["setting"] == "outside the built-up area"
                     and r["dose_class"] == "1–2" and r["year"] == "2024" and r["month"] == "5"]
    assert sum(int(r["n_cells"]) for r in outside_small) == 426


def test_within_stratum_comparison_reports_its_limited_support():
    x = rows("gee_isolation_within_strata_v6.csv")
    assert len(x) == 11
    assert sum(int(r["n_isolated"]) for r in x) == 12
    assert sum(int(r["n_nonisolated"]) for r in x) == 103
    assert all(int(r["n_isolated"]) > 0 and int(r["n_nonisolated"]) > 0
               and int(r["n_controls"]) >= 3 for r in x)
