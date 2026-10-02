"""Inventory saved-data reduced-ring tables without modifying prior registries."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "tables/table_population_v8_followup4.csv"
NEW = ROOT / "tables/table_population_v8_followup5.csv"
ROWS = {
    "ring_reduced_coefficients_v8_followup5.csv": (
        "primary outside matched 1–2-pixel treated cells (419, 32 blocks)",
        "Full, far-ring-omitted and merged-outer-ring zero-intercept fits; paired block draws"),
    "ring_reduced_comparisons_v8_followup5.csv": (
        "same 419 cells and 2,000 spatial-block draws",
        "Paired own/near-ring changes relative to the full four-term model"),
    "table_population_v8_followup5.csv": (
        "all table CSV filenames through follow-up 5",
        "Versioned table inventory; unrun subring export has no result table"),
}


def main():
    if NEW.exists():
        raise SystemExit(f"Refusing to overwrite {NEW}")
    prior = pd.read_csv(OLD)
    filenames = {p.name for p in (ROOT / "tables").glob("*.csv")} | {NEW.name}
    if filenames - set(prior.table) != set(ROWS):
        raise SystemExit("Unclassified new table filenames")
    extra = pd.DataFrame([dict(table=name, population=population, interpretation=interpretation)
                          for name, (population, interpretation) in ROWS.items()])
    current = pd.concat([prior, extra], ignore_index=True)
    if current.table.duplicated().any() or set(current.table) != filenames:
        raise SystemExit("Table inventory is not one-to-one")
    current.to_csv(NEW, index=False)
    print(f"Wrote {NEW.name}: {len(current)} entries covering {len(filenames)} CSV files")


if __name__ == "__main__":
    main()
