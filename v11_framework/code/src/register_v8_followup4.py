"""Inventory new saved-data robustness tables without changing older registries."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "tables/table_population_v8_followup3.csv"
NEW = ROOT / "tables/table_population_v8_followup4.csv"
ROWS = {
    "ring_pair_collinearity_v8_followup4.csv": (
        "primary outside matched treated, all/1–2/belt/non-belt groups",
        "Pairwise external-ring geometry and three-/four-column condition numbers"),
    "ring_leave_one_block_out_v8_followup4.csv": (
        "primary outside matched 1–2-pixel class, 32 separate treated-block deletions",
        "Refitted matched contrasts; descriptive one-block leverage sensitivity"),
    "ring_leave_one_block_summary_v8_followup4.csv": (
        "same 32 primary outside 1–2-pixel block deletions",
        "Range and largest change of the 0–90 m coefficient; not an interval"),
    "table_population_v8_followup4.csv": (
        "all table CSV filenames through follow-up 4",
        "Versioned inventory only; blinded imagery key not opened"),
}


def main():
    if NEW.exists():
        raise SystemExit(f"Refusing to overwrite {NEW}")
    prior = pd.read_csv(OLD)
    filenames = {p.name for p in (ROOT / "tables").glob("*.csv")} | {NEW.name}
    if filenames - set(prior.table) != set(ROWS):
        raise SystemExit("Unclassified new table filenames")
    extra = pd.DataFrame([dict(table=name, population=pop, interpretation=note)
                          for name, (pop, note) in ROWS.items()])
    current = pd.concat([prior, extra], ignore_index=True)
    if current.table.duplicated().any() or set(current.table) != filenames:
        raise SystemExit("Table inventory is not one-to-one")
    current.to_csv(NEW, index=False)
    print(f"Wrote {NEW.name}: {len(current)} entries covering {len(filenames)} CSV files")


if __name__ == "__main__":
    main()
