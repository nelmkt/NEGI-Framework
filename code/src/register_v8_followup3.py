"""Write a new, exhaustive table-population registry without changing v8 predecessors."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "tables/table_population_v8_followup2.csv"
NEW = ROOT / "tables/table_population_v8_followup3.csv"
ROWS = {
    "ring_exposure_regression_v8_followup3.csv": (
        "primary outside matched treated cells, all/1–2/belt/non-belt groups",
        "Four-column zero-intercept matched-contrast regression; descriptive, not causal ring response"),
    "joint_exposure_collinearity_v8_followup3.csv": (
        "all 15 earlier primary-outside joint own/external regression rows",
        "Uncentred design geometry and zero-external share; full rank does not identify causal split"),
    "isolation_subgroups_v8_followup3.csv": (
        "primary outside matched 1–2-pixel cells by radius, own dose, and belt",
        "Unweighted selected-site comparison; rows under 20 cells diagnostic only"),
    "table_population_v8_followup3.csv": (
        "all table CSV filenames through follow-up 3",
        "Versioned inventory only; blinded imagery key not opened"),
}


def main():
    if NEW.exists():
        raise SystemExit(f"Refusing to overwrite {NEW}")
    prior = pd.read_csv(OLD)
    filenames = {p.name for p in (ROOT / "tables").glob("*.csv")} | {NEW.name}
    if filenames - set(prior.table) != set(ROWS):
        raise SystemExit("Unclassified or missing new table filenames")
    rows = pd.DataFrame([dict(table=name, population=pop, interpretation=note)
                         for name, (pop, note) in ROWS.items()])
    registry = pd.concat([prior, rows], ignore_index=True)
    if registry.table.duplicated().any() or set(registry.table) != filenames:
        raise SystemExit("Versioned registry is not one-to-one")
    registry.to_csv(NEW, index=False)
    print(f"Wrote {NEW.name}: {len(registry)} entries covering {len(filenames)} CSV files")


if __name__ == "__main__":
    main()
