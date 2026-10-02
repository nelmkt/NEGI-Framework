"""Inventory new paired middle-ring table without modifying earlier registries."""
import pandas as pd

from analyze_isolation_reweighting_v8 import ROOT

OLD = ROOT / "tables/table_population_v8_followup5.csv"
NEW = ROOT / "tables/table_population_v8_followup6.csv"
ROWS = {
    "ring_reduced_coefficients_v8_followup6.csv": (
        "primary outside matched 1–2-pixel treated cells (419, 32 blocks)",
        "Full, far-ring-omitted and merged-outer-ring fits, reproduced from follow-up 5"),
    "ring_reduced_comparisons_v8_followup6.csv": (
        "same 419 cells and 2,000 paired spatial-block draws",
        "Includes the middle-ring coefficient change after omitting the far ring"),
    "table_population_v8_followup6.csv": (
        "all table CSV filenames through follow-up 6",
        "Versioned inventory; 30/60 m Earth Engine export still unrun"),
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
