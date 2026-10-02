"""Versioned inventory after observed 30/60 m pixel-subring export."""
import pandas as pd

from analyze_isolation_reweighting_v8 import ROOT

OLD = ROOT / "tables/table_population_v8_followup6.csv"
NEW = ROOT / "tables/table_population_v8_followup8.csv"
ROWS = {
    "pixel_subring_regression_v8_followup6.csv": (
        "primary outside matched 1–2-pixel treated cells (419, 32 blocks)",
        "Fine 0–30/30–60/60–90 m and coarse 0–30/30–90 m joint fits"),
    "pixel_subring_collinearity_v8_followup6.csv": (
        "same 419 cells and 32 blocks",
        "Pairwise uncentred cosines, Pearson correlations and zero-count shares"),
    "pixel_subring_pair_differences_v8_followup7.csv": (
        "same 419 cells and 2,000 paired block draws",
        "Three direct within-90 m coefficient differences and intervals"),
    "table_population_v8_followup8.csv": (
        "all table CSV filenames through follow-up 8",
        "Versioned one-to-one inventory"),
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
