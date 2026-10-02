"""Versioned table-population inventory for the saved-data follow-ups."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "tables/table_population_v8.csv"
NEW = ROOT / "tables/table_population_v8_followup2.csv"

NEW_ROWS = {
    "combined_exposure_effects_v8.csv": ("primary pre-treatment-only matched cells by setting and own-dose class",
                                         "Own and own-plus-external denominators are descriptive"),
    "combined_exposure_fig5a_v8.csv": ("744/310 concurrent-change sensitivity matched population",
                                       "Hypothetical irrigation ratios; no resource ranking"),
    "combined_exposure_isolation_v8.csv": ("primary outside matched isolation subsets by own-dose class",
                                           "High-dose isolated samples are sparse or empty"),
    "combined_class_spread_v8_followup.csv": ("primary outside matched four own-dose classes",
                                              "Historical absolute class-slope spread, superseded by relative table"),
    "isolation_covariates_v8_followup.csv": ("primary outside matched 1–2-pixel class",
                                              "Baseline covariates by radius-defined isolation"),
    "isolation_reweighted_slopes_v8_followup.csv": ("primary outside matched 1–2-pixel class",
                                                     "Earlier three-continuous-covariate calibration"),
    "joint_exposure_regression_v8_followup2.csv": ("primary outside matched cells, all and specified own-dose classes",
                                                    "Zero-intercept conditional association, not causal decomposition"),
    "isolation_balance_v8_followup2.csv": ("primary outside matched 1–2-pixel class",
                                            "Expanded calibration on three continuous and two binary margins"),
    "isolation_subgroups_v8_followup2.csv": ("primary outside matched 1–2-pixel class",
                                              "Unweighted within-group comparisons; under-20 rows diagnostic"),
    "combined_class_relative_spread_v8_followup2.csv": ("primary outside matched four own-dose classes",
                                                          "Range divided by absolute mean class slope"),
    "table_population_v8_followup2.csv": ("all table CSV filenames through follow-up 2",
                                            "Versioned inventory only; blinded key not opened"),
}


def main():
    if NEW.exists():
        raise SystemExit(f"Refusing to overwrite {NEW}")
    old = pd.read_csv(OLD)
    names = {p.name for p in (ROOT / "tables").glob("*.csv")} | {NEW.name}
    absent = names - set(old.table)
    if absent != set(NEW_ROWS):
        raise SystemExit(f"Inventory source changed; unclassified filenames: {absent ^ set(NEW_ROWS)}")
    extra = pd.DataFrame([dict(table=name, population=pop, interpretation=meaning)
                          for name, (pop, meaning) in NEW_ROWS.items()])
    registry = pd.concat([old, extra], ignore_index=True)
    if registry.table.duplicated().any() or set(registry.table) != names:
        raise SystemExit("The versioned table inventory is not one-to-one")
    registry.to_csv(NEW, index=False)
    print(f"Wrote {NEW.name}: {len(registry)} entries covering {len(names)} CSV files")


if __name__ == "__main__":
    main()
