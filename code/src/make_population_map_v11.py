"""Independent population resolver for the complete follow-up-10 inventory.

The original v8 helper owns the original table families. New follow-up names are
mapped here explicitly; no inventory CSV (and no blinded imagery key) is read.
"""
from make_population_map_v8 import population as original_population


EXTRA = {
    "combined_class_relative_spread_v8_followup2.csv": ("primary outside matched four own-dose classes", "Range divided by absolute mean class slope"),
    "combined_class_spread_v8_followup.csv": ("primary outside matched four own-dose classes", "Historical absolute class-slope spread, superseded by relative table"),
    "combined_exposure_effects_v8.csv": ("primary pre-treatment-only matched cells by setting and own-dose class", "Own and own-plus-external denominators are descriptive"),
    "combined_exposure_fig5a_v8.csv": ("744/310 concurrent-change sensitivity matched population", "Hypothetical irrigation ratios; no resource ranking"),
    "combined_exposure_isolation_v8.csv": ("primary outside matched isolation subsets by own-dose class", "High-dose isolated samples are sparse or empty"),
    "isolation_balance_v8_followup2.csv": ("primary outside matched 1–2-pixel class", "Expanded calibration on three continuous and two binary margins"),
    "isolation_covariates_v8_followup.csv": ("primary outside matched 1–2-pixel class", "Baseline covariates by radius-defined isolation"),
    "isolation_reweighted_slopes_v8_followup.csv": ("primary outside matched 1–2-pixel class", "Earlier three-continuous-covariate calibration"),
    "isolation_subgroups_v8_followup2.csv": ("primary outside matched 1–2-pixel class", "Unweighted within-group comparisons; under-20 rows diagnostic"),
    "isolation_subgroups_v8_followup3.csv": ("primary outside matched 1–2-pixel cells by radius, own dose, and belt", "Unweighted selected-site comparison; rows under 20 cells diagnostic only"),
    "joint_exposure_collinearity_v8_followup3.csv": ("all 15 earlier primary-outside joint own/external regression rows", "Uncentred design geometry and zero-external share; full rank does not identify causal split"),
    "joint_exposure_regression_v8_followup2.csv": ("primary outside matched cells, all and specified own-dose classes", "Zero-intercept conditional association, not causal decomposition"),
    "pixel_subring_collinearity_v8_followup6.csv": ("same 419 cells and 32 blocks", "Pairwise uncentred cosines, Pearson correlations and zero-count shares"),
    "pixel_subring_pair_differences_v8_followup7.csv": ("same 419 cells and 2,000 paired block draws", "Three direct within-90 m coefficient differences and intervals"),
    "pixel_subring_regression_v8_followup6.csv": ("primary outside matched 1–2-pixel treated cells (419, 32 blocks)", "Fine 0–30/30–60/60–90 m and coarse 0–30/30–90 m joint fits"),
    "ring_exposure_regression_v8_followup3.csv": ("primary outside matched treated cells, all/1–2/belt/non-belt groups", "Four-column zero-intercept matched-contrast regression; descriptive, not causal ring response"),
    "ring_leave_one_block_out_v8_followup4.csv": ("primary outside matched 1–2-pixel class, 32 separate treated-block deletions", "Refitted matched contrasts; descriptive one-block leverage sensitivity"),
    "ring_leave_one_block_summary_v8_followup4.csv": ("same 32 primary outside 1–2-pixel block deletions", "Range and largest change of the 0–90 m coefficient; not an interval"),
    "ring_pair_collinearity_v8_followup4.csv": ("primary outside matched treated, all/1–2/belt/non-belt groups", "Pairwise external-ring geometry and three-/four-column condition numbers"),
    "ring_reduced_coefficients_v8_followup5.csv": ("primary outside matched 1–2-pixel treated cells (419, 32 blocks)", "Full, far-ring-omitted and merged-outer-ring zero-intercept fits; paired block draws"),
    "ring_reduced_coefficients_v8_followup6.csv": ("primary outside matched 1–2-pixel treated cells (419, 32 blocks)", "Full, far-ring-omitted and merged-outer-ring fits, reproduced from follow-up 5"),
    "ring_reduced_comparisons_v8_followup5.csv": ("same 419 cells and 2,000 spatial-block draws", "Paired own/near-ring changes relative to the full four-term model"),
    "ring_reduced_comparisons_v8_followup6.csv": ("same 419 cells and 2,000 paired spatial-block draws", "Includes the middle-ring coefficient change after omitting the far ring"),
    "table_population_v8_followup10.csv": ("all table CSV filenames in the v10 package", "Current inventory metadata only; blinded key not opened"),
    "table_population_v8_followup2.csv": ("all table CSV filenames through follow-up 2", "Versioned inventory only; blinded key not opened"),
    "table_population_v8_followup3.csv": ("all table CSV filenames through follow-up 3", "Versioned inventory only; blinded imagery key not opened"),
    "table_population_v8_followup4.csv": ("all table CSV filenames through follow-up 4", "Versioned inventory only; blinded imagery key not opened"),
    "table_population_v8_followup5.csv": ("all table CSV filenames through follow-up 5", "Versioned table inventory; unrun subring export has no result table"),
    "table_population_v8_followup6.csv": ("all table CSV filenames through follow-up 6", "Versioned inventory; 30/60 m Earth Engine export still unrun"),
    "table_population_v8_followup8.csv": ("all table CSV filenames through follow-up 8", "Versioned one-to-one inventory"),
}


def population(name: str) -> tuple[str, str]:
    """Resolve every saved table's population without consulting the inventory."""
    if name == "table_population_v8.csv":
        return "all table filenames in v8", "Inventory metadata only; blinded key not opened"
    if name in EXTRA:
        return EXTRA[name]
    return original_population(name)
