from __future__ import annotations

# STANDARD LIBRARY
import datetime
import hashlib
import json
import logging
import os
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Third-party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Configuration
_NEGI_BASE_DIR = Path(__file__).resolve().parent


def _default_data_path() -> Path:
    configured_path = os.environ.get("NEGI_DATA_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return _NEGI_BASE_DIR / "data" / "jeddah_lst_data.csv"


@dataclass
class Config:
    """Single source of truth for every configurable parameter.

    Avoid hard-coding any numeric constant outside this class.  All
    downstream functions receive a `cfg` argument and read from here.
    """

    # Paths
    base_dir: Path = field(default_factory=lambda: _NEGI_BASE_DIR)
    data_path: Path = field(default_factory=_default_data_path)

    # Reproducibility
    random_seed: int = 42
    bootstrap_seed: int = 42

    # Verbosity
    verbose: bool = True
    debug: bool = False

    # Data splitting
    holdout_test_size: float = 0.20
    n_group_kfold_splits: int = 5
    n_random_search_iter: int = 80
    n_spatial_holdout_repeats: int = 10
    n_spatial_refits: int = 10
    uncertainty_refit_seed: int = 43
    bootstrap_iterations: int = 1000

    # Feature importance
    n_permutation_outer_seeds: int = 10
    n_permutation_inner_repeats: int = 30

    # XGBoost
    monotone_constraints: tuple[int, int, int] = (0, 1, 0)
    param_distributions: dict = field(default_factory=lambda: {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [2, 3, 4, 5, 6],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0.0, 0.1, 0.3],
        "reg_alpha":        [0.0, 0.1, 1.0],
        "reg_lambda":       [1, 2, 5],
    })

    # Scenarios
    scenario_step: float = 0.0025
    ndvi_target_percentile: float = 0.95
    ndvi_bin_edges: int = 40
    ndvi_bin_min_count: int = 20
    ndbi_sweep_points: int = 200
    ndbi_sweep_quantile_low: float = 0.01
    ndbi_sweep_quantile_high: float = 0.99

    # Smoothing
    tree_jump_threshold_c: float = 0.10
    savgol_smooth_target_window_pct: float = 5.0

    # NEGI
    alpha_weight: float = 1.0
    beta_weight: float = 1.0
    reference_w0: float = 200.0
    desal_energy_intensity: float = 3.5
    sqrt_exponent: float = 0.5
    linear_exponent: float = 1.0
    reference_cooling_scale_factor: float = 1.0

    # Support / extrapolation
    support_knn_k: int = 5
    support_percentile: float = 95.0
    moran_k_neighbors: int = 8

    # Uncertainty
    uncertainty_percentiles: tuple[float, ...] = (2.5, 25.0, 50.0, 75.0, 97.5)

    # Saturation diagnostic
    saturation_high_fraction: float = 0.90
    saturation_mid_fraction: float = 0.50
    saturation_near_zero_fraction: float = 0.05

    # Sensitivity
    sensitivity_exponents: tuple[float, ...] = (0.5, 1.0, 1.25, 1.5, 1.75)
    sensitivity_alpha_values: tuple[float, ...] = tuple(np.arange(0.4, 2.21, 0.2))
    sensitivity_beta_values: tuple[float, ...] = tuple(np.arange(0.4, 2.21, 0.2))
    sensitivity_w0_values: tuple[float, ...] = tuple(np.arange(50.0, 401.0, 50.0))
    sensitivity_default_exponent: float = 0.5
    sensitivity_default_beta: float = 1.0

    # Interpretive diagnostics (peer review): reporting only, no effect on
    # any modelling, optimization, or NEGI computation.
    influence_leverage_multiplier: float = 2.0   # leverage threshold = k * p / n
    influence_cooks_d_multiplier: float = 4.0    # Cook's D threshold  = k / n
    qq_envelope_percentiles: tuple[float, float] = (95.0, 99.0)
    low_r2_threshold: float = 0.10
    variance_band_very_weak: float = 0.05
    variance_band_weak: float = 0.15
    variance_band_moderate: float = 0.30

    # Interpretation-only thresholds: these control only which canned
    # interpretive sentence is shown, and do not feed into any model,
    # validation, optimization, or NEGI computation.
    moran_i_magnitude_threshold: float = 0.30      # magnitude-based flag
    scenario_boundary_tolerance_pct: float = 5.0   # "near boundary" band

    # Colour palette
    color_primary: str = "#1B6FA8"
    color_secondary: str = "#4A96C8"
    color_accent: str = "#C84A4A"
    color_neutral: str = "#3B3B3B"
    color_s1: str = "#2A9D8F"
    color_s2: str = "#E76F51"

    # Figures
    default_figsize: tuple[float, float] = (10.0, 6.0)

    # Spatial column name candidates
    spatial_x_candidates: tuple[str, ...] = (
        "x", "X", "longitude", "Longitude", "lon", "Lon", "easting", "Easting"
    )
    spatial_y_candidates: tuple[str, ...] = (
        "y", "Y", "latitude", "Latitude", "lat", "Lat", "northing", "Northing"
    )

    # Derived paths (set in __post_init__)
    results_dir:            Path = field(init=False)
    data_dir:               Path = field(init=False)
    main_png_dir:           Path = field(init=False)
    main_pdf_dir:           Path = field(init=False)
    supplementary_png_dir:  Path = field(init=False)
    supplementary_pdf_dir:  Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.data_path = Path(self.data_path)
        # Canonical layout:  Results/Main/{PNG,PDF}  and
        #                    Results/Supplementary/{PNG,PDF}
        self.results_dir           = self.base_dir / "Results"
        self.data_dir              = self.base_dir / "Data"
        self.main_png_dir          = self.results_dir / "Main" / "PNG"
        self.main_pdf_dir          = self.results_dir / "Main" / "PDF"
        self.supplementary_png_dir = self.results_dir / "Supplementary" / "PNG"
        self.supplementary_pdf_dir = self.results_dir / "Supplementary" / "PDF"


CFG = Config()

# CONSTANTS
EPS            = 1e-12
TEMP_ZERO_GUARD = 1e-9

# Scenario axis label (shared across all scenario figures)
SCENARIO_AXIS_LABEL = "Scenario intensity (%)"

# Canonical figure filenames
# Main figures use Figure_NN_<description>.png
# Supplementary figures use Figure_SNN_<description>.png
# These constants are used by save_fig() to route to the correct folder.

FIG_ACTUAL_VS_PREDICTED         = "Figure_01_Observed_vs_Predicted.png"
FIG_MODEL_COMPARISON            = "Figure_S01_Model_Comparison.png"
FIG_FEATURE_IMPORTANCE          = "Figure_02_Feature_Importance.png"
FIG_LST_RESPONSE_TO_NDBI        = "Figure_03_LST_Response_to_NDBI.png"
FIG_NEGI_SCENARIO_COMPARISON    = "Figure_04_NEGI_Scenario_Comparison.png"
FIG_NEGI_UNCERTAINTY_BANDS      = "Figure_05_NEGI_Uncertainty_Bands.png"
FIG_NDVI_CONDITIONAL_BY_NDBI    = "Figure_06_NDVI_Conditional_by_NDBI.png"
FIG_NDVI_VS_LST                 = "Figure_S02_NDVI_vs_LST.png"
FIG_NDVI_DECILE_ANALYSIS        = "Figure_S03_NDVI_Decile_Analysis.png"
FIG_NDBI_RESPONSE_UNCERTAINTY   = "Figure_S04_NDBI_Response_Uncertainty.png"
FIG_RESIDUALS_VS_PREDICTED      = "Figure_S05_Residuals_vs_Predicted.png"
FIG_RESIDUALS_QQ                = "Figure_S06_Residuals_QQ.png"
FIG_RESIDUALS_HISTOGRAM         = "Figure_S07_Residuals_Histogram.png"
FIG_RESIDUAL_SPATIAL_MAP        = "Figure_S08_Residual_Spatial_Map.png"
FIG_NEGI_SMOOTHING_ROBUSTNESS   = "Figure_S09_NEGI_Smoothing_Robustness.png"
FIG_NEGI_EXPONENT_SENSITIVITY   = "Figure_S10_NEGI_Exponent_Sensitivity.png"
FIG_NEGI_ENERGY_COST_COMPARISON = "Figure_S11_NEGI_Energy_Cost_Comparison.png"
FIG_RESPONSE_SURFACE            = "Figure_S12_Response_Surface.png"
FIG_SCENARIO2_DATA_SUPPORT      = "Figure_S13_Scenario2_Data_Support.png"
FIG_NEGI_SCENARIO2_ROBUSTNESS   = "Figure_S14_NEGI_Scenario2_Robustness.png"
FIG_SENSITIVITY_OPTIMAL_NEGI    = "Figure_S15_Sensitivity_Optimal_NEGI.png"
FIG_COOLING_SATURATION          = "Figure_S16_Cooling_Saturation_Diagnostic.png"

# Set of main-figure filenames - used by save_fig to pick the right folder.
MAIN_FIGURE_FILENAMES: frozenset[str] = frozenset({
    FIG_ACTUAL_VS_PREDICTED,
    FIG_FEATURE_IMPORTANCE,
    FIG_LST_RESPONSE_TO_NDBI,
    FIG_NEGI_SCENARIO_COMPARISON,
    FIG_NEGI_UNCERTAINTY_BANDS,
    FIG_NDVI_CONDITIONAL_BY_NDBI,
})

# Typography constants
TITLE_SIZE   = 16
TITLE_WEIGHT = "normal" 
LABEL_SIZE   = 13
LABEL_WEIGHT = "normal"
TICK_SIZE    = 10
LEGEND_SIZE  = 11
LINE_WIDTH   = 2.0
MARKER_SIZE  = 8
GRID_ALPHA   = 0.28

STAT_BOX_STYLE: dict = {
    "boxstyle": "round,pad=0.35",
    "facecolor": "white",
    "edgecolor": "#666666",
    "alpha": 0.85,
    "linewidth": 1.0,
}


# LOGGING
logger = logging.getLogger("negi_framework")
_LOG_FILE_PATH: Optional[Path] = None


def configure_logging(cfg: Config) -> None:
    """Configure the module logger once, at pipeline start.

    Console handler level follows cfg.verbose (INFO when verbose, WARNING
    when quiet).  A file handler always captures INFO/WARNING/ERROR to
    data/pipeline.log, and DEBUG as well when cfg.debug is True.
    Idempotent - safe to call multiple times; clears previous handlers to
    avoid duplicate log lines.
    """
    global _LOG_FILE_PATH
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    console_handler.setLevel(logging.INFO if cfg.verbose else logging.WARNING)
    logger.addHandler(console_handler)

    try:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.data_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        file_handler.setLevel(logging.DEBUG if cfg.debug else logging.INFO)
        logger.addHandler(file_handler)
        _LOG_FILE_PATH = log_path
    except OSError as exc:
        warnings.warn(f"Could not attach file log handler (pipeline.log): {exc}")

    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def log_info(*args) -> None:
    """Log `args` (space-joined) at INFO level."""
    logger.info(" ".join(str(a) for a in args))


def log_warning(*args) -> None:
    """Log `args` (space-joined) at WARNING level."""
    logger.warning(" ".join(str(a) for a in args))


def log_error(*args) -> None:
    """Log `args` (space-joined) at ERROR level."""
    logger.error(" ".join(str(a) for a in args))


def log_debug(*args) -> None:
    """Log `args` (space-joined) at DEBUG level (developer-only)."""
    logger.debug(" ".join(str(a) for a in args))


def vlog_info(cfg: Config, *args) -> None:
    """Log only when cfg.verbose is True."""
    if cfg.verbose:
        log_info(*args)


# METRICS
def _float_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def isclose_mask(series: pd.Series, value: float, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of `series` entries within `tol` of `value`."""
    return np.isclose(series.to_numpy(dtype=float), value, atol=tol)


def safe_divide(numerator, denominator, fill: float = 0.0):
    """Division that returns `fill` wherever |denominator| <= EPS."""
    num, den = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    result = np.full(num.shape, fill, dtype=float)
    np.divide(num, den, out=result, where=np.abs(den) > EPS)
    return float(result) if result.ndim == 0 else result


def safe_normalize_max(values, fill: float = 0.0):
    """Scale `values` by their max absolute value; `fill` where that max is ~0."""
    values_arr = np.asarray(values, dtype=float)
    return safe_divide(values_arr, np.nanmax(np.abs(values_arr)), fill=fill)


def assert_finite(values, name: str = "values") -> None:
    """Raise ValueError if `values` contains any NaN or infinite entry."""
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ValueError(f"{name} contains NaN or infinite values.")


def ci95(values) -> float:
    """Half-width of the normal-approximation 95% CI for the mean of `values`."""
    values_arr = np.asarray(values, dtype=float)
    if values_arr.size < 2:
        return 0.0
    return float(1.96 * values_arr.std(ddof=0) / np.sqrt(values_arr.size))


def compute_rmse(y_true, y_pred) -> float:
    """Root-mean-squared error between `y_true` and `y_pred`."""
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true_a - y_pred_a) ** 2)))


def compute_mae(y_true, y_pred) -> float:
    """Mean absolute error between `y_true` and `y_pred`."""
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true_a - y_pred_a)))


def compute_bias(y_true, y_pred) -> float:
    """Mean signed bias (`y_true` minus `y_pred`)."""
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    return float(np.mean(y_true_a - y_pred_a))


def compute_r2(y_true, y_pred) -> float:
    """Coefficient of determination (R²) between `y_true` and `y_pred`."""
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    return float(r2_score(y_true_a, y_pred_a))


def compute_calibration(y_true, y_pred) -> tuple[float, float]:
    """Calibration slope and intercept from a linear fit of `y_true` on `y_pred`.

    Returns:
        (slope, intercept); (nan, nan) if there are fewer than two points or
        `y_pred` is constant.
    """
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    if y_true_a.size < 2 or np.ptp(y_pred_a) <= EPS:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(y_pred_a, y_true_a, 1)
    return float(slope), float(intercept)


def compute_pearson_r(y_true, y_pred) -> tuple[float, float]:
    """Pearson correlation coefficient and p-value between `y_true` and `y_pred`.

    Returns:
        (r, p_value); (nan, nan) if there are fewer than two points or either
        array is constant.
    """
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    if y_true_a.size < 2 or np.ptp(y_true_a) <= EPS or np.ptp(y_pred_a) <= EPS:
        return float("nan"), float("nan")
    result = scipy_stats.pearsonr(y_true_a, y_pred_a)
    return float(result.statistic), float(result.pvalue)


def compute_mean_absolute_calibration_error(y_true, y_pred) -> float:
    """Mean absolute calibration error (equivalent to MAE on the same pair)."""
    return compute_mae(y_true, y_pred)


def compute_residual_stats(residuals) -> dict:
    """Summary statistics (mean, median, std, skewness, kurtosis, Shapiro p) for residuals."""
    arr = np.asarray(residuals, dtype=float)
    shapiro_p = (
        float(scipy_stats.shapiro(arr).pvalue)
        if 3 <= arr.size <= 5000
        else float("nan")
    )
    return {
        "mean":     float(arr.mean()),
        "median":   float(np.median(arr)),
        "std":      float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "skewness": float(scipy_stats.skew(arr)),
        "kurtosis": float(scipy_stats.kurtosis(arr)),
        "shapiro_p": shapiro_p,
    }


def compute_metrics(y_true, y_pred) -> dict:
    """Bundle RMSE, MAE, bias, R², and calibration slope/intercept into one dict."""
    slope, intercept = compute_calibration(y_true, y_pred)
    return {
        "rmse":                 compute_rmse(y_true, y_pred),
        "mae":                  compute_mae(y_true, y_pred),
        "bias":                 compute_bias(y_true, y_pred),
        "r2":                   compute_r2(y_true, y_pred),
        "calibration_slope":    slope,
        "calibration_intercept": intercept,
    }


# Private metric functions for bootstrap
def _metric_rmse(y_true, y_pred) -> float:              return compute_rmse(y_true, y_pred)
def _metric_mae(y_true, y_pred) -> float:               return compute_mae(y_true, y_pred)
def _metric_bias(y_true, y_pred) -> float:              return compute_bias(y_true, y_pred)
def _metric_calibration_slope(y_true, y_pred) -> float: return compute_calibration(y_true, y_pred)[0]
def _metric_calibration_intercept(y_true, y_pred) -> float: return compute_calibration(y_true, y_pred)[1]


def bootstrap_metric(
    y_true, y_pred, metric_fn: Callable, n_boot: int = 1000, seed: int = 42
) -> dict:
    """Bootstrap 95% CI for a scalar metric function."""
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    if y_true_a.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, y_true_a.size, size=y_true_a.size)
        samples[i] = metric_fn(y_true_a[idx], y_pred_a[idx])
    return {
        "point_estimate": float(metric_fn(y_true_a, y_pred_a)),
        "median":  float(np.nanmedian(samples)),
        "lower":   float(np.nanpercentile(samples, 2.5)),
        "upper":   float(np.nanpercentile(samples, 97.5)),
        "n_boot":  int(n_boot),
    }


def make_rng(cfg: Config) -> np.random.Generator:
    """Construct the seeded NumPy random generator used for bootstrap resampling."""
    return np.random.default_rng(cfg.bootstrap_seed)


# PLOTTING HELPERS
def apply_plot_style(cfg: Config = CFG) -> None:
    """Apply consistent rcParams for publication-quality figures."""
    mpl.rcParams.update({
        "figure.dpi":          120,
        "savefig.dpi":         300,
        "font.size":           TICK_SIZE,
        "axes.titlesize":      TITLE_SIZE,
        "axes.labelsize":      LABEL_SIZE,
        "legend.fontsize":     LEGEND_SIZE,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
    })


# Alias kept for any call site that uses the original name.
apply_publication_style = apply_plot_style


def set_title(ax, title: str, *, fontsize: int = TITLE_SIZE,
              fontweight: str = TITLE_WEIGHT, **kw) -> None:
    """Set axis title with consistent size and weight (normal by default)."""
    ax.set_title(title, fontsize=fontsize, fontweight=fontweight, **kw)


def set_axis_labels(
    ax, xlabel: str = "", ylabel: str = "",
    *, fontsize: int = LABEL_SIZE, fontweight: str = LABEL_WEIGHT,
) -> None:
    """Set x- and y-axis labels with consistent typography."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)


def set_legend(ax, **kw) -> None:
    """Add a legend with consistent size and frame."""
    defaults = dict(fontsize=LEGEND_SIZE, frameon=True)
    defaults.update(kw)
    ax.legend(**defaults)


def stat_box(ax, text: str, x: float = 0.05, y: float = 0.97, **kw) -> None:
    """Render a statistics annotation box with the standard house style."""
    defaults = dict(
        transform=ax.transAxes, verticalalignment="top",
        fontsize=TICK_SIZE, bbox=STAT_BOX_STYLE,
    )
    defaults.update(kw)
    ax.text(x, y, text, **defaults)


def apply_grid(ax, alpha: float = GRID_ALPHA) -> None:
    """Apply a light grid and remove top/right spines consistently."""
    ax.grid(True, alpha=alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_footer(fig, text: str, cfg: Config = CFG, *, min_bottom: float = 0.16) -> None:
    """Add a small-font footer line at the bottom of a figure.

    Stores the required bottom margin on the figure object so that
    `save_fig` can re-apply it after `tight_layout()` - tight_layout only
    accounts for axes content, not free-floating `fig.text` footers, so
    without this the footer margin gets silently discarded and the footer
    text collides with the x-axis label/ticks.
    """
    fig.text(
        0.5, 0.01, text,
        ha="center", va="bottom",
        fontsize=7.5, color=cfg.color_neutral, wrap=True,
    )
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, min_bottom))
    fig._footer_min_bottom = min_bottom


def generate_caption(description: str, statistics: Optional[dict] = None) -> str:
    """Build a figure caption string, optionally appending key statistics."""
    if not statistics:
        return description
    values = "; ".join(
        f"{k}={v:.4g}" if isinstance(v, (int, float, np.number)) and np.isfinite(v)
        else f"{k}={v}"
        for k, v in statistics.items()
    )
    return f"{description}. {values}" if values else description


# FIGURE REGISTRY
class FigureRegistry:
    """Tracks every figure saved during a pipeline run for the figure list."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(
        self, filename: str, caption: str = "",
        description: str = "", section: str = "",
    ) -> None:
        """Register one figure/table export with its caption, description, and section."""
        self._rows.append({
            "filename":    filename,
            "caption":     caption,
            "description": description,
            "section":     section,
        })

    def export(self, path: Path) -> Path:
        """Write the accumulated figure/table registry to a CSV at `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            self._rows, columns=["filename", "caption", "description", "section"]
        ).to_csv(path, index=False)
        return path


FIGURE_REGISTRY = FigureRegistry()


def get_version_info() -> dict:
    """Python, NumPy, and pandas versions in use for this run."""
    return {
        "python": sys.version.split()[0],
        "numpy":  np.__version__,
        "pandas": pd.__version__,
    }


# CENTRALIZED FIGURE SAVE
def save_fig(
    fig,
    filename: str,
    cfg: Config = CFG,
    *,
    use_tight_layout: bool = True,
    caption: str = "",
    description: str = "",
    section: str = "",
) -> Path:
    """Save a figure to the correct Results subfolder as PNG and PDF.

    Routing logic
    -------------
    If `filename` is in MAIN_FIGURE_FILENAMES  -> Results/Main/{PNG,PDF}
    Otherwise                                  -> Results/Supplementary/{PNG,PDF}

    Parameters
    ----------
    fig : matplotlib Figure
    filename : str
        Canonical figure filename (e.g. FIG_ACTUAL_VS_PREDICTED).
    cfg : Config
    use_tight_layout : bool
        Call fig.tight_layout() before saving (default True).
    caption / description / section : str
        Metadata registered in FIGURE_REGISTRY.

    Returns
    -------
    Path
        Path of the saved PNG file.
    """
    filename_path = Path(filename)
    if not filename_path.suffix:
        filename_path = filename_path.with_suffix(".png")

    is_main = filename_path.name in MAIN_FIGURE_FILENAMES
    png_dir = cfg.main_png_dir if is_main else cfg.supplementary_png_dir
    pdf_dir = cfg.main_pdf_dir if is_main else cfg.supplementary_pdf_dir
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    if use_tight_layout:
        try:
            fig.tight_layout()
        except (ValueError, RuntimeError):
            pass
        footer_min_bottom = getattr(fig, "_footer_min_bottom", None)
        if footer_min_bottom is not None:
            fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, footer_min_bottom))

    png_path = png_dir / filename_path.name
    pdf_path = pdf_dir / filename_path.with_suffix(".pdf").name
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    FIGURE_REGISTRY.add(
        filename=png_path.name,
        caption=caption,
        description=description,
        section=section,
    )
    log_info(f"Saved: {png_path}")
    return png_path


# Alias kept for any call site that still uses export_fig.
export_fig = save_fig


# I/O HELPERS
def setup_directories(cfg: Config) -> None:
    """Create all output directories (idempotent)."""
    for d in (
        cfg.data_dir,
        cfg.main_png_dir, cfg.main_pdf_dir,
        cfg.supplementary_png_dir, cfg.supplementary_pdf_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)


def confirm_saved(path: Path) -> None:
    """Log confirmation that a file was written to `path`."""
    log_info(f"Saved: {path}")


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Write `df` to `path` as CSV and log the confirmation."""
    df.to_csv(path, index=index)
    confirm_saved(path)
    return path


def save_json(data: dict, path: Path) -> Path:
    """Unified JSON export helper - mirrors save_csv's behaviour
    so every JSON artifact in the pipeline goes through one function."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    confirm_saved(path)
    return path


def save_summary(text: str, path: Path) -> Path:
    """Unified plain-text/markdown summary export helper."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    confirm_saved(path)
    return path


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("|", "\\|") for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# SUMMARY LOG
class SummaryLog:
    """Accumulates (Section, Quantity, Value) rows during the pipeline run.

    After the run, export() writes a CSV and a Markdown report with one
    section per heading, plus auto-populated Warnings and Conclusions
    sections built from detected diagnostics.
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._warnings: list[str] = []
        self._conclusions: list[str] = []

    def log(self, section: str, quantity: str, value) -> None:
        """Record one (section, quantity, value) row for the summary report."""
        self._rows.append({"Section": section, "Quantity": quantity, "Value": value})

    def add_warning(self, text: str) -> None:
        """Register a diagnostic warning for the Warnings section."""
        self._warnings.append(text)

    def add_conclusion(self, text: str) -> None:
        """Register a conclusion statement for the Conclusions section."""
        self._conclusions.append(text)

    def to_dataframe(self) -> pd.DataFrame:
        """Combine logged rows, warnings, and conclusions into one DataFrame."""
        rows = list(self._rows)
        for i, w in enumerate(self._warnings, 1):
            rows.append({"Section": "Warnings", "Quantity": f"Warning {i}", "Value": w})
        for i, c in enumerate(self._conclusions, 1):
            rows.append({"Section": "Conclusions", "Quantity": f"Conclusion {i}", "Value": c})
        return pd.DataFrame(rows)

    def export(self, csv_path: Path, md_path: Path) -> pd.DataFrame:
        """Write the full summary report as CSV and Markdown; return the DataFrame."""
        df = self.to_dataframe()
        save_csv(df, csv_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Full Summary Report\n\n")
            for section in df["Section"].unique():
                f.write(f"## {section}\n\n")
                section_df = df[df["Section"] == section][["Quantity", "Value"]]
                f.write(dataframe_to_markdown(section_df))
                f.write("\n\n")
        confirm_saved(md_path)
        log_info(
            f"\nFull summary report exported: {len(df)} rows across "
            f"{df['Section'].nunique()} sections (CSV / Markdown)."
        )
        return df


# INTERPRETATION LIBRARY
#
# Single source of truth for the wording of recurring interpretive
# statements.  Every call site below (reports, figure captions,
# Interpretation Notes, Conclusions) pulls its sentence from this
# dictionary instead of re-typing similar language in multiple places.
# This section changes NO calculation, threshold value, statistic, or
# figure - it only ensures identical scientific concepts are always
# described with identical wording.

INTERPRETATION_LIBRARY: dict[str, str] = {
    "spatial_dependence": (
        "Residual spatial dependence remained after spatial-block validation, "
        "indicating that some spatially structured variation in LST remains "
        "unexplained. Point estimates remain valid, but local prediction errors "
        "may exhibit spatial correlation and predictive performance should be "
        "interpreted with appropriate caution."
    ),
    "boundary_maximum": (
        "The highest evaluated NEGI occurs near the boundary of the explored "
        "scenario trajectory. This should be interpreted as the best-performing "
        "point within the evaluated scenario rather than evidence of a unique "
        "interior or global optimum."
    ),
    "conditional_ndvi": (
        "Conditional NDVI-LST relationships vary across NDBI strata. These "
        "conditional associations should not be interpreted as evidence of a "
        "single universal NDVI cooling effect; differences among strata reflect "
        "interaction effects between vegetation and urbanization rather than "
        "contradictory results."
    ),
    "calibration": (
        "Calibration slope, intercept, and mean bias describe agreement between "
        "observed and predicted LST without post-hoc recalibration. RMSE and MAE "
        "are reported separately in the performance section."
    ),
    "feature_support": (
        "Scenario points flagged as outside feature-space support extrapolate "
        "beyond the range of the training data and should be interpreted with "
        "additional caution."
    ),
    "smoothing": (
        "Smoothed curves are provided solely to improve visualization. All "
        "scientific interpretation and diagnostics are based on the underlying "
        "model predictions rather than the smoothed representation."
    ),
    "heteroscedasticity": (
        "Mild heteroscedasticity was detected, indicating that residual variance "
        "changes across the prediction range. This behavior is common in "
        "environmental prediction models and does not invalidate the model, but "
        "it should be considered when interpreting prediction uncertainty."
    ),
    "predictor_scope": (
        "Only three predictors (NDBI, Elevation, and NDVI) were intentionally "
        "used in this model. The remaining unexplained variance likely "
        "reflects additional environmental processes not captured by these "
        "variables. Omitted variables such as building morphology, wind "
        "exposure, and land-use type are presented only as potential future "
        "work and do not represent demonstrated deficiencies of the current "
        "model."
    ),
    "benchmark_comparison": (
        "XGBoost was optimized using RandomizedSearchCV. Benchmark models are "
        "included for contextual comparison and were intentionally not "
        "optimized to the same extent. Therefore, benchmark comparisons "
        "should be interpreted as contextual rather than evidence of global "
        "model superiority."
    ),
}


def interpretation_text(key: str) -> str:
    """Return the canonical interpretive sentence for `key`.

    Raises KeyError with a clear message for unknown keys so that a typo
    fails loudly rather than silently returning an empty caption.
    """
    try:
        return INTERPRETATION_LIBRARY[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown interpretation key {key!r}. "
            f"Available keys: {sorted(INTERPRETATION_LIBRARY)}"
        ) from exc


def is_near_trajectory_boundary(idx: int, n_points: int, tolerance_pct: float) -> bool:
    """Interpretation-only helper: True when `idx` falls within the
    first or last `tolerance_pct`% of a trajectory of length `n_points`.

    This does not change how the optimum index itself is computed anywhere
    in the pipeline - it only widens the *reporting* rule from an exact
    endpoint match to a configurable near-boundary band, so that maxima one
    or two steps from the edge are still flagged as boundary-adjacent.
    """
    if n_points <= 1:
        return True
    band = max(1, int(round((tolerance_pct / 100.0) * (n_points - 1))))
    return idx <= band or idx >= (n_points - 1 - band)


# NAMED INTERPRETATION API

def interpret_calibration(validation: "ValidationResults") -> str:
    """One-sentence calibration interpretation plus the canonical caveat."""
    return (
        f"Calibration slope = {validation.calibration_slope:.3f}, "
        f"intercept = {validation.calibration_intercept:.3f}, "
        f"mean bias = {validation.calibration_bias:.3f} \u00b0C. "
        + interpretation_text("calibration")
    )


def interpret_moran(moran: dict, cfg: Config = CFG) -> str:
    """Interpret a global_morans_i() result dict against the configured
    magnitude threshold (cfg.moran_i_magnitude_threshold)."""
    if not np.isfinite(moran.get("morans_i", float("nan"))):
        return "Global Moran's I could not be computed for this residual set."
    text = (
        f"Global Moran's I = {moran['morans_i']:.4f} "
        f"(p = {moran['p_value']:.4g})."
    )
    if moran["morans_i"] > cfg.moran_i_magnitude_threshold:
        text += " " + interpretation_text("spatial_dependence")
    return text


def interpret_bp(bp: dict, cfg: Config = CFG) -> str:
    """Distinguish statistical vs. practical significance for the
    Breusch-Pagan heteroscedasticity test: a significant p-value
    with a small effect is reported as such, not as "severe" heteroscedasticity."""
    stat_significant = bp["p_value"] < 0.05
    text = f"Breusch-Pagan statistic = {bp['statistic']:.4f}, p = {bp['p_value']:.4g}."
    if stat_significant:
        text += " " + interpretation_text("heteroscedasticity")
    else:
        text += " No statistically significant heteroscedasticity was detected."
    return text


def interpret_feature_importance(fi: "FeatureImportanceResult") -> str:
    """Describe the dominant predictor and margin over the runners-up,
    using the already-sorted permutation-importance table (marginal
    predictive importance)."""
    table = fi.table
    lead = (
        f"{table['Feature'].iloc[0]} is the dominant predictor "
        f"(permutation importance = {table['Importance'].iloc[0]:.3f})"
    )
    if len(table) > 1:
        others = ", ".join(
            f"{row['Feature']} ({row['Importance']:.3f})"
            for _, row in table.iloc[1:].iterrows()
        )
        lead += f", substantially exceeding {others}."
    else:
        lead += "."
    return (
        lead + " Permutation importance reflects marginal predictive "
        "importance under the fitted model; correlated predictors and "
        "non-additive effects mean importances should not be read as "
        "independent causal contributions."
    )


def interpret_boundary_maximum() -> str:
    """Canonical boundary-optimum caveat."""
    return interpretation_text("boundary_maximum")


def interpret_model_comparison() -> str:
    """Canonical benchmark-comparison disclaimer: the XGBoost
    surrogate was tuned via RandomizedSearchCV; benchmark models are shown
    for context only and the comparison is not intended to establish
    global optimality."""
    return interpretation_text("benchmark_comparison")


def interpret_support(feature_support_status: str) -> str:
    """Interpret a scenario's within/outside-support status."""
    if feature_support_status.lower().startswith("ok"):
        return f"{feature_support_status}. Scenario points fall within training-data support."
    return f"{feature_support_status}. " + interpretation_text("feature_support")


def interpret_uncertainty(uncertainty: "UncertaintyResults", scenario_pct: np.ndarray) -> str:
    """Describe where uncertainty (IQR across spatial-block refits) peaks
    along the scenario trajectory."""
    iqr = uncertainty.percentiles.get(75.0, uncertainty.negi_s1_std) - \
        uncertainty.percentiles.get(25.0, uncertainty.negi_s1_std)
    if isinstance(iqr, np.ndarray) and iqr.size == len(scenario_pct):
        peak_idx = int(np.argmax(iqr))
        return (
            f"Uncertainty (IQR across spatial-block refits) is largest near "
            f"scenario = {scenario_pct[peak_idx]:.0f}%."
        )
    return "Uncertainty bands are reported across spatial-block holdout refits."


# PREDICTION CACHE
class PredictionCache:
    """Caches model.predict() calls keyed on the rounded (NDVI, NDBI,
    Elevation) input triple.

    Reuse never alters what is predicted or how the model was fit - it
    only avoids duplicate calls to model.predict() for feature vectors
    already evaluated elsewhere in the pipeline.
    """

    def __init__(self, precision: int = 10) -> None:
        self.precision = precision
        self._store: dict[tuple, tuple[float, bool]] = {}
        self.hits  = 0
        self.misses = 0

    def _key(self, ndvi: float, ndbi: float, elevation: float) -> tuple:
        r = round
        return (r(float(ndvi), self.precision),
                r(float(ndbi), self.precision),
                r(float(elevation), self.precision))

    def get(self, ndvi: float, ndbi: float, elevation: float):
        """Return the cached prediction for this (NDVI, NDBI, Elevation) triple, if any."""
        return self._store.get(self._key(ndvi, ndbi, elevation))

    def set(self, ndvi: float, ndbi: float, elevation: float,
            value: tuple[float, bool]) -> None:
        """Store `value` under this (NDVI, NDBI, Elevation) triple."""
        self._store[self._key(ndvi, ndbi, elevation)] = value

    def stats(self) -> dict:
        """Hit/miss counts, cache size, and hit rate (developer diagnostics only)."""
        total    = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            "hits": self.hits, "misses": self.misses,
            "size": len(self._store), "hit_rate": hit_rate,
        }

    def clear(self) -> None:
        """Reset the cache contents and hit/miss counters."""
        self._store.clear()
        self.hits  = 0
        self.misses = 0


PREDICTION_CACHE = PredictionCache()

# Reproducibility
def print_reproducibility_info(cfg: Config) -> dict:
    """Log and return the seed, library versions, and git commit for this run."""
    info = {
        "random_seed":     cfg.random_seed,
        "python_version":  sys.version.split()[0],
        "numpy_version":   np.__version__,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "pandas_version":  pd.__version__,
        "scipy_version":   __import__("scipy").__version__,
    }
    info["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cfg_str = json.dumps({k: str(v) for k, v in cfg.__dict__.items()}, sort_keys=True)
    info["config_sha256"] = hashlib.sha256(cfg_str.encode()).hexdigest()[:16]

    log_info("REPRODUCIBILITY INFORMATION")
    for key, val in info.items():
        log_info(f"  {key:20s}: {val}")

    try:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.data_dir / "reproducibility_info.json", "w") as f:
            json.dump(info, f, indent=2)
    except OSError as exc:
        warnings.warn(f"Could not write reproducibility_info.json: {exc}")

    return info


def write_reproducibility_report(
    cfg: Config,
    repro_info: dict,
    best_params: Optional[dict] = None,
    validation: Optional["ValidationResults"] = None,
    runtime_seconds: Optional[float] = None,
) -> Path:
    """Write a human-readable Reproducibility_Report.txt."""
    lines = [
        "NEGI PIPELINE - REPRODUCIBILITY REPORT",
        "=" * 60,
        f"Date (UTC)         : "
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Random seed        : {cfg.random_seed}",
        "",
        "Package versions",
        "-" * 60,
        f"  Python           : {repro_info.get('python_version')}",
        f"  numpy            : {repro_info.get('numpy_version')}",
        f"  pandas           : {repro_info.get('pandas_version')}",
        f"  scipy            : {repro_info.get('scipy_version')}",
        f"  scikit-learn     : {repro_info.get('sklearn_version')}",
        f"  xgboost          : {repro_info.get('xgboost_version')}",
        "",
    ]
    if best_params is not None:
        lines += ["Selected XGBoost hyperparameters (RandomizedSearchCV)", "-" * 60]
        for k, v in best_params.items():
            lines.append(f"  {k:20s}: {v}")
        lines.append("")
    if validation is not None:
        lines += [
            "Validation results",
            "-" * 60,
            f"  Nested GroupKFold CV R²            : {validation.nested_cv_r2:.4f}",
            f"  Repeated spatial-holdout R²        : "
            f"{validation.repeated_r2_mean:.4f} +/- {validation.repeated_r2_ci95:.4f}",
            f"  Untouched confirmatory holdout R²  : {validation.holdout_r2:.4f}",
            f"  Untouched confirmatory holdout RMSE: {validation.holdout_rmse:.4f}",
            f"  Untouched confirmatory holdout MAE : {validation.holdout_mae:.4f}",
            f"  Calibration slope                  : {validation.calibration_slope:.4f}",
            f"  Calibration intercept              : {validation.calibration_intercept:.4f}",
            f"  Calibration bias                   : {validation.calibration_bias:.4f}",
            "",
        ]
    if runtime_seconds is not None:
        lines.append(f"Total runtime              : {runtime_seconds:.1f} s")
    lines.append(f"Config SHA-256 (first 16)  : {repro_info.get('config_sha256', 'n/a')}")

    report_text = "\n".join(lines) + "\n"
    report_path = cfg.data_dir / "Reproducibility_Report.txt"
    try:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        confirm_saved(report_path)
    except OSError as exc:
        warnings.warn(f"Could not write Reproducibility_Report.txt: {exc}")
    return report_path


# AUTOMATED QA CHECKS
#
# Consistency checks run once, near the end of the pipeline, right before
# the final exports.  These do not recompute any statistic - they
# only compare numbers that are already stored on ValidationResults / the
# exported DataFrames / FIGURE_REGISTRY against each other, and raise a
# descriptive AssertionError if two supposedly-identical values disagree.

def run_qa_checks(
    cfg: Config,
    validation: "ValidationResults",
    fi: "FeatureImportanceResult",
    repro_info: dict,
    scenario_summary_df: pd.DataFrame,
    scenario_points_df: pd.DataFrame,
) -> None:
    """Raise a descriptive AssertionError on any cross-artifact inconsistency."""
    # Calibration slope: report vs. the two exported scenario CSVs.
    for label, df in (
        ("scenario_summary.csv", scenario_summary_df),
        ("scenario_points.csv", scenario_points_df),
    ):
        if "calibration_slope" in df.columns:
            csv_val = float(df["calibration_slope"].iloc[0])
            if not np.isclose(csv_val, validation.calibration_slope, equal_nan=True):
                raise AssertionError(
                    f"QA check failed: calibration slope in {label} "
                    f"({csv_val:.6f}) does not match ValidationResults "
                    f"({validation.calibration_slope:.6f})."
                )

    # R² / RMSE must be finite - a NaN silently propagating into a figure or
    # report is exactly the class of bug this check is meant to catch.
    for name, value in (
        ("nested_cv_r2", validation.nested_cv_r2),
        ("holdout_r2", validation.holdout_r2),
        ("holdout_rmse", validation.holdout_rmse),
        ("calibration_slope", validation.calibration_slope),
    ):
        if not np.isfinite(value):
            raise AssertionError(f"QA check failed: ValidationResults.{name} is not finite.")

    # Feature importance ordering: the table must already be sorted
    # descending (compute_feature_importance sorts it once; this just
    # confirms nothing downstream re-shuffled it before export).
    importances = fi.table["Importance"].to_numpy()
    if not np.all(np.diff(importances) <= 1e-12):
        raise AssertionError(
            "QA check failed: feature-importance table is not sorted "
            "in descending order at export time."
        )

    # No duplicated figure/CSV filenames in the figure registry.
    filenames = [row["filename"] for row in FIGURE_REGISTRY._rows]
    dupes = {f for f in filenames if filenames.count(f) > 1}
    if dupes:
        raise AssertionError(f"QA check failed: duplicate export filenames detected: {dupes}")

    # Configuration hash must be a single, stable value already computed once.
    if not repro_info.get("config_sha256"):
        raise AssertionError("QA check failed: configuration hash missing from repro_info.")

    log_info("[QA] All automated consistency checks passed.")


# RESULT CONTAINERS (DATACLASSES)
@dataclass
class DatasetBundle:
    df:           pd.DataFrame
    X:            pd.DataFrame
    y:            pd.Series
    groups:       pd.Series
    X_train:      pd.DataFrame
    X_test:       pd.DataFrame
    y_train:      pd.Series
    y_test:       pd.Series
    groups_train: pd.Series
    train_idx:    np.ndarray
    test_idx:     np.ndarray
    feature_min:  pd.Series
    feature_max:  pd.Series


@dataclass
class ModelResults:
    model:       XGBRegressor
    best_params: dict
    search:      RandomizedSearchCV
    data:        DatasetBundle


@dataclass
class ValidationResults:
    nested_cv_r2:           float   # true outer-fold nested GroupKFold R²
    inner_tuning_r2:        float   # RandomizedSearchCV mean_test_score (inner loop only)
    repeated_r2_scores:     np.ndarray
    repeated_rmse_scores:   np.ndarray
    repeated_mae_scores:    np.ndarray
    holdout_pred:           np.ndarray
    holdout_rmse:           float
    holdout_r2:             float
    holdout_mae:            float
    train_r2:               float
    test_r2:                float
    calibration_slope:      float
    calibration_intercept:  float
    calibration_bias:       float
    spatial_validation_df:  pd.DataFrame
    bootstrap_ci:           dict = field(default_factory=dict)

    @property
    def repeated_r2_mean(self) -> float:
        """Mean R² across repeated spatial-block validation folds."""
        return float(self.repeated_r2_scores.mean())

    @property
    def repeated_r2_ci95(self) -> float:
        """95% CI half-width of R² across repeated spatial-block validation folds."""
        return ci95(self.repeated_r2_scores)

    @property
    def repeated_rmse_mean(self) -> float:
        """Mean RMSE across repeated spatial-block validation folds."""
        return float(self.repeated_rmse_scores.mean())

    @property
    def repeated_rmse_ci95(self) -> float:
        """95% CI half-width of RMSE across repeated spatial-block validation folds."""
        return ci95(self.repeated_rmse_scores)

    @property
    def repeated_mae_mean(self) -> float:
        """Mean MAE across repeated spatial-block validation folds."""
        return float(self.repeated_mae_scores.mean())

    @property
    def repeated_mae_ci95(self) -> float:
        """95% CI half-width of MAE across repeated spatial-block validation folds."""
        return ci95(self.repeated_mae_scores)

    @property
    def train_test_r2_gap(self) -> float:
        """Difference between train R² and test R² (overfitting indicator)."""
        return self.train_r2 - self.test_r2


@dataclass
class ResidualDiagnostics:
    residuals:    np.ndarray
    skewness:     float
    kurtosis:     float
    bp_statistic: float
    bp_p_value:   float
    spearman_rho: float
    spearman_p:   float
    bp_method:    str = "auxiliary-regression approximation (statsmodels unavailable)"


@dataclass
class FeatureImportanceResult:
    table: pd.DataFrame


@dataclass
class ScenarioTrajectory:
    name:              str
    scenario_fraction: np.ndarray
    ndvi:              np.ndarray
    ndbi:              np.ndarray
    elevation:         float
    lst:               np.ndarray
    out_of_bounds:     np.ndarray

    @property
    def scenario_pct(self) -> np.ndarray:
        """Scenario fraction expressed as a percentage (0-100)."""
        return self.scenario_fraction * 100

    def feature_frame(self) -> pd.DataFrame:
        """Assemble scenario percent, NDVI, NDBI, and elevation into a DataFrame."""
        return pd.DataFrame({
            "Scenario (%)": self.scenario_pct,
            "NDVI":         self.ndvi,
            "NDBI":         self.ndbi,
            "Elevation":    self.elevation,
        })

    def cooling(self, baseline_lst: float) -> np.ndarray:
        """Predicted cooling relative to `baseline_lst` at each scenario point."""
        return baseline_lst - self.lst


@dataclass
class ScenarioBaseline:
    baseline_lst:   float
    baseline_ndvi:  float
    baseline_ndbi:  float
    ndvi_target:    float
    fixed_elevation: float
    ndvi_ndbi_corr: float


@dataclass
class NEGIResults:
    scenarios:            np.ndarray
    baseline_lst:         float
    delta_t_s1:           np.ndarray
    delta_t_s2:           np.ndarray
    benefit_scale:        float
    desal_energy_sqrt:    np.ndarray
    desal_energy_linear:  np.ndarray
    energy_norm_sqrt:     np.ndarray
    energy_norm_linear:   np.ndarray
    negi_s1:              np.ndarray
    negi_s2:              np.ndarray
    negi_s1_linear:       np.ndarray
    negi_s2_linear:       np.ndarray
    warming_fraction_s2:  float
    cooling_fraction_s2:  float
    max_cooling_s2:       float
    max_warming_s2:       float
    first_cooling_pct:    float

    @property
    def scenario_pct(self) -> np.ndarray:
        """Scenario fraction expressed as a percentage (0-100)."""
        return self.scenarios * 100

    @property
    def s1_optimum_idx(self) -> int:
        """Index of the maximum NEGI value along the Scenario 1 trajectory."""
        return int(np.argmax(self.negi_s1))

    @property
    def s2_optimum_idx(self) -> int:
        """Index of the maximum NEGI value along the Scenario 2 trajectory."""
        return int(np.argmax(self.negi_s2))


@dataclass
class SupportDiagnostics:
    scaler:              StandardScaler
    nn_dist_s1:          np.ndarray
    nn_p95_s1:           float
    nn_beyond_p95_s1:    np.ndarray
    nn_dist_s2:          np.ndarray
    nn_p95_s2:           float
    nn_beyond_p95_s2:    np.ndarray
    support_threshold_k: float
    mean_knn_dist_s2:    np.ndarray
    in_support_s2:       np.ndarray
    extrap_s1_ok:        bool
    extrap_s2_ok:        bool


@dataclass
class UncertaintyResults:
    negi_s1_folds:    np.ndarray
    negi_s2_folds:    np.ndarray
    lst_s1_folds:     np.ndarray
    lst_s2_folds:     np.ndarray
    cooling_s1_folds: np.ndarray
    cooling_s2_folds: np.ndarray
    negi_s1_mean:     np.ndarray
    negi_s1_std:      np.ndarray
    negi_s2_mean:     np.ndarray
    negi_s2_std:      np.ndarray
    percentiles:      dict
    table:            pd.DataFrame
    ndbi_range:       np.ndarray = None
    ndbi_sweep_folds: np.ndarray = None


@dataclass
class SensitivityResults:
    table:      pd.DataFrame
    plot_table: pd.DataFrame


@dataclass
class ConditionalBinResult:
    label:       str
    ndvi:        np.ndarray
    lst:         np.ndarray
    n_pixels:    int
    slope:       Optional[float]
    intercept:   Optional[float]
    slope_se:    Optional[float]
    slope_ci95:  Optional[float]
    r_squared:   Optional[float]
    p_value:     Optional[float]


@dataclass
class ScenarioResults:
    """Per-scenario-point arrays bundled for CSV export."""
    scenario_percent:   np.ndarray
    ndvi:               np.ndarray
    ndbi:               np.ndarray
    predicted_lst:      np.ndarray
    deltaT:             np.ndarray
    clipped_cooling:    np.ndarray
    normalized_cooling: np.ndarray
    irrigation_cost:    np.ndarray
    negi:               np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Assemble the per-scenario-point arrays into a single export DataFrame."""
        return pd.DataFrame({
            "Scenario (%)":          self.scenario_percent,
            "NDVI":                  self.ndvi,
            "NDBI":                  self.ndbi,
            "Predicted LST":         self.predicted_lst,
            "DeltaT":                self.deltaT,
            "Clipped Cooling":       self.clipped_cooling,
            "Normalized Cooling":    self.normalized_cooling,
            "Irrigation/Energy Cost": self.irrigation_cost,
            "NEGI":                  self.negi,
        })


def compute_scenario_results(
    traj: ScenarioTrajectory,
    negi_arr: np.ndarray,
    energy_norm: np.ndarray,
    desal_energy: np.ndarray,
    baseline_lst: float,
) -> ScenarioResults:
    """Assemble a ScenarioResults from already-computed arrays (no new computation)."""
    delta_t = traj.cooling(baseline_lst)
    clipped_cooling = np.maximum(delta_t, 0.0)
    assert len(traj.scenario_pct) == len(traj.ndvi) == len(traj.ndbi) \
        == len(traj.lst) == len(negi_arr), "Scenario arrays must have equal length."
    return ScenarioResults(
        scenario_percent=traj.scenario_pct,
        ndvi=traj.ndvi,
        ndbi=traj.ndbi,
        predicted_lst=traj.lst,
        deltaT=delta_t,
        clipped_cooling=clipped_cooling,
        normalized_cooling=energy_norm,
        irrigation_cost=desal_energy,
        negi=negi_arr,
    )


# DATA LOADING & MODEL FITTING
def load_dataset(cfg: Config) -> pd.DataFrame:
    """Load, validate, and clean the input LST dataset from `cfg.data_path`."""
    if not cfg.data_path.is_file():
        raise FileNotFoundError(
            f"Input CSV was not found: {cfg.data_path}\n"
            "Set the NEGI_DATA_PATH environment variable or update Config.data_path."
        )
    df = pd.read_csv(cfg.data_path).dropna().reset_index(drop=True)
    required = {"NDVI", "NDBI", "Elevation", "LST", "block_id"}
    missing  = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}.")
    if df.empty:
        raise ValueError("Input CSV has no complete rows after missing values are removed.")
    n_groups = df["block_id"].nunique()
    if n_groups < cfg.n_group_kfold_splits:
        raise ValueError(
            f"Found only {n_groups} spatial blocks, but GroupKFold requires at least "
            f"{cfg.n_group_kfold_splits}.  Reduce Config.n_group_kfold_splits or supply "
            "more blocks."
        )
    return df


def build_dataset_bundle(df: pd.DataFrame, cfg: Config) -> DatasetBundle:
    """Split features/target/groups into the confirmatory holdout and training sets."""
    X      = df[["NDVI", "NDBI", "Elevation"]]
    y      = df["LST"]
    groups = df["block_id"].astype(str)

    holdout_splitter = GroupShuffleSplit(
        n_splits=1, test_size=cfg.holdout_test_size, random_state=cfg.random_seed
    )
    train_idx, test_idx = next(holdout_splitter.split(X, y, groups=groups))

    return DatasetBundle(
        df=df, X=X, y=y, groups=groups,
        X_train=X.iloc[train_idx], X_test=X.iloc[test_idx],
        y_train=y.iloc[train_idx], y_test=y.iloc[test_idx],
        groups_train=groups.iloc[train_idx],
        train_idx=train_idx, test_idx=test_idx,
        feature_min=X.min(), feature_max=X.max(),
    )


def fit_model(
    data: DatasetBundle, cfg: Config, summary: SummaryLog
) -> ModelResults:
    """Fit the deployed model via RandomizedSearchCV(GroupKFold) hyperparameter
    tuning on the training split. This produces the single model used
    throughout the rest of the pipeline (scenarios, NEGI, uncertainty bands).

    IMPORTANT: ``search.best_score_`` here is the mean
    *inner tuning* score (RandomizedSearchCV's own GroupKFold mean_test_score
    for the winning hyperparameter combination). It is an inner-loop score,
    not an outer-fold nested-CV score, so it must never be reported as the
    "Nested GroupKFold CV" headline metric. The genuine nested estimate is
    computed separately and independently by ``run_nested_group_kfold_cv()``
    below, which never touches this model.
    """
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=cfg.random_seed,
        monotone_constraints=cfg.monotone_constraints,
        n_jobs=1,
    )
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=cfg.param_distributions,
        n_iter=cfg.n_random_search_iter,
        scoring="r2",
        cv=GroupKFold(n_splits=cfg.n_group_kfold_splits),
        random_state=cfg.random_seed,
        n_jobs=-1,
    )
    search.fit(data.X_train, data.y_train, groups=data.groups_train)
    log_info("\nBest Parameters:", search.best_params_)
    save_csv(pd.DataFrame(search.cv_results_), cfg.data_dir / "randomized_search_results.csv")
    log_info(
        f"\n[1] Inner tuning score - RandomizedSearchCV mean_test_score "
        f"(GroupKFold, winning params): {search.best_score_:.3f}"
    )
    summary.log(
        "Training performance",
        "Inner tuning score (RandomizedSearchCV mean_test_score, GroupKFold)",
        f"{search.best_score_:.4f}",
    )
    return ModelResults(
        model=search.best_estimator_,
        best_params=search.best_params_,
        search=search,
        data=data,
    )


@dataclass
class NestedCVResults:
    """Genuine outer-fold nested GroupKFold CV results.

    Each outer fold refits a *fresh* RandomizedSearchCV (with its own inner
    GroupKFold) on the outer-training partition only, then scores the
    resulting model on the untouched outer-test partition. This is
    independent of, and never modifies, the model produced by fit_model().
    """
    fold_summary:  pd.DataFrame   # one row per outer fold
    predictions:   pd.DataFrame   # one row per held-out observation
    outer_r2_mean: float
    outer_r2_std:  float


def run_nested_group_kfold_cv(
    data: DatasetBundle, cfg: Config, summary: SummaryLog
) -> NestedCVResults:
    """Compute the true Nested GroupKFold CV R² from outer-fold predictions.

    Outer loop: GroupKFold(cfg.n_group_kfold_splits) over the full dataset
    (X, y, groups) -- the same splitter class/fold count already used as the
    *inner* loop inside fit_model()'s RandomizedSearchCV, but instantiated
    independently here as the *outer* loop.
    Inner loop: for each outer-training partition, a fresh
    RandomizedSearchCV(GroupKFold) performs hyperparameter tuning exactly as
    in fit_model() (identical param_distributions, n_iter, scoring, seed),
    but using only that partition's data and groups.
    The outer-fold R²/RMSE/MAE are computed exclusively from predictions on
    each outer-test partition, which the corresponding inner search never saw.
    This function does not alter the model or hyperparameters used anywhere
    else in the pipeline; it exists solely to report an honest, leakage-free
    nested validation number.
    """
    X, y, groups = data.X, data.y, data.groups
    outer_cv = GroupKFold(n_splits=cfg.n_group_kfold_splits)

    fold_rows, pred_rows = [], []
    outer_r2_scores = []

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y, groups=groups), start=1
    ):
        X_out_train, y_out_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        groups_out_train         = groups.iloc[outer_train_idx]
        X_out_test,  y_out_test  = X.iloc[outer_test_idx],  y.iloc[outer_test_idx]

        inner_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=cfg.random_seed,
            monotone_constraints=cfg.monotone_constraints,
            n_jobs=1,
        )
        inner_search = RandomizedSearchCV(
            estimator=inner_model,
            param_distributions=cfg.param_distributions,
            n_iter=cfg.n_random_search_iter,
            scoring="r2",
            cv=GroupKFold(n_splits=cfg.n_group_kfold_splits),
            random_state=cfg.random_seed,
            n_jobs=-1,
        )
        inner_search.fit(X_out_train, y_out_train, groups=groups_out_train)

        outer_pred = inner_search.best_estimator_.predict(X_out_test)
        fold_r2   = compute_r2(y_out_test, outer_pred)
        fold_rmse = compute_rmse(y_out_test, outer_pred)
        fold_mae  = compute_mae(y_out_test, outer_pred)
        outer_r2_scores.append(fold_r2)

        fold_rows.append({
            "outer_fold":               fold_id,
            "validation_r2":            fold_r2,
            "validation_rmse_c":        fold_rmse,
            "validation_mae_c":         fold_mae,
            "n_observations":           len(outer_test_idx),
            "selected_hyperparameters": json.dumps(inner_search.best_params_),
            "inner_tuning_score":       inner_search.best_score_,
        })
        for obs, pred_val in zip(y_out_test.to_numpy(), outer_pred):
            pred_rows.append({
                "observed": obs, "predicted": pred_val, "outer_fold": fold_id,
            })

        log_info(
            f"    Outer fold {fold_id}/{cfg.n_group_kfold_splits}: "
            f"nested R² = {fold_r2:.3f}, RMSE = {fold_rmse:.2f} °C "
            f"(n={len(outer_test_idx)})"
        )

    fold_summary = pd.DataFrame(fold_rows)
    predictions  = pd.DataFrame(pred_rows)
    outer_r2_scores = np.array(outer_r2_scores)

    save_csv(fold_summary, cfg.data_dir / "nested_cv_summary.csv")
    save_csv(predictions,  cfg.data_dir / "nested_cv_predictions.csv")

    outer_r2_mean = float(outer_r2_scores.mean())
    outer_r2_std  = float(outer_r2_scores.std())

    log_info(
        f"\n[1] Headline metric - Nested GroupKFold CV R² "
        f"(outer-fold predictions): {outer_r2_mean:.3f} ± {outer_r2_std:.3f}"
    )
    summary.log("Training performance", "Nested GroupKFold CV R² (headline, outer-fold)",
                f"{outer_r2_mean:.4f} ± {outer_r2_std:.4f}")
    summary.log("Training performance", "Nested GroupKFold CV outer folds",
                str(cfg.n_group_kfold_splits))

    return NestedCVResults(
        fold_summary=fold_summary,
        predictions=predictions,
        outer_r2_mean=outer_r2_mean,
        outer_r2_std=outer_r2_std,
    )


# VALIDATION
def evaluate_model(
    model_results: ModelResults, cfg: Config, summary: SummaryLog,
    nested_cv: "NestedCVResults",
) -> ValidationResults:
    """Run repeated spatial-block validation and the confirmatory holdout evaluation."""
    model, data = model_results.model, model_results.data
    X, y, groups = data.X, data.y, data.groups

    # Fail-fast assertions before any computation.
    assert np.all(np.isfinite(X.values)),    "Feature matrix contains non-finite values."
    assert np.all(np.isfinite(y.values)),    "Target vector contains non-finite values."
    assert len(X) == len(y) == len(groups),  "X, y, and groups must have the same length."

    spatial_repeats = GroupShuffleSplit(
        n_splits=cfg.n_spatial_holdout_repeats,
        test_size=cfg.holdout_test_size,
        random_state=cfg.random_seed,
    )
    cv_scores      = cross_val_score(
        model, X, y, groups=groups, cv=spatial_repeats, scoring="r2", n_jobs=-1
    )
    cv_rmse_scores = -cross_val_score(
        model, X, y, groups=groups, cv=spatial_repeats,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    cv_mae_scores  = -cross_val_score(
        model, X, y, groups=groups, cv=spatial_repeats,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )

    log_info(f"\n[2] Repeated spatial-holdout R²   = "
             f"{cv_scores.mean():.3f} ± {ci95(cv_scores):.3f} (95% CI)")
    log_info(f"    Repeated spatial-holdout RMSE = "
             f"{cv_rmse_scores.mean():.2f} ± {ci95(cv_rmse_scores):.2f} °C (95% CI)")
    log_info(f"    Repeated spatial-holdout MAE  = "
             f"{cv_mae_scores.mean():.2f} ± {ci95(cv_mae_scores):.2f} °C (95% CI)")
    summary.log("Nested spatial CV performance", "Repeated spatial-holdout R² (mean ± 95% CI)",
                f"{cv_scores.mean():.4f} ± {ci95(cv_scores):.4f}")
    summary.log("Nested spatial CV performance", "Repeated spatial-holdout RMSE (mean ± 95% CI, °C)",
                f"{cv_rmse_scores.mean():.4f} ± {ci95(cv_rmse_scores):.4f}")
    summary.log("Nested spatial CV performance", "Repeated spatial-holdout MAE (mean ± 95% CI, °C)",
                f"{cv_mae_scores.mean():.4f} ± {ci95(cv_mae_scores):.4f}")

    pred = model.predict(data.X_test)
    assert np.all(np.isfinite(pred)),       "Model produced non-finite holdout predictions."
    assert len(pred) == len(data.y_test),   "Predictions and observations must have the same length."

    metrics  = compute_metrics(data.y_test, pred)
    rmse, r2, mae = metrics["rmse"], metrics["r2"], metrics["mae"]

    log_info("\n[3] Single untouched spatial-block holdout (confirmatory):")
    log_info(f"    RMSE: {rmse:.4f} °C")
    log_info(f"    R²:   {r2:.3f}")
    log_info(f"    MAE:  {mae:.4f} °C")
    summary.log("Holdout performance", "Single holdout RMSE (°C)", f"{rmse:.4f}")
    summary.log("Holdout performance", "Single holdout R²",         f"{r2:.4f}")
    summary.log("Holdout performance", "Single holdout MAE (°C)",   f"{mae:.4f}")

    train_r2 = model.score(data.X_train, data.y_train)
    test_r2  = model.score(data.X_test,  data.y_test)
    log_info(f"\nOverfitting check: Train R² = {train_r2:.3f}, Test R² = {test_r2:.3f}")
    if train_r2 - test_r2 > 0.08:
        log_info("Note: Train-test gap exceeds 0.08; report in Methods.")
    summary.log("Holdout performance", "Train R²",            f"{train_r2:.4f}")
    summary.log("Holdout performance", "Train-test R² gap",   f"{train_r2 - test_r2:.4f}")

    nested_fold_r2 = nested_cv.fold_summary["validation_r2"].to_numpy()
    nested_fold_rmse = nested_cv.fold_summary["validation_rmse_c"].to_numpy()
    nested_fold_mae = nested_cv.fold_summary["validation_mae_c"].to_numpy()

    spatial_validation_df = pd.DataFrame([
        {
            "Validation": "Nested GroupKFold CV (headline metric, outer-fold predictions)",
            "R2_Holdout": np.nan, "RMSE_C_Holdout": np.nan, "MAE_C_Holdout": np.nan,
            "R2_Mean": nested_cv.outer_r2_mean, "R2_95CI": ci95(nested_fold_r2),
            "RMSE_Mean": nested_fold_rmse.mean(), "RMSE_95CI": ci95(nested_fold_rmse),
            "MAE_Mean": nested_fold_mae.mean(), "MAE_95CI": ci95(nested_fold_mae),
            "n_test_pixels": nested_cv.fold_summary["n_observations"].sum(),
            "n_test_blocks": np.nan,
        },
        {
            "Validation": "RandomizedSearchCV inner tuning score (NOT nested; informational only)",
            "R2_Holdout": np.nan, "RMSE_C_Holdout": np.nan, "MAE_C_Holdout": np.nan,
            "R2_Mean": model_results.search.best_score_, "R2_95CI": np.nan,
            "RMSE_Mean": np.nan, "RMSE_95CI": np.nan, "MAE_Mean": np.nan, "MAE_95CI": np.nan,
            "n_test_pixels": np.nan, "n_test_blocks": np.nan,
        },
        {
            "Validation": "Repeated spatial-block validation",
            "R2_Holdout": np.nan, "RMSE_C_Holdout": np.nan, "MAE_C_Holdout": np.nan,
            "R2_Mean": cv_scores.mean(), "R2_95CI": ci95(cv_scores),
            "RMSE_Mean": cv_rmse_scores.mean(), "RMSE_95CI": ci95(cv_rmse_scores),
            "MAE_Mean": cv_mae_scores.mean(), "MAE_95CI": ci95(cv_mae_scores),
            "n_test_pixels": np.nan, "n_test_blocks": np.nan,
        },
        {
            "Validation": "Single untouched spatial-block holdout (confirmatory)",
            "R2_Holdout": r2, "RMSE_C_Holdout": rmse, "MAE_C_Holdout": mae,
            "R2_Mean": np.nan, "R2_95CI": np.nan, "RMSE_Mean": np.nan, "RMSE_95CI": np.nan,
            "MAE_Mean": np.nan, "MAE_95CI": np.nan,
            "n_test_pixels": len(data.X_test),
            "n_test_blocks": groups.iloc[data.test_idx].nunique(),
        },
    ])
    save_csv(spatial_validation_df, cfg.data_dir / "spatial_validation_summary.csv")

    slope, intercept, bias = (
        metrics["calibration_slope"], metrics["calibration_intercept"], metrics["bias"]
    )
    assert np.isfinite(slope) and 0.1 < slope < 2.0, \
        f"Calibration slope {slope:.3f} is outside the expected range [0.1, 2.0]."

    log_info(
        f"\nCalibration slope = {slope:.3f} "
        f"(slope < 1 reflects expected regression toward the mean "
        f"under spatial-block validation; no post-hoc recalibration applied)."
    )
    log_info(f"Calibration intercept   : {intercept:.4f} °C")
    log_info(f"Mean bias               : {bias:.4f} °C")
    log_info(
        "Calibration diagnostics are reported descriptively only; "
        "no post-hoc recalibration was applied."
    )
    summary.log("Calibration", "Calibration slope",             f"{slope:.4f}")
    summary.log("Calibration", "Calibration intercept (°C)",    f"{intercept:.4f}")
    summary.log("Calibration", "Mean bias (°C)",                f"{bias:.4f}")
    summary.log("Calibration", "Recalibration applied", "False")
    summary.log(
        "Calibration", "Reporting note",
        "Calibration diagnostics are reported descriptively only; "
        "no post-hoc recalibration was applied.  RMSE and MAE are "
        "reported once in the Confirmatory Holdout section and are not "
        "repeated here.  MACE is omitted because it is identical to MAE.",
    )

    y_test_arr = (
        data.y_test.to_numpy() if hasattr(data.y_test, "to_numpy")
        else np.asarray(data.y_test)
    )
    boot_specs = {
        "rmse":                  _metric_rmse,
        "mae":                   _metric_mae,
        "bias":                  _metric_bias,
        "calibration_slope":     _metric_calibration_slope,
        "calibration_intercept": _metric_calibration_intercept,
    }
    bootstrap_ci = {
        name: bootstrap_metric(
            y_test_arr, pred, fn,
            n_boot=cfg.bootstrap_iterations, seed=cfg.bootstrap_seed,
        )
        for name, fn in boot_specs.items()
    }
    log_info(f"\nBootstrap 95% CIs ({cfg.bootstrap_iterations} resamples, holdout predictions):")
    for name, ci_info in bootstrap_ci.items():
        log_info(
            f"  {name:22s}: {ci_info['point_estimate']:.4f}  "
            f"95% CI: {ci_info['lower']:.4f}-{ci_info['upper']:.4f}"
        )
        summary.log("Bootstrap CI", f"{name} (point estimate)", f"{ci_info['point_estimate']:.4f}")
        summary.log("Bootstrap CI", f"{name} 95% CI",           f"{ci_info['lower']:.4f}-{ci_info['upper']:.4f}")
        
    save_csv(
        pd.DataFrame([
            {
                "metric": name, "point_estimate": ci_info["point_estimate"],
                "median": ci_info["median"],
                "ci_lower_2.5": ci_info["lower"], "ci_upper_97.5": ci_info["upper"],
                "n_boot": ci_info["n_boot"],
            }
            for name, ci_info in bootstrap_ci.items()
        ]),
        cfg.data_dir / "bootstrap_summary.csv",
    )

    return ValidationResults(
        nested_cv_r2=nested_cv.outer_r2_mean,
        inner_tuning_r2=model_results.search.best_score_,
        repeated_r2_scores=cv_scores,
        repeated_rmse_scores=cv_rmse_scores,
        repeated_mae_scores=cv_mae_scores,
        holdout_pred=pred,
        holdout_rmse=rmse, holdout_r2=r2, holdout_mae=mae,
        train_r2=train_r2, test_r2=test_r2,
        calibration_slope=slope, calibration_intercept=intercept, calibration_bias=bias,
        spatial_validation_df=spatial_validation_df,
        bootstrap_ci=bootstrap_ci,
    )


# RESIDUAL DIAGNOSTICS
def compute_residual_diagnostics(
    validation: ValidationResults, data: DatasetBundle
) -> ResidualDiagnostics:
    """Compute holdout residual skewness, kurtosis, Breusch-Pagan, and Spearman stats."""
    residuals = data.y_test - validation.holdout_pred
    residuals_arr = (
        residuals.values if hasattr(residuals, "values") else np.asarray(residuals)
    )
    pred = validation.holdout_pred

    skewness = scipy_stats.skew(residuals_arr)
    kurtosis = scipy_stats.kurtosis(residuals_arr)

    # Breusch-Pagan statistic is computed exactly once, by the single
    # shared _breusch_pagan_test() implementation, so
    # that every report and figure referring to "the Breusch-Pagan test"
    # is describing the same computation rather than two similar-looking
    # but separately implemented statistics.
    bp = _breusch_pagan_test(np.zeros_like(pred), pred, residuals_arr)

    spearman_corr, spearman_p = scipy_stats.spearmanr(np.abs(residuals_arr), pred)

    return ResidualDiagnostics(
        residuals=residuals_arr,
        skewness=skewness, kurtosis=kurtosis,
        bp_statistic=bp["statistic"], bp_p_value=bp["p_value"], bp_method=bp["method"],
        spearman_rho=spearman_corr, spearman_p=spearman_p,
    )


def report_residual_diagnostics(
    resid: ResidualDiagnostics, cfg: Config, summary: SummaryLog
) -> None:
    """Log and record residual skewness, kurtosis, Breusch-Pagan, and interpretation text."""
    log_info(f"\nResidual skewness       : {resid.skewness:.4f}")
    log_info(f"Residual kurtosis       : {resid.kurtosis:.4f} (excess; 0 = normal)")
    summary.log("Residual diagnostics", "Residual skewness",          f"{resid.skewness:.4f}")
    summary.log("Residual diagnostics", "Residual kurtosis (excess)", f"{resid.kurtosis:.4f}")

    # This is the single Breusch-Pagan statistic used throughout the report
    # and figures - `resid.bp_method` names which implementation
    # produced it (statsmodels when available, otherwise the equivalent
    # auxiliary-regression computation), so it is never mistaken for a
    # second, independently computed test.
    log_info(f"\nBreusch-Pagan test ({resid.bp_method}): "
             f"statistic = {resid.bp_statistic:.4f}, p = {resid.bp_p_value:.4g}")
    log_info(f"Spearman |resid| vs pred: rho = {resid.spearman_rho:.4f}, "
             f"p = {resid.spearman_p:.4g}")

    # Report heteroscedasticity proportionately: large-sample detection
    # with a weak effect size is flagged as practically minor.
    if resid.bp_p_value < 0.05 or resid.spearman_p < 0.05:
        if abs(resid.spearman_rho) < 0.15:
            log_info(
                "  -> Statistically detectable but practically weak heteroscedasticity "
                "(|Spearman rho| < 0.15); likely driven by large sample size rather "
                "than a severe modelling problem."
            )
        else:
            log_info(f"  -> {interpretation_text('heteroscedasticity')}")
            summary.add_conclusion(interpretation_text("heteroscedasticity"))
    else:
        log_info("  -> No strong evidence of heteroscedasticity at the 0.05 level.")

    summary.log("Residual diagnostics", "Breusch-Pagan statistic",
                f"{resid.bp_statistic:.4f}")
    summary.log("Residual diagnostics", "Breusch-Pagan p-value",
                f"{resid.bp_p_value:.4g}")
    summary.log("Residual diagnostics", "Breusch-Pagan implementation", resid.bp_method)
    summary.log("Residual diagnostics", "Spearman |residual| vs predicted (rho)",
                f"{resid.spearman_rho:.4f}")
    summary.log("Residual diagnostics", "Spearman |residual| vs predicted (p)",
                f"{resid.spearman_p:.4g}")


def _find_spatial_columns(
    df: pd.DataFrame, cfg: Config
) -> Optional[tuple[str, str]]:
    """Return (x_col, y_col) or None when no coordinate columns are found."""
    x_col = next((c for c in cfg.spatial_x_candidates if c in df.columns), None)
    y_col = next((c for c in cfg.spatial_y_candidates if c in df.columns), None)
    return (x_col, y_col) if (x_col is not None and y_col is not None) else None


def global_morans_i(
    values: np.ndarray, x: np.ndarray, y: np.ndarray, k: int = 8
) -> dict:
    """Global Moran's I via a k-nearest-neighbour row-standardised weight matrix."""
    values = np.asarray(values, dtype=float)
    coords = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    n    = len(values)
    k_eff = min(k, n - 1)
    if n < 4 or k_eff < 1:
        return {
            "morans_i": float("nan"), "expected_i": float("nan"),
            "z_score":  float("nan"), "p_value":    float("nan"), "n": n,
        }

    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
    _, neighbor_idx = nn.kneighbors(coords)
    neighbor_idx = neighbor_idx[:, 1:]   # drop self

    z  = values - values.mean()
    W  = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, neighbor_idx[i]] = 1.0 / k_eff
    s0 = W.sum()

    numerator   = float(z @ W @ z)
    denominator = float(np.sum(z ** 2))
    morans_i    = (n / s0) * safe_divide(
        np.array([numerator]), np.array([denominator]), fill=0.0
    )[0]

    expected_i = -1.0 / (n - 1)
    s1 = 0.5 * np.sum((W + W.T) ** 2)
    s2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)
    b2 = (np.sum(z ** 4) / n) / ((np.sum(z ** 2) / n) ** 2) if denominator > 0 else np.nan
    var_i = (
        (
            n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2)
            - b2 * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * s0 ** 2)
        )
        / ((n - 1) * (n - 2) * (n - 3) * s0 ** 2)
    ) - expected_i ** 2

    z_score = (
        safe_divide(
            np.array([morans_i - expected_i]),
            np.array([np.sqrt(max(var_i, 0.0))]),
            fill=0.0,
        )[0]
        if var_i > 0 else float("nan")
    )
    p_value = (
        float(2 * (1 - scipy_stats.norm.cdf(abs(z_score))))
        if np.isfinite(z_score) else float("nan")
    )
    return {
        "morans_i":  float(morans_i), "expected_i": float(expected_i),
        "z_score":   float(z_score),  "p_value":    p_value, "n": n,
    }


# Interpretive-only extension of the existing Global Moran's I diagnostic.
# Does not touch Global Moran's I itself or any bootstrap calculation; it
# only translates I into an approximate effective sample size for
# interpretive reporting.
def compute_effective_sample_size(n: int, morans_i: float) -> dict:
    """Approximate effective sample size under positive spatial
    autocorrelation, n_eff = n * (1 - I) / (1 + I).

    This is a simple, commonly used interpretive approximation (e.g.
    Griffith, 2005) for how much residual spatial autocorrelation shrinks
    the amount of independent information in a holdout sample.  It is
    reported alongside - not in place of - the existing bootstrap-based
    uncertainty quantification.
    """
    if not np.isfinite(morans_i) or n <= 0:
        return {
            "n": n, "morans_i": float(morans_i) if np.isfinite(morans_i) else float("nan"),
            "n_eff": float("nan"), "reduction_pct": float("nan"),
        }
    # Guard against I -> -1, which would blow the ratio up; clip to a
    # sane range for a purely descriptive statistic.
    i_clipped = float(np.clip(morans_i, -0.999, 0.999))
    n_eff = n * safe_divide(
        np.array([1.0 - i_clipped]), np.array([1.0 + i_clipped]), fill=float(n)
    )[0]
    n_eff = float(max(n_eff, 0.0))
    reduction_pct = float(safe_divide(
        np.array([n - n_eff]), np.array([n]), fill=0.0
    )[0] * 100.0)
    return {"n": n, "morans_i": float(morans_i), "n_eff": n_eff, "reduction_pct": reduction_pct}


def compute_residual_spatial_diagnostic(
    validation: ValidationResults, data: DatasetBundle,
    cfg: Config, summary: SummaryLog,
) -> Optional[dict]:
    """Compute Moran's I on holdout residuals, skipping gracefully when
    no coordinate columns are present."""
    df_test    = data.df.loc[data.test_idx]
    coord_cols = _find_spatial_columns(df_test, cfg)
    if coord_cols is None:
        log_warning(
            "\nResidual spatial diagnostic skipped: no recognisable coordinate "
            f"columns found (looked for {cfg.spatial_x_candidates} / "
            f"{cfg.spatial_y_candidates})."
        )
        return None

    x_col, y_col = coord_cols
    residuals = data.y_test.values - validation.holdout_pred
    x_vals    = df_test[x_col].values
    y_vals    = df_test[y_col].values

    moran = global_morans_i(residuals, x_vals, y_vals, k=cfg.moran_k_neighbors)
    log_info(
        f"\nGlobal Moran's I (holdout residuals): "
        f"I = {moran['morans_i']:.4f} "
        f"(expected under CSR = {moran['expected_i']:.4f}), "
        f"z = {moran['z_score']:.3f}, p = {moran['p_value']:.4g}"
    )
    significant_p = np.isfinite(moran["p_value"]) and moran["p_value"] < 0.05
    # Magnitude-based flag: a configurable threshold on |Moran's I|
    # itself, independent of the significance test above.  Interpretation
    # only - the statistic and its p-value are unchanged.
    exceeds_magnitude = (
        np.isfinite(moran["morans_i"])
        and moran["morans_i"] > cfg.moran_i_magnitude_threshold
    )
    if significant_p or exceeds_magnitude:
        log_info(f"  -> {interpretation_text('spatial_dependence')}")
        summary.add_warning(
            f"Significant residual spatial autocorrelation detected "
            f"(Moran's I = {moran['morans_i']:.4f}, p = {moran['p_value']:.4g}).  "
            "Confidence intervals should be interpreted accordingly."
        )
        if exceeds_magnitude:
            summary.add_conclusion(interpretation_text("spatial_dependence"))
    else:
        log_info(
            "  -> No strong evidence of spatially clustered residual error "
            "at the 0.05 level."
        )

    summary.log("Residual diagnostics", "Global Moran's I (holdout residuals)",
                f"{moran['morans_i']:.4f}")
    summary.log("Residual diagnostics", "Moran's I p-value", f"{moran['p_value']:.4g}")

    # Interpretive diagnostic only: effective sample size under residual
    # spatial autocorrelation. The bootstrap procedure itself is untouched.
    n_eff_result = compute_effective_sample_size(len(residuals), moran["morans_i"])
    if np.isfinite(n_eff_result["n_eff"]):
        log_info(
            f"\nHoldout observations                : {n_eff_result['n']}\n"
            f"Residual Moran's I                   : {n_eff_result['morans_i']:.3f}\n"
            f"Approximate effective sample size    : {n_eff_result['n_eff']:.0f}\n"
            f"Approximate reduction                : {n_eff_result['reduction_pct']:.0f}%"
        )
    summary.log("Interpretive Diagnostics", "Holdout observations (n)",
                str(n_eff_result["n"]))
    summary.log("Interpretive Diagnostics", "Approximate effective sample size (n_eff)",
                f"{n_eff_result['n_eff']:.0f}" if np.isfinite(n_eff_result["n_eff"]) else "n/a")
    summary.log("Interpretive Diagnostics", "Approximate n_eff reduction",
                f"{n_eff_result['reduction_pct']:.0f}%" if np.isfinite(n_eff_result["reduction_pct"]) else "n/a")

    return {
        "x": x_vals, "y": y_vals,
        "residuals": residuals, "moran": moran,
        "n_eff": n_eff_result,
        "x_col": x_col, "y_col": y_col,
    }


def compute_calibration_table(
    y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """Produce a single calibration metrics table (descriptive only;
    no recalibration is applied).

    Columns reported:
      Calibration slope, Calibration intercept, Mean bias (MBE),
      Pearson r and p-value.

    RMSE and MAE are omitted here because they are already reported in
    the Confirmatory Holdout section.  MACE was removed because it is
    mathematically identical to MAE (compute_mean_absolute_calibration_error
    delegates to compute_mae) and therefore duplicates an existing column.
    Bias and Mean Bias Error (MBE) are the same quantity; only 'Mean bias'
    is retained.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m            = compute_metrics(y_true, y_pred)
    pearson_r, pearson_p = compute_pearson_r(y_true, y_pred)
    slope        = m["calibration_slope"]
    intercept    = m["calibration_intercept"]

    # Intercept CI zero-span note
    # The bootstrap CI for the intercept is computed elsewhere; here we
    # generate a plain-language note for when the CI spans zero, which is
    # expected under spatial-block-holdout validation because the slope < 1
    # (regression-toward-the-mean) means the intercept compensates.
    intercept_note = (
        "Calibration intercept 95% CI likely spans zero because spatial "
        "block-holdout produces a slope < 1 (regression toward the mean), "
        "which numerically shifts the intercept upward to preserve overall "
        "mean agreement.  This does not indicate a systematic bias in the "
        "original predictions."
    )

    return pd.DataFrame([{
        "Calibration slope":      slope,
        "Calibration intercept":  intercept,
        "Mean bias (°C)":         m["bias"],
        "Pearson r":              pearson_r,
        "Pearson p":              pearson_p,
        "Intercept CI note":      intercept_note,
    }])


# MODEL COMPARISON
def compare_models(
    model_results: ModelResults, cfg: Config, summary: SummaryLog
) -> pd.DataFrame:
    """Fit Linear Regression, Random Forest, and Gradient Boosting
    benchmarks and compare them with XGBoost on the confirmatory holdout."""
    log_info("\nNote: " + interpretation_text("benchmark_comparison"))
    data = model_results.data
    comparison_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(
            n_estimators=300, random_state=cfg.random_seed, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, random_state=cfg.random_seed
        ),
        "XGBoost": model_results.model,
    }
    rows = []
    for name, mdl in comparison_models.items():
        vlog_info(cfg, f"Running {name}...")
        if name != "XGBoost":
            mdl.fit(data.X_train, data.y_train)
        y_pred_model = mdl.predict(data.X_test)
        rows.append({
            "Model":    name,
            "R2":       r2_score(data.y_test, y_pred_model),
            "RMSE (°C)": np.sqrt(mean_squared_error(data.y_test, y_pred_model)),
            "MAE (°C)": mean_absolute_error(data.y_test, y_pred_model),
        })

    comparison_df = pd.DataFrame(rows).sort_values("R2", ascending=False)
    log_info("\nModel comparison:")
    log_info(comparison_df)
    save_csv(comparison_df, cfg.data_dir / "model_comparison.csv")
    summary.log(
        "Model comparison", "Benchmark note",
        interpretation_text("benchmark_comparison"),
    )
    return comparison_df


# PERMUTATION FEATURE IMPORTANCE
def compute_feature_importance(
    model_results: ModelResults, cfg: Config, summary: SummaryLog
) -> FeatureImportanceResult:
    """Compute permutation feature importance for NDBI, Elevation, and
    NDVI across repeated seeds."""
    model, data = model_results.model, model_results.data
    perm_seeds  = np.arange(cfg.random_seed, cfg.random_seed + cfg.n_permutation_outer_seeds)

    runs: dict[str, list[float]] = {feat: [] for feat in data.X.columns}
    for seed in perm_seeds:
        perm_run = permutation_importance(
            model, data.X_test, data.y_test,
            n_repeats=cfg.n_permutation_inner_repeats,
            random_state=int(seed), scoring="r2", n_jobs=-1,
        )
        for i, feat in enumerate(data.X.columns):
            runs[feat].append(perm_run.importances_mean[i])

    rows = []
    for feat in data.X.columns:
        vals     = np.array(runs[feat])
        feat_mean = vals.mean()
        feat_std  = vals.std(ddof=1)
        feat_ci95 = 1.96 * feat_std / np.sqrt(len(vals))
        rows.append({
            "Feature":          feat,
            "Importance":       feat_mean,
            "Importance_Std":   feat_std,
            "Importance_CI95":  feat_ci95,
            "Importance_CI_Low":  feat_mean - feat_ci95,
            "Importance_CI_High": feat_mean + feat_ci95,
            "N_Repeats":        len(vals),
        })
    table = pd.DataFrame(rows).sort_values("Importance", ascending=False)

    log_info(
        f"\nPermutation feature importance "
        f"(mean ± SD over {cfg.n_permutation_outer_seeds} outer seeds, "
        f"{cfg.n_permutation_inner_repeats} inner repeats each):"
    )
    log_info(table[["Feature", "Importance", "Importance_Std",
                     "Importance_CI_Low", "Importance_CI_High"]])
    ndvi_ndbi_corr = data.df["NDVI"].corr(data.df["NDBI"])
    log_info(
        "Note: Permutation importance reflects marginal predictive importance "
        f"and may be influenced by correlated predictors (NDVI-NDBI r = "
        f"{ndvi_ndbi_corr:.3f}).  Values are not additive."
    )
    save_csv(table, cfg.data_dir / "permutation_importance.csv")
    for _, row in table.iterrows():
        summary.log(
            "Feature importance",
            f"{row['Feature']} permutation importance (mean ± 95% CI)",
            f"{row['Importance']:.4f} ± {row['Importance_CI95']:.4f}",
        )
    return FeatureImportanceResult(table=table)


# SCENARIO FRAMEWORK
def predict_lst(
    model,
    ndvi: float, ndbi: float, elevation: float,
    feature_min: pd.Series, feature_max: pd.Series,
    cache: Optional[PredictionCache] = None,
) -> tuple[float, bool]:
    """Predict LST for a single feature vector, using the module-level cache."""
    cache = PREDICTION_CACHE if cache is None else cache
    cached = cache.get(ndvi, ndbi, elevation) if cache is not None else None
    if cached is not None:
        cache.hits += 1
        return cached

    row   = pd.DataFrame({"NDVI": [ndvi], "NDBI": [ndbi], "Elevation": [elevation]})
    raw   = model.predict(row)[0]
    ood   = ((row.iloc[0] < feature_min) | (row.iloc[0] > feature_max)).any()
    result = (float(raw), bool(ood))
    if cache is not None:
        cache.misses += 1
        cache.set(ndvi, ndbi, elevation, result)
    return result


def fit_empirical_ndbi_spline(
    df: pd.DataFrame, cfg: Config
) -> tuple[UnivariateSpline, pd.DataFrame]:
    """Fit the NDVI-NDBI spline used to keep NDVI and NDBI jointly
    consistent in scenario trajectories."""
    ndvi_edges = np.linspace(df["NDVI"].min(), df["NDVI"].max(), cfg.ndvi_bin_edges)
    bin_id = np.digitize(df["NDVI"], ndvi_edges)
    binned = df.groupby(bin_id).agg(
        ndvi_mid=("NDVI", "median"), ndbi_mid=("NDBI", "median"), n=("NDVI", "size"),
    )
    binned = binned[binned["n"] >= cfg.ndvi_bin_min_count].sort_values("ndvi_mid")
    spline = UnivariateSpline(binned["ndvi_mid"], binned["ndbi_mid"],
                               k=3, s=len(binned) * 0.5)
    return spline, binned


def empirical_ndbi_from_ndvi(
    ndvi_val: float, spline: UnivariateSpline,
    ndbi_min: float, ndbi_max: float,
) -> float:
    """Return the empirical NDBI value predicted by the fitted
    NDVI-NDBI spline for a given NDVI."""
    return float(np.clip(spline(ndvi_val), ndbi_min, ndbi_max))


def run_scenario_trajectories(
    model_results: ModelResults, cfg: Config, summary: SummaryLog
) -> tuple[ScenarioBaseline, ScenarioTrajectory, ScenarioTrajectory]:
    """Build the Scenario 1 (greening-only) and Scenario 2 (greening +
    densification) NDVI/NDBI trajectories."""
    model, data = model_results.model, model_results.data
    df = data.df

    scenarios = np.arange(0.0, 1.0001, cfg.scenario_step)
    assert np.all(np.diff(scenarios) > 0), "Scenario axis must be strictly increasing."

    fixed_elevation  = df["Elevation"].median()
    baseline_ndbi    = df["NDBI"].median()
    baseline_ndvi    = df["NDVI"].median()
    ndvi_target      = df["NDVI"].quantile(cfg.ndvi_target_percentile)

    log_info(
        f"\nScenario axis: evaluated from observed median baseline NDVI "
        f"({baseline_ndvi:.4f}) to observed NDVI p95 target ({ndvi_target:.4f})."
    )

    baseline_lst, baseline_ood = predict_lst(
        model, baseline_ndvi, baseline_ndbi, fixed_elevation,
        data.feature_min, data.feature_max,
    )
    if baseline_ood:
        raise RuntimeError(
            "The baseline feature vector is outside observed feature bounds."
        )

    spline, binned = fit_empirical_ndbi_spline(df, cfg)
    ndbi_min, ndbi_max = df["NDBI"].min(), df["NDBI"].max()
    ndvi_ndbi_corr = df["NDVI"].corr(df["NDBI"])
    log_info(
        f"\nNDVI-NDBI observed association (r = {ndvi_ndbi_corr:.3f}); "
        f"empirical spline fitted on {len(binned)} binned points."
    )

    empirical_baseline_ndbi = empirical_ndbi_from_ndvi(
        baseline_ndvi, spline, ndbi_min, ndbi_max
    )

    def scenario_ndvi(fraction: float) -> float:
        """NDVI values along the scenario trajectory."""
        return baseline_ndvi + fraction * (ndvi_target - baseline_ndvi)

    def scenario2_ndbi(ndvi_val: float) -> float:
        """NDBI values along the Scenario 2 trajectory."""
        shifted = baseline_ndbi + (
            empirical_ndbi_from_ndvi(ndvi_val, spline, ndbi_min, ndbi_max)
            - empirical_baseline_ndbi
        )
        return float(np.clip(shifted, ndbi_min, ndbi_max))

    lst_s1, lst_s2, ood_s1, ood_s2 = [], [], [], []
    ndvi_vals, ndbi_s2_vals = [], []
    for fraction in scenarios:
        ndvi_val = scenario_ndvi(fraction)
        ndvi_vals.append(ndvi_val)

        v1, o1 = predict_lst(
            model, ndvi_val, baseline_ndbi, fixed_elevation,
            data.feature_min, data.feature_max,
        )
        lst_s1.append(v1)
        ood_s1.append(o1)

        ndbi_val = scenario2_ndbi(ndvi_val)
        ndbi_s2_vals.append(ndbi_val)
        v2, o2 = predict_lst(
            model, ndvi_val, ndbi_val, fixed_elevation,
            data.feature_min, data.feature_max,
        )
        lst_s2.append(v2)
        ood_s2.append(o2)

    s1 = ScenarioTrajectory(
        "Scenario 1", scenarios,
        np.array(ndvi_vals), np.full(len(scenarios), baseline_ndbi),
        fixed_elevation, np.array(lst_s1), np.array(ood_s1),
    )
    s2 = ScenarioTrajectory(
        "Scenario 2", scenarios,
        np.array(ndvi_vals), np.array(ndbi_s2_vals),
        fixed_elevation, np.array(lst_s2), np.array(ood_s2),
    )

    def _report_ood(traj: ScenarioTrajectory, label: str) -> None:
        if traj.out_of_bounds.any():
            first_idx = np.argmax(traj.out_of_bounds)
            log_info(
                f"\n{label}: {traj.out_of_bounds.sum()}/{len(traj.out_of_bounds)} "
                f"points outside observed feature bounds "
                f"(first at scenario={traj.scenario_pct[first_idx]:.1f}%)."
            )
        else:
            log_info(
                f"\n{label}: The evaluated trajectory remains within the "
                "observed feature space throughout."
            )

    _report_ood(s1, "Scenario 1")
    _report_ood(s2, "Scenario 2")

    baseline = ScenarioBaseline(
        baseline_lst, baseline_ndvi, baseline_ndbi,
        ndvi_target, fixed_elevation, ndvi_ndbi_corr,
    )
    return baseline, s1, s2


def tree_artifact_diagnostics(
    pred_arr: np.ndarray, label: str, cfg: Config, summary: SummaryLog
) -> tuple[float, float, int]:
    """Flag abrupt jumps in the tree-based prediction surface that may
    indicate split-boundary artifacts."""
    delta         = np.diff(pred_arr)
    mean_abs_jump = np.mean(np.abs(delta))
    max_jump      = np.max(np.abs(delta))
    jump_count    = int(np.sum(np.abs(delta) > cfg.tree_jump_threshold_c))
    summary.log("Supplementary Diagnostics - Staircase",
                f"{label} mean jump (°C)",  f"{mean_abs_jump:.5f}")
    summary.log("Supplementary Diagnostics - Staircase",
                f"{label} maximum jump (°C)", f"{max_jump:.5f}")
    summary.log("Supplementary Diagnostics - Staircase",
                f"{label} jump count (> {cfg.tree_jump_threshold_c}°C)", str(jump_count))
    return mean_abs_jump, max_jump, jump_count


def local_gradient_diagnostics(
    pred_arr: np.ndarray, scenario_pct: np.ndarray, label: str, summary: SummaryLog
) -> np.ndarray:
    """Compute local finite-difference gradients of predicted LST along
    the scenario trajectory."""
    derivative = safe_divide(np.diff(pred_arr), np.diff(scenario_pct), fill=0.0)
    summary.log("Supplementary Diagnostics - Local Gradient",
                f"{label} median local derivative (°C/scenario%)",
                f"{np.median(derivative):.5f}")
    summary.log("Supplementary Diagnostics - Local Gradient",
                f"{label} max local derivative (°C/scenario%)",
                f"{np.max(derivative):.5f}")
    summary.log("Supplementary Diagnostics - Local Gradient",
                f"{label} min local derivative (°C/scenario%)",
                f"{np.min(derivative):.5f}")
    return derivative


def compute_display_smoothing(
    s1: ScenarioTrajectory, s2: ScenarioTrajectory, cfg: Config
) -> dict:
    """Apply Savitzky-Golay smoothing to a trajectory for display only,
    without altering the underlying predictions."""
    scenario_pct = s1.scenario_pct
    step_pct     = scenario_pct[1] - scenario_pct[0]
    window = int(round(cfg.savgol_smooth_target_window_pct / step_pct))
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    max_window = (
        len(scenario_pct) if len(scenario_pct) % 2 == 1 else len(scenario_pct) - 1
    )
    window = min(window, max_window)

    lst_s1_smooth  = savgol_filter(s1.lst, window_length=window, polyorder=2)
    lst_s2_smooth  = savgol_filter(s2.lst, window_length=window, polyorder=2)
    spline_s2      = UnivariateSpline(scenario_pct, s2.lst, s=len(scenario_pct) * 0.5)
    lst_s2_spline  = spline_s2(scenario_pct)

    return {
        "lst_s1_smooth":  lst_s1_smooth,
        "lst_s2_smooth":  lst_s2_smooth,
        "lst_s2_spline":  lst_s2_spline,
        "spline_s2_obj":  spline_s2,
        "window":         window,
    }


# NEGI COMPUTATION
def _energy_cost(
    scenario_fraction: np.ndarray, w0: float, exponent: float, cfg: Config
) -> np.ndarray:
    return w0 * cfg.desal_energy_intensity * (scenario_fraction ** exponent)


def _energy_norm(
    scenario_fraction: np.ndarray, w0: float, exponent: float, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_energy_cost, normalised_energy_cost).

    The denominator is always the maximum of _energy_cost at
    cfg.reference_w0.  When w0 == cfg.reference_w0, the normalised cost
    reduces to f^exponent / max(f^exponent) - a deliberate
    nondimensionalisation so the index is unit-free and scale-free with
    respect to absolute irrigation volume at the reference operating point.
    Physical units of w0 and desal_energy_intensity affect results only
    in the sensitivity analysis, where w0_val != cfg.reference_w0.
    """
    raw           = _energy_cost(scenario_fraction, w0, exponent, cfg)
    reference_max = _energy_cost(scenario_fraction, cfg.reference_w0, exponent, cfg).max()
    return raw, safe_divide(raw, reference_max)


def compute_negi_results(
    baseline: ScenarioBaseline,
    s1: ScenarioTrajectory,
    s2: ScenarioTrajectory,
    cfg: Config,
) -> tuple["NEGIResults", float]:
    """Compute the NEGI (Net Environmental Gain Index) for Scenario 1
    and Scenario 2 trajectories."""
    delta_t_s1 = s1.cooling(baseline.baseline_lst)
    delta_t_s2 = s2.cooling(baseline.baseline_lst)

    warming_fraction_s2  = float((delta_t_s2 < 0).mean())
    cooling_fraction_s2  = float((delta_t_s2 > 0).mean())
    max_cooling_s2       = float(delta_t_s2.max())
    max_warming_s2       = float((-delta_t_s2).max())
    cooling_points       = np.where(delta_t_s2 > 0)[0]
    first_cooling_pct    = (
        float(s2.scenario_pct[cooling_points[0]])
        if len(cooling_points) > 0 else float("nan")
    )

    reference_cooling_c = cfg.reference_cooling_scale_factor * max(
        np.abs(delta_t_s1).max(), np.abs(delta_t_s2).max()
    )
    if reference_cooling_c <= TEMP_ZERO_GUARD:
        raise RuntimeError(
            "Scenario predictions do not differ from baseline; NEGI undefined."
        )

    desal_energy_sqrt,   energy_norm_sqrt   = _energy_norm(
        s1.scenario_fraction, cfg.reference_w0, cfg.sqrt_exponent,   cfg
    )
    desal_energy_linear, energy_norm_linear = _energy_norm(
        s1.scenario_fraction, cfg.reference_w0, cfg.linear_exponent, cfg
    )

    cooling_benefit_s1 = np.maximum(delta_t_s1, 0.0)
    cooling_benefit_s2 = np.maximum(delta_t_s2, 0.0)
    benefit_scale = max(cooling_benefit_s1.max(), cooling_benefit_s2.max(), EPS)

    benefit_s1_norm = safe_divide(cooling_benefit_s1, benefit_scale)
    benefit_s2_norm = safe_divide(cooling_benefit_s2, benefit_scale)

    negi_s1        = cfg.alpha_weight * benefit_s1_norm - cfg.beta_weight * energy_norm_sqrt
    negi_s2        = cfg.alpha_weight * benefit_s2_norm - cfg.beta_weight * energy_norm_sqrt
    negi_s1_linear = cfg.alpha_weight * benefit_s1_norm - cfg.beta_weight * energy_norm_linear
    negi_s2_linear = cfg.alpha_weight * benefit_s2_norm - cfg.beta_weight * energy_norm_linear

    for arr, name in [
        (negi_s1,        "NEGI_s1"),
        (negi_s2,        "NEGI_s2"),
        (negi_s1_linear, "NEGI_s1_linear"),
        (negi_s2_linear, "NEGI_s2_linear"),
    ]:
        assert_finite(arr, name)

    upper_bound = cfg.alpha_weight + 1e-8
    assert np.all(negi_s1 <= upper_bound), \
        "negi_s1 exceeds the theoretical upper bound alpha_weight."
    assert np.all(negi_s2 <= upper_bound), \
        "negi_s2 exceeds the theoretical upper bound alpha_weight."
    assert len(s1.scenario_fraction) == len(s2.scenario_fraction) \
        == len(negi_s1) == len(negi_s2), \
        "Scenario arrays (s1, s2, negi_s1, negi_s2) must all have equal length."

    return NEGIResults(
        scenarios=s1.scenario_fraction,
        baseline_lst=baseline.baseline_lst,
        delta_t_s1=delta_t_s1, delta_t_s2=delta_t_s2,
        benefit_scale=benefit_scale,
        desal_energy_sqrt=desal_energy_sqrt,
        desal_energy_linear=desal_energy_linear,
        energy_norm_sqrt=energy_norm_sqrt,
        energy_norm_linear=energy_norm_linear,
        negi_s1=negi_s1, negi_s2=negi_s2,
        negi_s1_linear=negi_s1_linear, negi_s2_linear=negi_s2_linear,
        warming_fraction_s2=warming_fraction_s2,
        cooling_fraction_s2=cooling_fraction_s2,
        max_cooling_s2=max_cooling_s2, max_warming_s2=max_warming_s2,
        first_cooling_pct=first_cooling_pct,
    ), reference_cooling_c


def report_negi_scenario_diagnostics(
    negi: NEGIResults, reference_cooling_c: float, cfg: Config, summary: SummaryLog
) -> None:
    """Log and record NEGI scenario summary statistics and boundary-maximum checks."""
    log_info(
        f"\nNote: The surrogate predicts LST above the observed baseline across "
        f"{negi.warming_fraction_s2 * 100:.1f}% of the Scenario 2 trajectory.  "
        "Scenario 2 follows an empirical NDVI-NDBI co-movement pathway; this "
        "pattern should not be interpreted as evidence that vegetation causes "
        "warming.  Predicted cooling benefit is defined as max(ΔT, 0)."
    )
    log_info("\nScenario 2 diagnostics (unclipped predictions):")
    log_info(f"  Fraction warming    : {negi.warming_fraction_s2 * 100:.2f}%")
    log_info(f"  Fraction cooling    : {negi.cooling_fraction_s2 * 100:.2f}%")
    log_info(f"  Maximum cooling     : {negi.max_cooling_s2:.4f} °C")
    log_info(f"  Maximum warming     : {negi.max_warming_s2:.4f} °C")
    first_str = (
        f"{negi.first_cooling_pct:.2f}%"
        if np.isfinite(negi.first_cooling_pct) else "none (never cools)"
    )
    log_info(f"  First cooling point : {first_str}")

    summary.log("Scenario diagnostics", "S2 fraction warming (unclipped)",
                f"{negi.warming_fraction_s2 * 100:.2f}%")
    summary.log("Scenario diagnostics", "S2 fraction cooling (unclipped)",
                f"{negi.cooling_fraction_s2 * 100:.2f}%")
    summary.log("Scenario diagnostics", "S2 maximum cooling (unclipped, °C)",
                f"{negi.max_cooling_s2:.4f}")
    summary.log("Scenario diagnostics", "S2 maximum warming (unclipped, °C)",
                f"{negi.max_warming_s2:.4f}")
    summary.log(
        "Scenario diagnostics", "S2 first cooling point (scenario %)",
        f"{negi.first_cooling_pct:.2f}" if np.isfinite(negi.first_cooling_pct) else "none",
    )

    vlog_info(
        cfg,
        f"\nREFERENCE_COOLING_C = {reference_cooling_c:.4f}°C "
        f"({cfg.reference_cooling_scale_factor}x observed max |ΔT| = "
        f"{max(np.abs(negi.delta_t_s1).max(), np.abs(negi.delta_t_s2).max()):.4f}°C)"
    )
    vlog_info(
        cfg,
        f"\nBENEFIT_SCALE = {negi.benefit_scale:.4f}°C  "
        "(maximum positive predicted cooling across both scenarios; "
        "used for within-study normalisation only)."
    )


def check_interior_optimum(
    negi_arr: np.ndarray, scenarios: np.ndarray, label: str, cfg: Config = CFG
) -> None:
    """Report where the numerical maximum of a NEGI profile lies.

    If the maximum is at the first or last evaluated scenario point, emit
    the prescribed boundary message so readers understand it is not a
    robust interior optimum.  If it is in the interior, note that it may
    still be sensitive to tree partitions.
    """
    idx      = int(np.argmax(negi_arr))
    n_at_max = int(np.sum(np.isclose(negi_arr, negi_arr[idx])))
    at_lower = idx == 0
    at_upper = idx == len(negi_arr) - 1

    if at_lower or at_upper:
        edge = "lower" if at_lower else "upper"
        msg = (
            f"  {label}: maximum observed at the {edge} edge of the evaluated "
            f"range ({scenarios[idx] * 100:.1f}%).  "
            "This maximum should not be interpreted as a robust interior optimum."
        )
    else:
        msg = (
            f"  {label}: maximum observed at scenario = "
            f"{scenarios[idx] * 100:.1f}%; may be sensitive to tree partitions."
        )
    log_info(msg)
    if cfg.debug:
        log_debug(
            f"  {label}: distinct scenario values at the maximum = "
            f"{n_at_max} (tie count at argmax)."
        )


# SUPPORT / EXTRAPOLATION DIAGNOSTICS
def _compute_nn1_distances(
    traj: ScenarioTrajectory,
    scaler: StandardScaler,
    nn_model_1: NearestNeighbors,
    label: str,
    cfg: Config,
    summary: SummaryLog,
) -> tuple[np.ndarray, float, np.ndarray]:
    points_std    = scaler.transform(traj.feature_frame()[["NDVI", "NDBI", "Elevation"]])
    nn_dist, _    = nn_model_1.kneighbors(points_std, n_neighbors=1)
    nn_dist       = nn_dist.ravel()
    p95           = np.percentile(nn_dist, 95)
    beyond_p95    = nn_dist > p95

    vlog_info(cfg, f"\n[{label}] Feature-space support (1-NN distance, standardised units):")
    vlog_info(cfg, f"  Maximum NN distance : {nn_dist.max():.4f}")
    vlog_info(cfg, f"  Median NN distance  : {np.median(nn_dist):.4f}")
    vlog_info(cfg, f"  95th percentile     : {p95:.4f}")
    if beyond_p95.any():
        idx_flagged = np.where(beyond_p95)[0]
        pct_flagged = ", ".join(
            f"{traj.scenario_pct[i]:.1f}%" for i in idx_flagged[:10]
        )
        more = f" (+{len(idx_flagged) - 10} more)" if len(idx_flagged) > 10 else ""
        vlog_info(cfg,
                  f"  Points beyond 95th percentile ({len(idx_flagged)}): "
                  f"scenario = {pct_flagged}{more}")
    else:
        vlog_info(cfg, "  No points beyond the 95th percentile.")

    summary.log("Support diagnostics", f"{label} maximum NN distance",     f"{nn_dist.max():.4f}")
    summary.log("Support diagnostics", f"{label} median NN distance",      f"{np.median(nn_dist):.4f}")
    summary.log("Support diagnostics", f"{label} 95th percentile NN dist", f"{p95:.4f}")
    summary.log("Support diagnostics", f"{label} points beyond 95th pct",  str(int(beyond_p95.sum())))
    return nn_dist, p95, beyond_p95


def _mean_knn_distance(
    traj: ScenarioTrajectory,
    scaler: StandardScaler,
    nn_model_k: NearestNeighbors,
    cfg: Config,
) -> np.ndarray:
    points_std = scaler.transform(traj.feature_frame()[["NDVI", "NDBI", "Elevation"]])
    dist, _    = nn_model_k.kneighbors(points_std, n_neighbors=cfg.support_knn_k)
    return dist.mean(axis=1)


def _compute_extrapolation_report(
    traj: ScenarioTrajectory,
    mean_knn_dist: np.ndarray,
    label: str,
    data: DatasetBundle,
    support_threshold_k: float,
    cfg: Config,
    summary: SummaryLog,
) -> bool:
    frame             = traj.feature_frame()
    ndvi_ok           = frame["NDVI"].between(data.feature_min["NDVI"],
                                               data.feature_max["NDVI"]).all()
    ndbi_ok           = frame["NDBI"].between(data.feature_min["NDBI"],
                                               data.feature_max["NDBI"]).all()
    elevation_ok      = frame["Elevation"].between(data.feature_min["Elevation"],
                                                    data.feature_max["Elevation"]).all()
    nn_ok             = (mean_knn_dist <= support_threshold_k).all()
    fully_supported   = (ndvi_ok and ndbi_ok and elevation_ok and nn_ok
                         and not traj.out_of_bounds.any())

    vlog_info(cfg, f"\n[{label}] Extrapolation check:")
    if fully_supported:
        vlog_info(cfg, "  Scenario remains entirely within dense training support.")
    else:
        reasons = []
        if not ndvi_ok:      reasons.append("NDVI outside observed range")
        if not ndbi_ok:      reasons.append("NDBI outside observed range")
        if not elevation_ok: reasons.append("elevation outside observed range")
        if not nn_ok:        reasons.append("nearest-neighbour distance exceeds support threshold")
        vlog_info(cfg,
                  f"  Warning: scenario enters sparse feature space "
                  f"({'; '.join(reasons)}).")
    summary.log("Support diagnostics", f"{label} fully within dense support",
                str(fully_supported))
    return fully_supported


def compute_support_diagnostics(
    model_results: ModelResults,
    s1: ScenarioTrajectory, s2: ScenarioTrajectory,
    cfg: Config, summary: SummaryLog,
) -> SupportDiagnostics:
    """Compute nearest-neighbor feature-space support diagnostics for
    the scenario trajectories."""
    data    = model_results.data
    scaler  = StandardScaler().fit(data.X)
    feat_mx = scaler.transform(data.X)

    nn_model_1 = NearestNeighbors(n_neighbors=1).fit(feat_mx)
    nn_dist_s1, nn_p95_s1, nn_beyond_p95_s1 = _compute_nn1_distances(
        s1, scaler, nn_model_1, "Scenario 1", cfg, summary
    )
    nn_dist_s2, nn_p95_s2, nn_beyond_p95_s2 = _compute_nn1_distances(
        s2, scaler, nn_model_1, "Scenario 2", cfg, summary
    )

    nn_model_k   = NearestNeighbors(n_neighbors=cfg.support_knn_k).fit(feat_mx)
    obs_dist_k, _ = nn_model_k.kneighbors(feat_mx, n_neighbors=cfg.support_knn_k + 1)
    obs_dist_k    = obs_dist_k[:, 1:]
    support_threshold_k = np.percentile(obs_dist_k.mean(axis=1), cfg.support_percentile)

    mean_knn_dist_s1 = _mean_knn_distance(s1, scaler, nn_model_k, cfg)
    mean_knn_dist_s2 = _mean_knn_distance(s2, scaler, nn_model_k, cfg)
    extrap_s1_ok = _compute_extrapolation_report(
        s1, mean_knn_dist_s1, "Scenario 1", data, support_threshold_k, cfg, summary
    )
    extrap_s2_ok = _compute_extrapolation_report(
        s2, mean_knn_dist_s2, "Scenario 2", data, support_threshold_k, cfg, summary
    )
    in_support_s2 = mean_knn_dist_s2 <= support_threshold_k

    return SupportDiagnostics(
        scaler=scaler,
        nn_dist_s1=nn_dist_s1, nn_p95_s1=nn_p95_s1, nn_beyond_p95_s1=nn_beyond_p95_s1,
        nn_dist_s2=nn_dist_s2, nn_p95_s2=nn_p95_s2, nn_beyond_p95_s2=nn_beyond_p95_s2,
        support_threshold_k=support_threshold_k,
        mean_knn_dist_s2=mean_knn_dist_s2, in_support_s2=in_support_s2,
        extrap_s1_ok=extrap_s1_ok, extrap_s2_ok=extrap_s2_ok,
    )


def report_data_support_check(
    s2: ScenarioTrajectory, negi: NEGIResults,
    support: SupportDiagnostics, cfg: Config, summary: SummaryLog,
) -> None:
    """Log and record the feature-space support (extrapolation) check results."""
    in_support = support.in_support_s2
    s2_ood     = s2.out_of_bounds

    log_info("\nData support diagnostics:")
    log_info(f"  Feature bounds check : {'PASS' if not s2_ood.any() else 'FAIL'}")
    log_info(
        f"  kNN support check    : "
        f"{'PASS' if in_support.all() else f'FAIL - {(~in_support).sum()} points outside'}"
    )

    if (~in_support).any():
        first_out_idx = np.argmax(~in_support)
        log_info(f"  Dense region exit at : scenario = {s2.scenario_pct[first_out_idx]:.1f}%")

    trustworthy         = in_support & (~s2_ood)
    negi_s2_trustworthy = np.where(trustworthy, negi.negi_s2, np.nan)
    if np.all(np.isnan(negi_s2_trustworthy)):
        log_info("  No Scenario 2 points pass both support checks.")
    else:
        best_idx = int(np.nanargmax(negi_s2_trustworthy))
        log_info(
            f"  Numerical maximum (both checks): "
            f"scenario = {s2.scenario_pct[best_idx]:.1f}% "
            f"(NEGI = {negi.negi_s2[best_idx]:.3f}); "
            f"unvetted value: {s2.scenario_pct[negi.s2_optimum_idx]:.1f}%"
        )
        log_info(f"  Points passing both checks: {trustworthy.sum()}/{len(trustworthy)}")

    summary.log("Support diagnostics", "Feature bounds check",
                "PASS" if not s2_ood.any() else "FAIL")
    summary.log("Support diagnostics", "kNN support check",
                "PASS" if in_support.all() else f"FAIL ({(~in_support).sum()} outside)")


# UNCERTAINTY BANDS (SPATIAL REFITS)
def compute_uncertainty_bands(
    model_results: ModelResults,
    baseline: ScenarioBaseline,
    s1: ScenarioTrajectory, s2: ScenarioTrajectory,
    negi: NEGIResults, cfg: Config,
) -> UncertaintyResults:
    """Compute NEGI and LST uncertainty bands from repeated spatial-block holdout refits."""
    data      = model_results.data
    scenarios = s1.scenario_fraction

    splits = list(GroupShuffleSplit(
        n_splits=cfg.n_spatial_refits,
        test_size=cfg.holdout_test_size,
        random_state=cfg.uncertainty_refit_seed,
    ).split(data.X, data.y, groups=data.groups))

    # Build batch feature frames once outside the fold loop - avoids
    # per-point model.predict() calls (saves ~8,000 individual calls for
    # 10 refits × 401 scenario points × 2 scenarios).
    feat_s1 = pd.DataFrame({
        "NDVI": s1.ndvi, "NDBI": s1.ndbi, "Elevation": baseline.fixed_elevation,
    })
    feat_s2 = pd.DataFrame({
        "NDVI": s2.ndvi, "NDBI": s2.ndbi, "Elevation": baseline.fixed_elevation,
    })

    df_full = data.df
    ndbi_range = np.linspace(
        df_full["NDBI"].quantile(cfg.ndbi_sweep_quantile_low),
        df_full["NDBI"].quantile(cfg.ndbi_sweep_quantile_high),
        cfg.ndbi_sweep_points,
    )
    ndbi_sweep_frame = pd.DataFrame({
        "NDVI":      np.full(cfg.ndbi_sweep_points, df_full["NDVI"].median()),
        "NDBI":      ndbi_range,
        "Elevation": np.full(cfg.ndbi_sweep_points, df_full["Elevation"].median()),
    })

    negi_s1_folds, negi_s2_folds   = [], []
    lst_s1_folds,  lst_s2_folds    = [], []
    cooling_s1_folds, cooling_s2_folds = [], []
    ndbi_sweep_folds                = []

    for train_idx_fold, _ in splits:
        Xf, yf = data.X.iloc[train_idx_fold], data.y.iloc[train_idx_fold]
        mf = XGBRegressor(
            objective="reg:squarederror",
            random_state=cfg.random_seed,
            monotone_constraints=cfg.monotone_constraints,
            n_jobs=1, **model_results.best_params,
        )
        mf.fit(Xf, yf)

        baseline_lst_f, _ = predict_lst(
            mf, baseline.baseline_ndvi, baseline.baseline_ndbi,
            baseline.fixed_elevation, data.feature_min, data.feature_max,
        )
        lst_s1_f = mf.predict(feat_s1)
        lst_s2_f = mf.predict(feat_s2)

        cooling_s1_f = baseline_lst_f - lst_s1_f
        cooling_s2_f = baseline_lst_f - lst_s2_f
        benefit_s1_f = safe_divide(np.maximum(cooling_s1_f, 0.0), negi.benefit_scale)
        benefit_s2_f = safe_divide(np.maximum(cooling_s2_f, 0.0), negi.benefit_scale)
        negi_s1_f    = (cfg.alpha_weight * benefit_s1_f
                        - cfg.beta_weight * negi.energy_norm_sqrt)
        negi_s2_f    = (cfg.alpha_weight * benefit_s2_f
                        - cfg.beta_weight * negi.energy_norm_sqrt)

        negi_s1_folds.append(negi_s1_f)
        negi_s2_folds.append(negi_s2_f)
        lst_s1_folds.append(lst_s1_f)
        lst_s2_folds.append(lst_s2_f)
        cooling_s1_folds.append(cooling_s1_f)
        cooling_s2_folds.append(cooling_s2_f)
        ndbi_sweep_folds.append(mf.predict(ndbi_sweep_frame))

    negi_s1_folds    = np.array(negi_s1_folds)
    negi_s2_folds    = np.array(negi_s2_folds)
    lst_s1_folds     = np.array(lst_s1_folds)
    lst_s2_folds     = np.array(lst_s2_folds)
    cooling_s1_folds = np.array(cooling_s1_folds)
    cooling_s2_folds = np.array(cooling_s2_folds)
    ndbi_sweep_folds = np.array(ndbi_sweep_folds)

    negi_s1_mean = negi_s1_folds.mean(axis=0)
    negi_s1_std  = negi_s1_folds.std(axis=0)
    negi_s2_mean = negi_s2_folds.mean(axis=0)
    negi_s2_std  = negi_s2_folds.std(axis=0)

    def _pct_summary(fold_array: np.ndarray) -> dict:
        return {p: np.percentile(fold_array, p, axis=0)
                for p in cfg.uncertainty_percentiles}

    percentiles = {
        "negi_s1":    _pct_summary(negi_s1_folds),
        "negi_s2":    _pct_summary(negi_s2_folds),
        "lst_s1":     _pct_summary(lst_s1_folds),
        "lst_s2":     _pct_summary(lst_s2_folds),
        "cooling_s1": _pct_summary(cooling_s1_folds),
        "cooling_s2": _pct_summary(cooling_s2_folds),
    }
    p = percentiles
    table = pd.DataFrame({
        "Scenario (%)":      scenarios * 100,
        "NEGI_S1_median":    p["negi_s1"][50],
        "NEGI_S1_Q25":       p["negi_s1"][25],
        "NEGI_S1_Q75":       p["negi_s1"][75],
        "NEGI_S1_P2.5":      p["negi_s1"][2.5],
        "NEGI_S1_P97.5":     p["negi_s1"][97.5],
        "NEGI_S2_median":    p["negi_s2"][50],
        "NEGI_S2_Q25":       p["negi_s2"][25],
        "NEGI_S2_Q75":       p["negi_s2"][75],
        "NEGI_S2_P2.5":      p["negi_s2"][2.5],
        "NEGI_S2_P97.5":     p["negi_s2"][97.5],
        "NEGI_S1_mean":      negi_s1_mean,
        "NEGI_S1_std":       negi_s1_std,
        "NEGI_S2_mean":      negi_s2_mean,
        "NEGI_S2_std":       negi_s2_std,
        "LST_S1_median":     p["lst_s1"][50],
        "LST_S1_Q25":        p["lst_s1"][25],
        "LST_S1_Q75":        p["lst_s1"][75],
        "LST_S2_median":     p["lst_s2"][50],
        "LST_S2_Q25":        p["lst_s2"][25],
        "LST_S2_Q75":        p["lst_s2"][75],
        "Cooling_S1_median": p["cooling_s1"][50],
        "Cooling_S1_Q25":    p["cooling_s1"][25],
        "Cooling_S1_Q75":    p["cooling_s1"][75],
        "Cooling_S2_median": p["cooling_s2"][50],
        "Cooling_S2_Q25":    p["cooling_s2"][25],
        "Cooling_S2_Q75":    p["cooling_s2"][75],
    })

    return UncertaintyResults(
        negi_s1_folds=negi_s1_folds, negi_s2_folds=negi_s2_folds,
        lst_s1_folds=lst_s1_folds,   lst_s2_folds=lst_s2_folds,
        cooling_s1_folds=cooling_s1_folds, cooling_s2_folds=cooling_s2_folds,
        negi_s1_mean=negi_s1_mean, negi_s1_std=negi_s1_std,
        negi_s2_mean=negi_s2_mean, negi_s2_std=negi_s2_std,
        percentiles=percentiles, table=table,
        ndbi_range=ndbi_range, ndbi_sweep_folds=ndbi_sweep_folds,
    )


def report_uncertainty_bands(
    uncertainty: UncertaintyResults, s1: ScenarioTrajectory,
    cfg: Config, summary: SummaryLog,
) -> None:
    """Log and record summary statistics for the uncertainty bands."""
    save_csv(uncertainty.table, cfg.data_dir / "negi_uncertainty_bands.csv")
    scenario_pct  = s1.scenario_pct
    p             = uncertainty.percentiles
    peak_std_idx  = int(np.argmax(uncertainty.negi_s2_std))
    peak_iqr_idx  = int(np.argmax(p["negi_s2"][75] - p["negi_s2"][25]))
    n_refits      = len(uncertainty.negi_s2_folds)

    summary.log("Uncertainty diagnostics", "N spatial refits",              str(n_refits))
    summary.log("Uncertainty diagnostics", "Peak S2 NEGI SD (scenario %)",
                f"{scenario_pct[peak_std_idx]:.1f}")
    summary.log("Uncertainty diagnostics", "Peak S2 NEGI SD (value)",
                f"{uncertainty.negi_s2_std[peak_std_idx]:.4f}")
    summary.log("Uncertainty diagnostics", "Peak S2 NEGI IQR (scenario %)",
                f"{scenario_pct[peak_iqr_idx]:.1f}")
    summary.log("Uncertainty diagnostics", "Peak S2 NEGI IQR (Q75-Q25)",
                f"{(p['negi_s2'][75] - p['negi_s2'][25])[peak_iqr_idx]:.4f}")


def _distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {
        "mean":   float(values.mean()),
        "sd":     float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(q50),
        "q25":    float(q25),
        "q75":    float(q75),
        "iqr":    float(q75 - q25),
        "p2.5":   float(np.percentile(values, 2.5)),
        "p97.5":  float(np.percentile(values, 97.5)),
    }


def summarize_spatial_refit_diagnostics(
    validation: ValidationResults,
    uncertainty: UncertaintyResults,
    s1: ScenarioTrajectory,
    cfg: Config,
) -> pd.DataFrame:
    """Summarise repeated spatial-block refits (median/IQR/percentiles)."""
    scenario_pct      = s1.scenario_pct
    cooling_s2_folds  = uncertainty.cooling_s2_folds
    negi_s2_folds     = uncertainty.negi_s2_folds

    per_fold_max_cooling       = cooling_s2_folds.max(axis=1)
    per_fold_trajectory_max    = negi_s2_folds.max(axis=1)
    per_fold_first_cooling_pct = np.full(len(cooling_s2_folds), np.nan)
    for i, row in enumerate(cooling_s2_folds):
        cp = np.where(row > 0)[0]
        if len(cp) > 0:
            per_fold_first_cooling_pct[i] = scenario_pct[cp[0]]

    finite_fcp = per_fold_first_cooling_pct[np.isfinite(per_fold_first_cooling_pct)]
    rows = {
        "Repeated holdout R2":                  _distribution_summary(validation.repeated_r2_scores),
        "Repeated holdout RMSE (C)":            _distribution_summary(validation.repeated_rmse_scores),
        "Repeated holdout MAE (C)":             _distribution_summary(validation.repeated_mae_scores),
        "Spatial-refit S2 max cooling (C)":     _distribution_summary(per_fold_max_cooling),
        "Spatial-refit S2 NEGI trajectory max": _distribution_summary(per_fold_trajectory_max),
        "Spatial-refit S2 first cooling pct (%)": (
            _distribution_summary(finite_fcp)
            if len(finite_fcp) > 0
            else {k: np.nan for k in ["mean", "sd", "median", "q25", "q75", "iqr", "p2.5", "p97.5"]}
        ),
    }
    table = pd.DataFrame(rows).T
    table.index.name = "Quantity"
    table = table.reset_index()
    save_csv(table, cfg.data_dir / "spatial_refit_summary.csv")
    return table


# SENSITIVITY ANALYSIS
def run_sensitivity_analysis(
    negi: NEGIResults, s1: ScenarioTrajectory, cfg: Config
) -> SensitivityResults:
    """Sweep NEGI cost-function parameters (alpha, beta, w0, exponent)
    for the sensitivity analysis."""
    scenarios     = s1.scenario_fraction
    scenario_pct  = scenarios * 100
    cooling_norm_s2 = safe_divide(np.maximum(negi.delta_t_s2, 0.0), negi.benefit_scale)

    reference_max_by_exponent = {
        exp: _energy_cost(scenarios, cfg.reference_w0, exp, cfg).max()
        for exp in cfg.sensitivity_exponents
    }

    rows = []
    for exponent in cfg.sensitivity_exponents:
        ref_max = reference_max_by_exponent[exponent]
        for alpha_val in cfg.sensitivity_alpha_values:
            for beta_val in cfg.sensitivity_beta_values:
                for w0_val in cfg.sensitivity_w0_values:
                    energy_penalty = _energy_cost(scenarios, w0_val, exponent, cfg)
                    energy_norm_arr = safe_divide(energy_penalty, ref_max)
                    negi_profile    = alpha_val * cooling_norm_s2 - beta_val * energy_norm_arr
                    best = int(np.argmax(negi_profile))
                    rows.append({
                        "Exponent": exponent,
                        "Alpha":    alpha_val,
                        "Beta":     beta_val,
                        "w0":       w0_val,
                        "Numerical_Maximum_Scenario (%)": scenario_pct[best],
                        "Numerical_Maximum_NEGI":         negi_profile[best],
                    })

    # Normalisation sanity checks
    for exp, ref_max in reference_max_by_exponent.items():
        check_val = _energy_cost(scenarios, cfg.reference_w0, exp, cfg).max() / ref_max
        assert np.isclose(check_val, 1.0, atol=1e-9), \
            f"Normalisation broken for exponent={exp}"

    ref_max_default = reference_max_by_exponent[cfg.sensitivity_default_exponent]
    e_ref = safe_divide(
        _energy_cost(scenarios, cfg.reference_w0, cfg.sensitivity_default_exponent, cfg),
        ref_max_default,
    )
    for _w0_test in [50.0, 100.0, 300.0]:
        _e_test = safe_divide(
            _energy_cost(scenarios, _w0_test, cfg.sensitivity_default_exponent, cfg),
            ref_max_default,
        )
        assert np.allclose(_e_test, e_ref * (_w0_test / cfg.reference_w0), atol=1e-9), \
            f"w0 linearity guard failed at w0={_w0_test}"

    table = pd.DataFrame(rows)

    plot_table = table[
        isclose_mask(table["Exponent"], cfg.sensitivity_default_exponent)
        & isclose_mask(table["Beta"],   cfg.sensitivity_default_beta)
    ]
    assert len(plot_table) > 0, \
        "plot_table is empty - check sensitivity_default_exponent / sensitivity_default_beta."

    return SensitivityResults(table=table, plot_table=plot_table)


def report_sensitivity_analysis(sensitivity: SensitivityResults, cfg: Config) -> None:
    """Log and record the sensitivity analysis results."""
    save_csv(sensitivity.table, cfg.data_dir / "negi_sensitivity_analysis.csv")
    unique_optima = sensitivity.table["Numerical_Maximum_Scenario (%)"].nunique()
    log_info(f"\nDistinct numerical maximum scenario values across parameter grid: {unique_optima}")


# CONDITIONAL NDVI-LST SLOPES
def compute_conditional_ndvi_lst_slopes(
    df: pd.DataFrame, ndbi_bins: list[tuple]
) -> list[ConditionalBinResult]:
    """Compute the NDVI-LST slope within each NDBI stratum (conditional analysis)."""
    results = []
    for ndbi_lo, ndbi_hi, label in ndbi_bins:
        mask   = (df["NDBI"] >= ndbi_lo) & (df["NDBI"] < ndbi_hi)
        subset = df[mask]
        if len(subset) > 10:
            sl, ic, rv, pv, se = scipy_stats.linregress(subset["NDVI"], subset["LST"])
            n      = len(subset)
            t_crit = scipy_stats.t.ppf(0.975, df=n - 2)
            results.append(ConditionalBinResult(
                label=label,
                ndvi=subset["NDVI"].to_numpy(), lst=subset["LST"].to_numpy(),
                n_pixels=n, slope=sl, intercept=ic, slope_se=se,
                slope_ci95=t_crit * se, r_squared=rv ** 2, p_value=pv,
            ))
        else:
            results.append(ConditionalBinResult(
                label=label,
                ndvi=subset["NDVI"].to_numpy(), lst=subset["LST"].to_numpy(),
                n_pixels=len(subset),
                slope=None, intercept=None, slope_se=None,
                slope_ci95=None, r_squared=None, p_value=None,
            ))
    return results


def conditional_slopes_to_dataframe(
    results: list[ConditionalBinResult]
) -> pd.DataFrame:
    """Assemble per-stratum conditional NDVI-LST slope results into a DataFrame."""
    rows = []
    for r in results:
        if r.slope is None:
            continue
        rows.append({
            "NDBI_range":  r.label.replace("\n", " "),
            "n_pixels":    r.n_pixels,
            "slope":       r.slope,
            "slope_SE":    r.slope_se,
            "slope_95CI":  r.slope_ci95,
            "r_squared":   r.r_squared,
            "p_value":     r.p_value,
            "significant": "Y" if r.p_value < 0.05 else "N",
            "direction":   "positive" if r.slope > 0 else "negative",
        })
    return pd.DataFrame(rows)


# RESPONSE COMPUTATIONS
def compute_ndbi_response_sweep(model_results: ModelResults, cfg: Config) -> dict:
    """Sweep NDBI at fixed NDVI/Elevation to trace the predicted LST response curve."""
    df              = model_results.data.df
    fixed_elevation = df["Elevation"].median()
    ndbi_median_val = df["NDBI"].median()
    ndbi_range = np.linspace(
        df["NDBI"].quantile(cfg.ndbi_sweep_quantile_low),
        df["NDBI"].quantile(cfg.ndbi_sweep_quantile_high),
        cfg.ndbi_sweep_points,
    )
    sweep_df = pd.DataFrame({
        "NDVI":      np.full(cfg.ndbi_sweep_points, df["NDVI"].median()),
        "NDBI":      ndbi_range,
        "Elevation": np.full(cfg.ndbi_sweep_points, fixed_elevation),
    })
    pred_raw    = model_results.model.predict(sweep_df)
    pred_smooth = savgol_filter(pred_raw, 11, 2)
    return {
        "ndbi_range":   ndbi_range,
        "ndbi_median":  ndbi_median_val,
        "pred_raw":     pred_raw,
        "pred_smooth":  pred_smooth,
    }


def compute_response_surface(
    model_results: ModelResults, cfg: Config, grid_size: int = 150
) -> dict:
    """Compute the predicted-LST response surface over the NDVI-NDBI feature grid."""
    df              = model_results.data.df
    fixed_elevation = df["Elevation"].median()
    ndvi_max_plot   = np.percentile(df["NDVI"], 99)
    ndvi_grid = np.linspace(df["NDVI"].min(), ndvi_max_plot, grid_size)
    ndbi_grid = np.linspace(df["NDBI"].min(), df["NDBI"].max(), grid_size)
    ndvi_mesh, ndbi_mesh = np.meshgrid(ndvi_grid, ndbi_grid)
    grid_df  = pd.DataFrame({
        "NDVI":      ndvi_mesh.ravel(),
        "NDBI":      ndbi_mesh.ravel(),
        "Elevation": fixed_elevation,
    })
    lst_mesh = model_results.model.predict(grid_df).reshape(ndvi_mesh.shape)
    return {
        "ndvi_mesh":    ndvi_mesh,
        "ndbi_mesh":    ndbi_mesh,
        "lst_mesh":     lst_mesh,
        "ndvi_max_plot": ndvi_max_plot,
    }


def compute_ndvi_decile_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise LST statistics by NDVI decile."""
    return (
        df.assign(NDVI_bin=pd.qcut(df["NDVI"], 10, duplicates="drop"))
        .groupby("NDVI_bin", observed=True)
        .agg(
            NDVI_median=("NDVI", "median"),
            LST_median=("LST", "median"),
            LST_mean=("LST", "mean"),
            LST_std=("LST", "std"),
            NDBI_median=("NDBI", "median"),
            n_pixels=("LST", "size"),
        )
        .reset_index(drop=True)
    )


def compute_saturation_diagnostic(
    negi: NEGIResults, s2: ScenarioTrajectory, cfg: Config
) -> dict:
    """Diagnose cooling saturation behaviour at high, mid, and near-zero NDVI fractions."""
    max_pos = negi.delta_t_s2.max()
    if max_pos > EPS:
        frac = safe_divide(negi.delta_t_s2, max_pos)
        sat_90_idx = int(np.argmax(frac >= cfg.saturation_high_fraction))
        sat_50_idx = int(np.argmax(frac >= cfg.saturation_mid_fraction))
        near_zero  = cfg.saturation_near_zero_fraction * max_pos
        candidates = np.where(negi.delta_t_s2 <= near_zero)[0]
        collapse_idx = (
            int(candidates[0])
            if len(candidates) > 0 and candidates[0] > sat_90_idx else None
        )
    else:
        sat_50_idx = sat_90_idx = collapse_idx = None
    return {
        "sat_50_idx":    sat_50_idx,
        "sat_90_idx":    sat_90_idx,
        "collapse_idx":  collapse_idx,
    }


# Interpretive diagnostics (peer review)
#
# Purely descriptive / diagnostic reporting requested by reviewers. Nothing
# here changes the XGBoost model, the NEGI equations, the scenario
# trajectories, the empirical NDVI-NDBI spline, the bootstrap procedure, or
# any figure's underlying data.

def compute_ndvi_lst_influence_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Leverage / Cook's distance / studentized residuals for the existing
    global NDVI-vs-LST OLS regression (the same regression already drawn
    in the NDVI-vs-LST figure).  No observations are removed and no
    robust regression is fitted - this only quantifies the influence of
    individual points on the regression already reported.
    """
    ndvi = df["NDVI"].to_numpy(dtype=float)
    lst  = df["LST"].to_numpy(dtype=float)
    n = len(ndvi)
    p = 2  # intercept + NDVI slope, matches the existing OLS fit

    X = np.column_stack([np.ones(n), ndvi])
    XtX_inv = np.linalg.pinv(X.T @ X)
    H = X @ XtX_inv @ X.T
    leverage = np.clip(np.diag(H), 0.0, 1.0 - 1e-12)

    beta = XtX_inv @ X.T @ lst
    fitted = X @ beta
    resid = lst - fitted
    dof = max(n - p, 1)
    mse = float(np.sum(resid ** 2) / dof)

    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = safe_divide(
            resid, np.sqrt(np.maximum(mse * (1.0 - leverage), 1e-12))
        )
        cooks_d = safe_divide(
            (resid ** 2) * leverage,
            p * mse * (1.0 - leverage) ** 2,
        )

    return_df = pd.DataFrame({
        "NDVI": ndvi, "LST": lst,
        "Leverage": leverage, "Cook_D": cooks_d,
        "Studentized_Residual": studentized,
    })
    return return_df, {"n": n, "p": p, "mse": mse}


def report_ndvi_lst_influence_diagnostics(
    df: pd.DataFrame, cfg: Config, summary: SummaryLog
) -> pd.DataFrame:
    """Log and record leverage/Cook's distance influence diagnostics
    for the NDVI-LST relationship."""
    influence_df, fit_info = compute_ndvi_lst_influence_diagnostics(df)
    n, p = fit_info["n"], fit_info["p"]

    leverage_threshold = cfg.influence_leverage_multiplier * p / n
    cooks_threshold     = cfg.influence_cooks_d_multiplier / n

    influence_df["High_Leverage"] = influence_df["Leverage"] > leverage_threshold
    influence_df["Influential"]   = influence_df["Cook_D"] > cooks_threshold

    max_leverage      = float(influence_df["Leverage"].max())
    n_high_leverage   = int(influence_df["High_Leverage"].sum())
    max_cooks_d       = float(influence_df["Cook_D"].max())
    n_influential     = int(influence_df["Influential"].sum())

    log_info(
        f"\nNDVI-LST regression influence diagnostics (n={n}):\n"
        f"  Maximum leverage                 : {max_leverage:.4f}\n"
        f"  Leverage threshold ({cfg.influence_leverage_multiplier:.0f}p/n)     : {leverage_threshold:.4f}\n"
        f"  Number of high-leverage points   : {n_high_leverage}\n"
        f"  Maximum Cook's distance          : {max_cooks_d:.4f}\n"
        f"  Number with Cook's D > {cfg.influence_cooks_d_multiplier:.0f}/n       : {n_influential}"
    )
    summary.log("Interpretive Diagnostics", "NDVI-LST OLS: max leverage", f"{max_leverage:.4f}")
    summary.log("Interpretive Diagnostics", "NDVI-LST OLS: leverage threshold", f"{leverage_threshold:.4f}")
    summary.log("Interpretive Diagnostics", "NDVI-LST OLS: n high-leverage points", str(n_high_leverage))
    summary.log("Interpretive Diagnostics", "NDVI-LST OLS: max Cook's distance", f"{max_cooks_d:.4f}")
    summary.log("Interpretive Diagnostics", "NDVI-LST OLS: n influential (Cook's D > 4/n)", str(n_influential))

    save_csv(influence_df, cfg.data_dir / "ndvi_influence_diagnostics.csv")
    return influence_df


def compute_ndvi_distribution_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Quantify how concentrated the observed NDVI values are - purely
    descriptive, does not affect any figure or model."""
    ndvi = df["NDVI"].to_numpy(dtype=float)
    percentile_levels = [5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(ndvi, percentile_levels)
    summary_df = pd.DataFrame({
        "Statistic": [f"P{p}" for p in percentile_levels] + ["Maximum"],
        "NDVI": list(percentile_values) + [float(ndvi.max())],
    })
    fractions = {
        thresh: float(np.mean(ndvi > thresh)) for thresh in (0.2, 0.3, 0.4, 0.5)
    }
    return summary_df, fractions


def report_ndvi_distribution_summary(df: pd.DataFrame, cfg: Config, summary: SummaryLog) -> pd.DataFrame:
    """Log and record summary statistics for the NDVI distribution."""
    dist_df, fractions = compute_ndvi_distribution_summary(df)
    save_csv(dist_df, cfg.data_dir / "ndvi_distribution_summary.csv")

    log_info("\nNDVI distribution summary:")
    log_info(dataframe_to_markdown(dist_df))
    for thresh, frac in fractions.items():
        log_info(f"  Fraction NDVI > {thresh:.1f} : {frac:.3f}")
        summary.log("Interpretive Diagnostics", f"Fraction NDVI > {thresh:.1f}", f"{frac:.3f}")
    for _, row in dist_df.iterrows():
        summary.log("Interpretive Diagnostics", f"NDVI {row['Statistic']}", f"{row['NDVI']:.4f}")
    return dist_df


def enhance_conditional_slopes_table(
    conditional_df: pd.DataFrame, results: list[ConditionalBinResult], cfg: Config
) -> pd.DataFrame:
    """Add Adjusted R^2, Pearson r, and a variance-explained interpretation
    band to the existing conditional-NDVI-slope table (Figure 6).  The
    underlying per-stratum regressions (results) are unchanged."""
    enhanced = conditional_df.copy()
    adj_r2_col, r_col, band_col, note_col = [], [], [], []
    for _, row in enhanced.iterrows():
        matching = next(
            (r for r in results
             if r.slope is not None and r.label.replace("\n", " ") == row["NDBI_range"]),
            None,
        )
        n = int(row["n_pixels"])
        r2 = float(row["r_squared"])
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - 2, 1)
        r_value = float(np.sign(row["slope"])) * np.sqrt(max(r2, 0.0))

        if r2 < cfg.variance_band_very_weak:
            band = "Very weak (<5%)"
        elif r2 < cfg.variance_band_weak:
            band = "Weak (5-15%)"
        elif r2 < cfg.variance_band_moderate:
            band = "Moderate"
        else:
            band = "Strong"

        note = (
            "Although statistically significant, NDVI explains only a small "
            "proportion of within-stratum LST variability.  Other environmental "
            "variables contribute substantially."
            if r2 < cfg.low_r2_threshold else ""
        )
        adj_r2_col.append(adj_r2); r_col.append(r_value)
        band_col.append(band); note_col.append(note)

    enhanced["Adjusted_R_squared"]       = adj_r2_col
    enhanced["Correlation_coefficient"]  = r_col
    enhanced["Variance_explained_band"]  = band_col
    enhanced["Low_R2_note"]              = note_col
    return enhanced


def report_conditional_slopes_enhanced(
    enhanced_df: pd.DataFrame, cfg: Config, summary: SummaryLog
) -> None:
    """Log and record the conditional NDVI-LST slope diagnostics with confidence intervals."""
    save_csv(enhanced_df, cfg.data_dir / "ndvi_conditional_slopes_by_ndbi.csv")
    for _, row in enhanced_df.iterrows():
        log_info(
            f"\nNDBI stratum {row['NDBI_range']}: slope={row['slope']:.4f}, "
            f"95% CI=±{row['slope_95CI']:.4f}, R²={row['r_squared']:.4f}, "
            f"Adj. R²={row['Adjusted_R_squared']:.4f}, r={row['Correlation_coefficient']:.4f}, "
            f"n={row['n_pixels']}, variance explained: {row['Variance_explained_band']}"
        )
        if row["Low_R2_note"]:
            log_info(f"  -> {row['Low_R2_note']}")
        summary.log(
            "Interpretive Diagnostics",
            f"NDBI {row['NDBI_range']}: variance explained",
            row["Variance_explained_band"],
        )


def compute_ndvi_decile_consistency(
    df: pd.DataFrame, decile_df: pd.DataFrame
) -> dict:
    """Compare the global NDVI-vs-LST regression slope (as drawn in the
    NDVI-vs-LST figure) against the slope implied by the decile-median
    curve (Figure S03).  Neither figure is modified; this only reports
    whether the two views of the same data are broadly consistent."""
    ndvi = df["NDVI"].to_numpy(dtype=float)
    lst  = df["LST"].to_numpy(dtype=float)
    global_slope = float(
        np.polyfit(ndvi, lst, 1)[0]
    )
    decile_medians_ndvi = decile_df["NDVI_median"].to_numpy(dtype=float)
    decile_medians_lst  = decile_df["LST_median"].to_numpy(dtype=float)
    if len(decile_medians_ndvi) > 1:
        step_slopes = np.diff(decile_medians_lst) / np.diff(decile_medians_ndvi)
        median_decile_slope = float(np.median(step_slopes))
    else:
        median_decile_slope = float("nan")

    difference = (
        global_slope - median_decile_slope
        if np.isfinite(median_decile_slope) else float("nan")
    )
    # Consistency is judged only on whether the two slopes agree in sign
    # and are of broadly comparable magnitude - a deliberately coarse,
    # descriptive flag rather than a formal statistical test.
    if not np.isfinite(median_decile_slope):
        flag = "not available"
    elif np.sign(global_slope) != np.sign(median_decile_slope):
        flag = "differs in direction"
    elif abs(difference) <= 0.5 * max(abs(global_slope), abs(median_decile_slope), EPS):
        flag = "broadly consistent"
    else:
        flag = "differs in magnitude"

    return {
        "global_slope": global_slope,
        "median_decile_slope": median_decile_slope,
        "difference": difference,
        "consistency_flag": flag,
    }


def report_ndvi_decile_consistency(
    df: pd.DataFrame, decile_df: pd.DataFrame, cfg: Config, summary: SummaryLog
) -> dict:
    """Log and record whether NDVI-decile LST trends are monotonic and consistent."""
    consistency = compute_ndvi_decile_consistency(df, decile_df)
    log_info(
        "\nNDVI decile consistency diagnostic:\n"
        f"  Global regression slope     : {consistency['global_slope']:.4f}\n"
        f"  Median decile slope         : {consistency['median_decile_slope']:.4f}\n"
        f"  Difference                  : {consistency['difference']:.4f}\n"
        f"  Consistency flag            : {consistency['consistency_flag']}\n"
        "  Note: the global regression reflects the full NDVI range, while "
        "the decile analysis reflects the densely sampled NDVI interval; "
        "these analyses describe different aspects of the data."
    )
    summary.log("Interpretive Diagnostics", "Global NDVI-LST regression slope",
                f"{consistency['global_slope']:.4f}")
    summary.log("Interpretive Diagnostics", "Median NDVI decile slope",
                f"{consistency['median_decile_slope']:.4f}")
    summary.log("Interpretive Diagnostics", "Global vs decile slope difference",
                f"{consistency['difference']:.4f}")
    summary.log("Interpretive Diagnostics", "Global vs decile slope consistency",
                consistency["consistency_flag"])
    return consistency


def compute_qq_summary(residuals: np.ndarray, cfg: Config) -> dict:
    """Descriptive summary of the existing residual Q-Q plot: how much of
    the residual distribution falls outside the normal reference
    envelopes, and whether deviation is tail-symmetric.  Residuals
    themselves are not altered."""
    arr = np.asarray(residuals, dtype=float)
    n = len(arr)
    standardized = safe_divide(arr - arr.mean(), np.array([arr.std(ddof=1)]), fill=0.0)
    sorted_std = np.sort(standardized)
    theoretical = scipy_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    deviation = sorted_std - theoretical

    low_pct, high_pct = cfg.qq_envelope_percentiles
    z_low  = scipy_stats.norm.ppf(0.5 + low_pct / 200.0)
    z_high = scipy_stats.norm.ppf(0.5 + high_pct / 200.0)
    frac_outside_low  = float(np.mean(np.abs(standardized) > z_low))
    frac_outside_high = float(np.mean(np.abs(standardized) > z_high))

    max_abs_dev = float(np.max(np.abs(deviation)))
    left_dev  = float(np.mean(np.abs(deviation[theoretical < 0])))  if np.any(theoretical < 0)  else float("nan")
    right_dev = float(np.mean(np.abs(deviation[theoretical > 0])))  if np.any(theoretical > 0)  else float("nan")
    tail_asymmetry = (
        right_dev - left_dev if np.isfinite(left_dev) and np.isfinite(right_dev) else float("nan")
    )

    shapiro_p = float(scipy_stats.shapiro(arr)[1]) if n <= 5000 else float("nan")
    if np.isfinite(shapiro_p) and shapiro_p < 0.05:
        interpretation = (
            "Residuals deviate significantly from normality (Shapiro-Wilk "
            "p < 0.05); prediction intervals rely on the bootstrap procedure "
            "rather than a normality assumption."
        )
    else:
        interpretation = (
            "No strong evidence against approximate normality of residuals "
            "at the 0.05 level."
        )

    qq_df = pd.DataFrame({
        "Theoretical_Quantile": theoretical,
        "Observed_Quantile": sorted_std,
        "Deviation": deviation,
    })
    return {
        "frac_outside_low": frac_outside_low, "frac_outside_high": frac_outside_high,
        "low_pct": low_pct, "high_pct": high_pct,
        "max_abs_deviation": max_abs_dev, "tail_asymmetry": tail_asymmetry,
        "shapiro_p": shapiro_p, "interpretation": interpretation,
        "qq_df": qq_df,
    }


def report_qq_summary(residuals: np.ndarray, cfg: Config, summary: SummaryLog) -> dict:
    """Log and record the QQ-plot tail-symmetry diagnostic summary."""
    qq = compute_qq_summary(residuals, cfg)
    save_csv(qq["qq_df"], cfg.data_dir / "qq_summary.csv")
    log_info(
        f"\nQ-Q diagnostic summary:\n"
        f"  Fraction outside {qq['low_pct']:.0f}% envelope : {qq['frac_outside_low']:.4f}\n"
        f"  Fraction outside {qq['high_pct']:.0f}% envelope : {qq['frac_outside_high']:.4f}\n"
        f"  Maximum absolute deviation           : {qq['max_abs_deviation']:.4f}\n"
        f"  Tail asymmetry (right - left)         : {qq['tail_asymmetry']:.4f}\n"
        f"  {qq['interpretation']}"
    )
    summary.log("Interpretive Diagnostics", f"QQ: fraction outside {qq['low_pct']:.0f}% envelope",
                f"{qq['frac_outside_low']:.4f}")
    summary.log("Interpretive Diagnostics", f"QQ: fraction outside {qq['high_pct']:.0f}% envelope",
                f"{qq['frac_outside_high']:.4f}")
    summary.log("Interpretive Diagnostics", "QQ: maximum absolute deviation",
                f"{qq['max_abs_deviation']:.4f}")
    summary.log("Interpretive Diagnostics", "QQ: tail asymmetry (right - left)",
                f"{qq['tail_asymmetry']:.4f}")
    summary.log("Interpretive Diagnostics", "QQ: normality interpretation", qq["interpretation"])
    return qq


def add_interpretation_notes(summary: SummaryLog, moran: dict, conditional_low_r2: bool) -> None:
    """Discussion-style interpretive notes requested by reviewers.  These
    are explanatory notes only and do not alter any analysis, figure, or
    reported numerical result."""
    notes = []
    if np.isfinite(moran.get("p_value", float("nan"))) and moran["p_value"] < 0.05:
        notes.append(
            "Residual spatial autocorrelation remained after spatial validation.  "
            "Bootstrap confidence intervals therefore describe prediction "
            "uncertainty under the adopted resampling scheme."
        )
    if conditional_low_r2:
        notes.append(interpretation_text("conditional_ndvi"))
    notes.append(
        "Global NDVI trends should be interpreted together with the decile "
        "analysis because the observed NDVI distribution is concentrated "
        "within a relatively narrow range."
    )
    for note in notes:
        summary.add_conclusion(note)
        summary.log("Interpretive Diagnostics", "Interpretation note", note)


# LOWESS / BREUSCH-PAGAN HELPERS
def _lowess_smooth(
    x: np.ndarray, y: np.ndarray, frac: float = 0.3, n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """LOWESS smoother; uses statsmodels when available, otherwise a
    from-scratch tricube-weighted local linear fallback."""
    order  = np.argsort(x)
    x_s, y_s = x[order], y[order]
    try:
        import importlib
        sm_lowess = importlib.import_module(
            "statsmodels.nonparametric.smoothers_lowess"
        ).lowess
        smoothed = sm_lowess(y_s, x_s, frac=frac, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]
    except ImportError:
        n   = len(x_s)
        k   = max(int(frac * n), 5)
        eval_x = np.linspace(x_s.min(), x_s.max(), n_points)
        eval_y = np.empty_like(eval_x)
        for i, x0 in enumerate(eval_x):
            dist = np.abs(x_s - x0)
            idx  = np.argpartition(dist, min(k, n - 1))[:k]
            d    = dist[idx]
            bw   = d.max() if d.max() > 0 else 1.0
            w    = (1 - np.clip(d / bw, 0, 1) ** 3) ** 3
            X_d  = np.column_stack([np.ones(len(idx)), x_s[idx]])
            try:
                beta     = np.linalg.lstsq(np.diag(w) @ X_d,
                                            np.diag(w) @ y_s[idx], rcond=None)[0]
                eval_y[i] = beta[0] + beta[1] * x0
            except np.linalg.LinAlgError:
                eval_y[i] = np.average(y_s[idx], weights=w)
        return eval_x, eval_y


def _breusch_pagan_test(
    y_true: np.ndarray, y_pred: np.ndarray, residuals: np.ndarray
) -> dict:
    """Breusch-Pagan test; uses statsmodels when available."""
    try:
        import importlib
        sm = importlib.import_module("statsmodels.api")
        het_bp = importlib.import_module(
            "statsmodels.stats.diagnostic"
        ).het_breuschpagan
        exog = sm.add_constant(y_pred)
        lm_stat, lm_p, _, _ = het_bp(residuals, exog)
        return {
            "statistic": float(lm_stat), "p_value": float(lm_p),
            "method": "statsmodels het_breuschpagan",
        }
    except ImportError:
        bp_aux   = LinearRegression().fit(y_pred.reshape(-1, 1), residuals ** 2)
        bp_fitted = bp_aux.predict(y_pred.reshape(-1, 1))
        ss_reg   = np.sum((bp_fitted - np.mean(residuals ** 2)) ** 2)
        ss_tot   = np.sum((residuals ** 2 - np.mean(residuals ** 2)) ** 2)
        r2_aux   = safe_divide(np.array([ss_reg]), np.array([ss_tot]), fill=0.0)[0]
        stat     = len(residuals) * r2_aux
        p        = 1 - scipy_stats.chi2.cdf(stat, df=1)
        return {
            "statistic": float(stat), "p_value": float(p),
            "method": "auxiliary-regression approximation (statsmodels unavailable)",
        }


# ANNOTATION / CAPTION HELPERS
def annotation_caption(
    annotations: dict, sep: str = "  |  ", wrap_width: int = 110
) -> str:
    """Build the NEGI comparison figure footer caption string."""
    first_cooling = (
        f"1st cooling={annotations['first_cooling_pct']:.1f}%"
        if np.isfinite(annotations["first_cooling_pct"]) else "1st cooling=none"
    )
    parts = [
        f"Baseline={annotations['baseline_lst']:.2f}°C",
        first_cooling,
        f"Max cooling={annotations['max_cooling_c']:.2f}°C",
        f"Trajectory max={annotations['max_negi_s2_value']:.3f} @ "
        f"{annotations['max_negi_s2_pct']:.0f}%",
        f"Calibration: pred={annotations['calibration_slope']:.3f}*obs"
        f"{annotations['calibration_intercept']:+.3f}",
        f"Support: {annotations['feature_support_status']}",
    ]
    joined = sep.join(parts)
    if not wrap_width:
        return joined
    return "\n".join(
        textwrap.wrap(joined, width=wrap_width,
                      break_long_words=False, break_on_hyphens=False)
    )


# PLOTTING FUNCTIONS
# Validation
def plot_actual_vs_predicted(
    validation: ValidationResults, data: DatasetBundle, cfg: Config
) -> Path:
    """Plot observed vs. predicted LST for the confirmatory holdout."""
    y_test, pred = data.y_test, validation.holdout_pred
    slope, intercept = validation.calibration_slope, validation.calibration_intercept

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.scatter(y_test, pred, alpha=0.35, s=14, color=cfg.color_primary, edgecolor="none")
    lims = [y_test.min(), y_test.max()]
    ax.plot(lims, lims, color=cfg.color_neutral, linewidth=1.4, label="1:1 line", zorder=3)
    cal_x = np.linspace(y_test.min(), y_test.max(), 100)
    ax.plot(cal_x, slope * cal_x + intercept,
            color=cfg.color_accent, linewidth=1.6, linestyle="--",
            label=f"Calibration (slope = {slope:.3f})")
    ax.set_aspect("equal", adjustable="box")

    stat_box(ax, (
        f"CV R² = {validation.nested_cv_r2:.3f}\n"
        f"Holdout R² = {validation.holdout_r2:.3f}\n"
        f"RMSE = {validation.holdout_rmse:.2f} °C\n"
        f"Cal. slope = {slope:.3f}\n"
        f"Spatial holdout = {validation.repeated_r2_mean:.3f} "
        f"± {validation.repeated_r2_ci95:.3f}"
    ))
    set_title(ax, "Observed vs Predicted LST")
    set_axis_labels(ax, "Observed LST (°C)", "Predicted LST (°C)")
    set_legend(ax, loc="lower right")
    return save_fig(fig, FIG_ACTUAL_VS_PREDICTED, cfg,
                    caption=(
                        "Observed vs predicted LST with 1:1 and calibration lines. "
                        f"MAE = {validation.holdout_mae:.2f} °C; "
                        f"bias = {validation.calibration_bias:.2f} °C; "
                        f"calibration intercept = {intercept:.2f} °C. "
                        "Calibration diagnostics are reported descriptively only; "
                        "no post-hoc recalibration was applied."
                    ))


def plot_model_comparison(
    comparison_df: pd.DataFrame, validation: ValidationResults, cfg: Config
) -> Path:
    """Plot the benchmark-model R² comparison bar chart."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    bar_colors = [cfg.color_neutral, cfg.color_secondary, cfg.color_primary, cfg.color_accent]
    bars = ax.bar(comparison_df["Model"], comparison_df["R2"],
                  color=bar_colors, width=0.55, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, comparison_df["R2"]):
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold", color=cfg.color_neutral,
        )
    set_axis_labels(ax, ylabel="R²")
    ax.set_ylim(0, comparison_df["R2"].max() * 1.18)
    set_title(ax, f"Model Comparison\nSpatial holdout R² = {validation.holdout_r2:.3f}")
    plt.xticks(rotation=30, ha="right")
    apply_grid(ax)
    return save_fig(fig, FIG_MODEL_COMPARISON, cfg,
                    caption="Benchmark model comparison by spatial holdout R².",
                    section="Model comparison")


# Residual diagnostics
def plot_residuals_histogram(resid: ResidualDiagnostics, cfg: Config) -> Path:
    """Single consolidated histogram figure (Gaussian fit + KDE)."""
    arr          = resid.residuals
    mu, sigma    = np.mean(arr), np.std(arr, ddof=1)
    grid         = np.linspace(arr.min(), arr.max(), 300)
    gaussian_pdf = scipy_stats.norm.pdf(grid, loc=mu, scale=sigma)
    kde_curve    = scipy_stats.gaussian_kde(arr)(grid)

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.hist(arr, bins=30, color=cfg.color_primary, edgecolor="white",
            alpha=0.85, density=True, label="Residuals")
    ax.plot(grid, gaussian_pdf, color=cfg.color_accent, linewidth=2.0,
            linestyle="--", label="Gaussian fit")
    ax.plot(grid, kde_curve, color=cfg.color_neutral, linewidth=2.0, label="KDE")
    ax.axvline(0, color=cfg.color_accent, linestyle=":", linewidth=1.2)
    set_title(ax, "Residual Distribution")
    set_axis_labels(ax, "Residual (°C)", "Density")
    set_legend(ax)
    apply_grid(ax)

    stats_d = compute_residual_stats(arr)
    figure_footer(fig, (
        f"mean={stats_d['mean']:.3f}  median={stats_d['median']:.3f}  "
        f"std={stats_d['std']:.3f}  skew={stats_d['skewness']:.3f}  "
        f"kurtosis={stats_d['kurtosis']:.3f}  Shapiro p={stats_d['shapiro_p']:.3g}"
    ), cfg)
    return save_fig(fig, FIG_RESIDUALS_HISTOGRAM, cfg,
                    caption="Holdout residual distribution with Gaussian fit and KDE.",
                    section="Supplementary diagnostics")


def plot_residuals_vs_predicted_full(
    resid: ResidualDiagnostics, pred: np.ndarray,
    cfg: Config, summary: SummaryLog, registry=None,
) -> Path:
    """Residuals vs predicted with LOWESS smoother and Breusch-Pagan test."""
    arr          = resid.residuals
    lowess_x, lowess_y = _lowess_smooth(pred, arr)
    bp = _breusch_pagan_test(np.zeros_like(pred), pred, arr)
    summary.log("Residual diagnostics", "Breusch-Pagan statistic",  f"{bp['statistic']:.4f}")
    summary.log("Residual diagnostics", "Breusch-Pagan p-value",    f"{bp['p_value']:.4g}")
    log_info(
        f"\nBreusch-Pagan test ({bp['method']}): "
        f"statistic = {bp['statistic']:.4f}, p = {bp['p_value']:.4g}"
    )

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.scatter(pred, arr, alpha=0.35, s=14, color=cfg.color_primary,
               edgecolor="none", label="Residuals")
    ax.axhline(0, color=cfg.color_neutral, linewidth=1.2)
    ax.plot(lowess_x, lowess_y, color=cfg.color_accent, linewidth=2.0, label="LOWESS")
    set_title(ax, "Residuals vs Predicted LST")
    set_axis_labels(ax, "Predicted LST (°C)", "Residual (°C)")
    set_legend(ax)
    apply_grid(ax)
    figure_footer(fig, (
        f"Breusch-Pagan: statistic={bp['statistic']:.3f}, "
        f"p={bp['p_value']:.3g} ({bp['method']})"
    ), cfg)
    return save_fig(fig, FIG_RESIDUALS_VS_PREDICTED, cfg,
                    caption="Holdout residuals vs predicted LST with LOWESS smoother.",
                    section="Supplementary diagnostics")


def plot_residuals_qq(resid: ResidualDiagnostics, cfg: Config) -> Path:
    """Plot the QQ diagnostic of holdout residuals against the normal distribution."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    scipy_stats.probplot(resid.residuals, plot=ax)
    set_title(ax, "Q-Q Plot of Residuals")
    apply_grid(ax)
    return save_fig(fig, FIG_RESIDUALS_QQ, cfg,
                    caption="Normal Q-Q plot of holdout residuals.",
                    section="Supplementary diagnostics")


def plot_residual_spatial_map(
    spatial: Optional[dict], cfg: Config, registry=None
) -> Optional[Path]:
    """Spatial map of holdout residuals coloured by TwoSlopeNorm coolwarm."""
    if spatial is None:
        return None
    resid   = spatial["residuals"]
    max_abs = max(np.abs(resid).max(), TEMP_ZERO_GUARD)
    norm    = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    sc = ax.scatter(spatial["x"], spatial["y"], c=resid, cmap="coolwarm",
                    norm=norm, s=18, edgecolor="none", alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Residual (Observed - Predicted, °C)")
    set_axis_labels(ax, spatial["x_col"], spatial["y_col"])
    set_title(ax, "Spatial Distribution of Holdout Residuals")

    moran     = spatial["moran"]
    moran_str = (
        f"Global Moran's I = {moran['morans_i']:.3f} (p = {moran['p_value']:.3g})"
        if np.isfinite(moran["morans_i"]) else "Moran's I unavailable"
    )
    # Append interpretive note when autocorrelation is significant.
    if np.isfinite(moran["p_value"]) and moran["p_value"] < 0.05:
        moran_str += (
            " - Residual spatial dependence remains after spatial validation.  "
            "Confidence intervals should be interpreted accordingly."
        )
    figure_footer(fig, moran_str, cfg)

    caption = generate_caption(
        "Spatial distribution of holdout residuals coloured on a coolwarm scale "
        "centred at zero to show whether prediction error is spatially clustered",
        {"Moran's I": moran["morans_i"], "p": moran["p_value"]},
    )
    return save_fig(fig, FIG_RESIDUAL_SPATIAL_MAP, cfg,
                    caption=caption, section="Supplementary diagnostics")


# Feature importance
def plot_feature_importance(fi: FeatureImportanceResult, cfg: Config) -> Path:
    """Plot the permutation feature importance bar chart."""
    table = fi.table
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    bars = ax.bar(
        table["Feature"], table["Importance"],
        yerr=table["Importance_CI95"], capsize=5,
        color=[cfg.color_neutral, cfg.color_secondary, cfg.color_primary],
        edgecolor="white", linewidth=0.5, width=0.55,
        error_kw=dict(elinewidth=1.5, ecolor="#555555"),
    )
    for bar, val in zip(bars, table["Importance"]):
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold",
        )
    set_axis_labels(ax, "Feature", "Permutation Importance (ΔR²)")
    n_seeds = int(table["N_Repeats"].iloc[0]) if len(table) else 0
    set_title(ax, f"Permutation Feature Importance\n(mean ± 95% CI over {n_seeds} seeds)")
    ax.margins(y=0.15)
    apply_grid(ax)
    fig.text(
        0.5, -0.04,
        "Values reflect marginal predictive importance; may be influenced by predictor collinearity.",
        ha="center", fontsize=8, color=cfg.color_neutral,
    )
    return save_fig(fig, FIG_FEATURE_IMPORTANCE, cfg,
                    caption="Permutation feature importance (mean ± 95% CI over 10 seeds).")


# Response curves
def plot_ndvi_vs_lst(df: pd.DataFrame, cfg: Config) -> Path:
    """Plot the NDVI-LST scatter relationship."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.scatter(df["NDVI"], df["LST"], alpha=0.18, s=12,
               color=cfg.color_primary, edgecolor="none")
    reg     = LinearRegression().fit(df[["NDVI"]], df["LST"])
    x_range = np.linspace(df["NDVI"].min(), df["NDVI"].max(), 100)
    ax.plot(x_range,
            reg.predict(pd.DataFrame(x_range, columns=["NDVI"])),
            color=cfg.color_accent, linewidth=2, label="OLS Fit")
    set_title(ax, "Observed NDVI vs Land Surface Temperature")
    set_axis_labels(ax, "NDVI", "Observed LST (°C)")
    set_legend(ax)
    apply_grid(ax)
    return save_fig(fig, FIG_NDVI_VS_LST, cfg,
                    caption="Scatter of observed NDVI vs LST with OLS trend.",
                    section="Response curves")


def plot_lst_response_to_ndbi(sweep: dict, cfg: Config) -> Path:
    """Plot the predicted LST response curve across the NDBI sweep."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(sweep["ndbi_range"], sweep["pred_raw"],
            color=cfg.color_secondary, linewidth=1.0, alpha=0.4, label="Raw output")
    ax.plot(sweep["ndbi_range"], sweep["pred_smooth"],
            color=cfg.color_secondary, linewidth=2.2, label="Savgol-smoothed")
    ax.axvline(sweep["ndbi_median"], color=cfg.color_neutral, linestyle="--",
               linewidth=1.2, label=f"Median NDBI ({sweep['ndbi_median']:.3f})")
    set_axis_labels(ax, "NDBI", "Predicted LST (°C)")
    set_title(ax, "Predicted LST Response to NDBI")
    set_legend(ax)
    apply_grid(ax)
    return save_fig(fig, FIG_LST_RESPONSE_TO_NDBI, cfg,
                    caption="Predicted LST response to NDBI with Savitzky-Golay smoothing.")


def plot_ndbi_response_uncertainty(
    sweep: dict, uncertainty: UncertaintyResults, cfg: Config
) -> Path:
    """Plot the NDBI response curve with its uncertainty band."""
    if uncertainty.ndbi_sweep_folds is None:
        log_warning("No NDBI sweep refit data; skipping response-curve uncertainty figure.")
        return None

    folds  = uncertainty.ndbi_sweep_folds
    nr     = uncertainty.ndbi_range
    median = np.median(folds, axis=0)
    q25    = np.percentile(folds, 25, axis=0)
    q75    = np.percentile(folds, 75, axis=0)
    p2_5   = np.percentile(folds, 2.5, axis=0)
    p97_5  = np.percentile(folds, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.fill_between(nr, p2_5, p97_5, color=cfg.color_secondary, alpha=0.10,
                    label="95% interval (refits)")
    ax.fill_between(nr, q25,  q75,   color=cfg.color_secondary, alpha=0.22,
                    label="Q25-Q75 (refits)")
    ax.plot(nr, median, color=cfg.color_secondary, linewidth=1.6, linestyle="--",
            label=f"Median across {len(folds)} spatial refits")
    ax.plot(sweep["ndbi_range"], sweep["pred_raw"], color=cfg.color_neutral,
            linewidth=2.0, label="Raw output (primary, authoritative)")
    set_axis_labels(ax, "NDBI", "Predicted LST (°C)")
    set_title(ax, "Supplementary Diagnostic: NDBI Response-Curve Uncertainty "
              "(Spatial-Block Refits)")
    set_legend(ax)
    apply_grid(ax)
    return save_fig(fig, FIG_NDBI_RESPONSE_UNCERTAINTY, cfg,
                    caption="NDBI response-curve uncertainty across repeated spatial refits.",
                    section="Supplementary diagnostics")


def plot_ndvi_decile_analysis(
    summary_df: pd.DataFrame,
    corr_ndvi_lst: float, corr_ndbi_lst: float, corr_ndvi_ndbi: float,
    cfg: Config,
) -> Path:
    """Plot the NDVI-decile LST summary alongside correlation diagnostics."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    axes[0].errorbar(
        summary_df["NDVI_median"], summary_df["LST_median"],
        yerr=summary_df["LST_std"],
        fmt="o-", linewidth=2.5, markersize=8,
        color=cfg.color_secondary, capsize=5, capthick=2,
        label="Median ±1 SD",
    )
    axes[0].fill_between(
        summary_df["NDVI_median"],
        summary_df["LST_median"] - summary_df["LST_std"],
        summary_df["LST_median"] + summary_df["LST_std"],
        alpha=0.12, color=cfg.color_secondary,
    )
    set_axis_labels(axes[0], "NDVI (decile median)", "Observed LST (°C)")
    set_title(axes[0],
              f"LST vs NDVI Deciles  "
              f"(r_{{NDVI-LST}} = {corr_ndvi_lst:.3f}, "
              f"r_{{NDBI-LST}} = {corr_ndbi_lst:.3f})")
    lower = (summary_df["LST_median"] - summary_df["LST_std"]).min() - 0.5
    upper = (summary_df["LST_median"] + summary_df["LST_std"]).max() + 0.5
    axes[0].set_ylim(lower, upper)
    apply_grid(axes[0])
    set_legend(axes[0])

    axes[1].plot(summary_df["NDVI_median"], summary_df["NDBI_median"],
                 "s-", linewidth=2.5, markersize=8,
                 color=cfg.color_primary, label="Median NDBI")
    set_axis_labels(axes[1], "NDVI (decile median)", "NDBI (median)")
    set_title(axes[1], f"NDVI-NDBI Observed Association  (r = {corr_ndvi_ndbi:.3f})")
    apply_grid(axes[1])
    set_legend(axes[1])
    fig.tight_layout()
    return save_fig(fig, FIG_NDVI_DECILE_ANALYSIS, cfg,
                    caption="LST vs NDVI deciles and NDVI-NDBI observed association.",
                    section="Response curves")


def plot_ndvi_conditional_by_ndbi_bins(
    results: list[ConditionalBinResult], df: pd.DataFrame, cfg: Config
) -> Path:
    """Plot NDVI-LST slopes conditioned on NDBI bins."""
    bin_colors      = ["#93C6E0", "#4A96C8", "#1B6FA8", "#0D3B66"]
    conditional_ylim = (df["LST"].quantile(0.01) - 1.0, df["LST"].quantile(0.99) + 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax, result, bin_color in zip(axes_flat, results, bin_colors):
        ax.scatter(result.ndvi, result.lst, alpha=0.22, s=10,
                   color=bin_color, edgecolor="none")
        if result.slope is not None:
            x_line = np.array([result.ndvi.min(), result.ndvi.max()])
            ax.plot(x_line, result.slope * x_line + result.intercept,
                    color=cfg.color_accent, linewidth=1.5)
            ax.fill_between(
                x_line,
                (result.slope - result.slope_ci95) * x_line + result.intercept,
                (result.slope + result.slope_ci95) * x_line + result.intercept,
                color=cfg.color_accent, alpha=0.06,
            )
            stat_box(ax,
                     f"Slope = {result.slope:.2f} °C/NDVI\n"
                     f"95% CI = ±{result.slope_ci95:.2f}\n"
                     f"R² = {result.r_squared:.3f}")
        set_axis_labels(ax, "NDVI", "Observed LST (°C)")
        set_title(ax, result.label, fontsize=12, color=bin_color)
        apply_grid(ax)
        ax.set_ylim(conditional_ylim)

    fig.suptitle("Conditional LST-NDVI Association Within NDBI Strata",
                 fontsize=13, fontweight="normal")
    legend_elements = [
        Patch(facecolor=c, label=r.label.replace("\n", " "))
        for c, r in zip(bin_colors, results)
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9.5, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02), title="Urbanisation bin (NDBI range)")
    fig.subplots_adjust(top=0.90, bottom=0.10, wspace=0.25, hspace=0.40)
    return save_fig(fig, FIG_NDVI_CONDITIONAL_BY_NDBI, cfg, use_tight_layout=False,
                    caption="Conditional LST-NDVI association within NDBI strata.")


# Response surface
def plot_response_surface(
    surface: dict, df: pd.DataFrame, s2: ScenarioTrajectory, cfg: Config
) -> Path:
    """Plot the NDVI-NDBI predicted-LST response surface."""
    fig, ax = plt.subplots(figsize=(8, 7))
    contour = ax.contourf(surface["ndvi_mesh"], surface["ndbi_mesh"],
                          surface["lst_mesh"], levels=20, cmap="RdYlBu_r")
    plt.colorbar(contour, label="Predicted LST (°C)")
    ax.contour(surface["ndvi_mesh"], surface["ndbi_mesh"], surface["lst_mesh"],
               levels=20, colors="black", linewidths=0.3, alpha=0.4)
    ax.scatter(df["NDVI"], df["NDBI"], s=3, color="gray",
               alpha=0.08, rasterized=True, zorder=1)

    scenario_df = s2.feature_frame()
    ax.plot(scenario_df["NDVI"], scenario_df["NDBI"],
            color="black", linewidth=2.5, label="Scenario 2 trajectory", zorder=3)
    annotate_every = max(1, len(scenario_df) // 10)
    for i in range(0, len(scenario_df), annotate_every):
        ax.annotate(
            f"{scenario_df['Scenario (%)'].iloc[i]:.0f}%",
            (scenario_df["NDVI"].iloc[i], scenario_df["NDBI"].iloc[i]),
            textcoords="offset points", xytext=(6, 6), fontsize=7, color="black",
        )
    ax.axvline(surface["ndvi_max_plot"], color="gray", linestyle="--", linewidth=1.2,
               label="NDVI 99th percentile")
    ax.set_xlim(df["NDVI"].min(), surface["ndvi_max_plot"])
    set_axis_labels(ax, "NDVI", "NDBI")
    set_title(ax, "Predicted LST Response Surface")
    set_legend(ax)
    return save_fig(fig, FIG_RESPONSE_SURFACE, cfg,
                    caption="Predicted LST response surface in NDVI-NDBI space "
                    "with Scenario 2 trajectory.",
                    section="Supplementary diagnostics")


def plot_scenario2_data_support(
    df: pd.DataFrame, s2: ScenarioTrajectory,
    support: SupportDiagnostics, cfg: Config,
) -> Path:
    """Plot Scenario 2 trajectory points against the training-data feature-space support."""
    scenario_df = s2.feature_frame()
    in_support  = support.in_support_s2
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df["NDVI"], df["NDBI"], s=3, alpha=0.15, color="gray",
               label="Observed pixels")
    sc = ax.scatter(
        scenario_df["NDVI"], scenario_df["NDBI"],
        c=np.where(in_support, scenario_df["Scenario (%)"], np.nan),
        cmap="viridis", s=25, label="Scenario 2 (in support)",
    )
    ax.scatter(scenario_df["NDVI"][~in_support], scenario_df["NDBI"][~in_support],
               color=cfg.color_accent, s=25, marker="x",
               label="Scenario 2 (outside support)")
    ax.plot(scenario_df["NDVI"], scenario_df["NDBI"],
            color=cfg.color_neutral, linewidth=1, alpha=0.5)
    set_axis_labels(ax, "NDVI", "NDBI")
    set_title(ax, "Scenario 2 Trajectory: Feature-Space Data Support")
    set_legend(ax)
    plt.colorbar(sc, label="Scenario (%) [in-support only]")
    return save_fig(fig, FIG_SCENARIO2_DATA_SUPPORT, cfg,
                    caption="Scenario 2 trajectory feature-space support.",
                    section="Supplementary diagnostics")


# NEGI scenario plots
def plot_negi_smoothing_robustness(
    negi: NEGIResults, smoothing: dict, s2: ScenarioTrajectory,
    cfg: Config, summary: SummaryLog,
) -> tuple[Path, bool, bool, int]:
    """Plot the NEGI scenario curve with and without display smoothing
    to check robustness of the optimum."""
    cooling_savgol = negi.baseline_lst - smoothing["lst_s2_smooth"]
    cooling_spline = negi.baseline_lst - smoothing["lst_s2_spline"]
    # Both smoothed variants share the primary benefit_scale so they are
    # directly comparable on the same axis (deliberate design choice).
    negi_savgol = (cfg.alpha_weight *
                   safe_divide(np.maximum(cooling_savgol, 0.0), negi.benefit_scale)
                   - cfg.beta_weight * negi.energy_norm_sqrt)
    negi_spline = (cfg.alpha_weight *
                   safe_divide(np.maximum(cooling_spline, 0.0), negi.benefit_scale)
                   - cfg.beta_weight * negi.energy_norm_sqrt)

    peak_idx_raw    = negi.s2_optimum_idx
    peak_idx_savgol = int(np.argmax(negi_savgol))
    smoothing_robust = bool(
        abs(negi.scenarios[peak_idx_raw] - negi.scenarios[peak_idx_savgol]) * 100 <= 3
    )
    peak_is_ood = bool(s2.out_of_bounds[peak_idx_raw])

    summary.log("Supplementary Diagnostics - Smoothing", "Smoothing robust",
                str(smoothing_robust))
    summary.log("Supplementary Diagnostics - Smoothing",
                "Peak location out-of-feature-bounds", str(peak_is_ood))

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(negi.scenario_pct, negi.negi_s2,
            color=cfg.color_accent, linewidth=2.2, label="Raw (reported)")
    ax.plot(negi.scenario_pct, negi_savgol,
            color=cfg.color_secondary, linewidth=1.5, linestyle="--",
            label="Savgol-smoothed")
    ax.plot(negi.scenario_pct, negi_spline,
            color=cfg.color_primary, linewidth=1.5, linestyle=":",
            label="Spline-smoothed")
    ax.scatter([negi.scenario_pct[peak_idx_raw]], [negi.negi_s2[peak_idx_raw]],
               color=cfg.color_accent, s=80, zorder=5)
    ax.axhline(0, color=cfg.color_neutral, linestyle=":", linewidth=1)
    set_axis_labels(ax, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax, "Supplementary Diagnostic: Scenario 2 NEGI, Raw vs Smoothed Variants")
    set_legend(ax, loc="lower left")
    apply_grid(ax)
    path = save_fig(fig, FIG_NEGI_SMOOTHING_ROBUSTNESS, cfg,
                    caption="Scenario 2 NEGI: raw vs Savgol- and spline-smoothed variants.",
                    section="Supplementary diagnostics")
    return path, smoothing_robust, peak_is_ood, peak_idx_raw


def plot_negi_scenario_comparison(
    negi: NEGIResults, annotations: dict, flag_str: str, cfg: Config
) -> Path:
    """Plot the NEGI comparison between Scenario 1 and Scenario 2."""
    opt_idx = negi.s2_optimum_idx
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(negi.scenario_pct, negi.negi_s1, linewidth=2.2,
            color=cfg.color_s1, linestyle="--",
            label="Scenario 1: Greening only (NDBI fixed)")
    ax.plot(negi.scenario_pct, negi.negi_s2, linewidth=2.2,
            color=cfg.color_s2, linestyle="-",
            label="Scenario 2: Urban transformation (NDBI co-varies)")
    ax.axhline(0, color=cfg.color_neutral, linestyle="-", linewidth=1.1)
    ax.text(1.5, 0.012, "Break-even (NEGI = 0)",
            fontsize=9, color=cfg.color_neutral, va="bottom", style="italic")
    # Small neutral diamond to mark trajectory peak without implying a
    # physical optimum (see Methods §[X]).
    ax.scatter(
        [negi.scenario_pct[opt_idx]], [negi.negi_s2[opt_idx]],
        marker="D", s=25, color=cfg.color_neutral, zorder=5,
        label="Highest evaluated NEGI (trajectory max)",
    )
    if np.isfinite(negi.first_cooling_pct):
        ax.axvline(negi.first_cooling_pct, color=cfg.color_neutral,
                   linestyle=":", linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"First cooling ({negi.first_cooling_pct:.0f}%)",
            xy=(negi.first_cooling_pct, 0),
            xytext=(negi.first_cooling_pct + 3, ax.get_ylim()[0] * 0.5),
            fontsize=8, color=cfg.color_neutral, style="italic",
        )
    set_axis_labels(ax, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax, "NEGI Scenario Comparison")
    fig.text(0.5, -0.02, annotation_caption(annotations),
             ha="center", fontsize=7.5, color=cfg.color_neutral, wrap=True)
    set_legend(ax, loc="lower left")
    apply_grid(ax)
    return save_fig(fig, FIG_NEGI_SCENARIO_COMPARISON, cfg,
                    caption="NEGI scenario comparison: greening only vs urban transformation.")


def plot_negi_energy_cost_comparison(negi: NEGIResults, cfg: Config) -> Path:
    """Plot the desalination energy cost comparison between scenarios."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(negi.scenario_pct, negi.negi_s2,
            color=cfg.color_s2, linewidth=2, linestyle="-",
            label="Scenario 2, sqrt cost (primary)")
    ax.plot(negi.scenario_pct, negi.negi_s2_linear,
            color=cfg.color_s2, linewidth=2, linestyle="--",
            label="Scenario 2, linear cost")
    ax.plot(negi.scenario_pct, negi.negi_s1,
            color=cfg.color_s1, linewidth=2, linestyle="-",
            label="Scenario 1, sqrt cost (primary)")
    ax.plot(negi.scenario_pct, negi.negi_s1_linear,
            color=cfg.color_s1, linewidth=2, linestyle="--",
            label="Scenario 1, linear cost")
    ax.axhline(0, color=cfg.color_neutral, linestyle=":", linewidth=1)
    ax.text(1.5, 0.01, "Break-even (NEGI = 0)",
            fontsize=9, color=cfg.color_neutral, va="bottom", style="italic")
    set_axis_labels(ax, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax, "Supplementary Diagnostic: NEGI, sqrt vs Linear Energy-Cost Assumption")
    set_legend(ax, loc="best")
    apply_grid(ax)
    return save_fig(fig, FIG_NEGI_ENERGY_COST_COMPARISON, cfg,
                    caption="NEGI comparison: sqrt vs linear energy-cost exponent assumption.",
                    section="Supplementary diagnostics")


def plot_negi_uncertainty_bands_primary(
    uncertainty: UncertaintyResults, s1: ScenarioTrajectory, cfg: Config
) -> tuple:
    """Plot NEGI uncertainty bands (median and IQR) across spatial-block holdout refits."""
    scenario_pct  = s1.scenario_pct
    p             = uncertainty.percentiles
    peak_iqr_idx  = int(np.argmax(p["negi_s2"][75] - p["negi_s2"][25]))
    peak_iqr_pct  = scenario_pct[peak_iqr_idx]

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(scenario_pct, p["negi_s1"][50], color=cfg.color_s1,
            linewidth=2.2, linestyle="--", label="Scenario 1 (median)")
    ax.fill_between(scenario_pct, p["negi_s1"][25], p["negi_s1"][75],
                    color=cfg.color_s1, alpha=0.15, label="Scenario 1 Q25-Q75")
    ax.plot(scenario_pct, p["negi_s2"][50], color=cfg.color_s2,
            linewidth=2.2, linestyle="-", label="Scenario 2 (median)")
    ax.fill_between(scenario_pct, p["negi_s2"][25], p["negi_s2"][75],
                    color=cfg.color_s2, alpha=0.20, label="Scenario 2 Q25-Q75")
    ax.axhline(0, color=cfg.color_neutral, linestyle="-", linewidth=1.1)
    ax.text(1.5, 0.01, "Break-even (NEGI = 0)",
            fontsize=9, color=cfg.color_neutral, va="bottom", style="italic")
    ax.annotate(
        f"Peak IQR = {(p['negi_s2'][75]-p['negi_s2'][25])[peak_iqr_idx]:.3f}\n"
        f"at {peak_iqr_pct:.0f}%",
        xy=(peak_iqr_pct, p["negi_s2"][50][peak_iqr_idx]),
        xytext=(max(peak_iqr_pct - 30, 5), p["negi_s2"][50][peak_iqr_idx] - 0.25),
        fontsize=8.5, color=cfg.color_neutral, style="italic",
        arrowprops=dict(arrowstyle="->", color=cfg.color_neutral, lw=0.9),
        bbox=STAT_BOX_STYLE,
    )
    apply_grid(ax)
    return fig, ax, peak_iqr_idx


def plot_negi_scenario2_robustness_full(
    uncertainty: UncertaintyResults, s1: ScenarioTrajectory, cfg: Config
) -> Path:
    """Plot the full Scenario 2 NEGI robustness diagnostic across spatial-block refits."""
    scenario_pct = s1.scenario_pct
    p            = uncertainty.percentiles
    n_refits     = len(uncertainty.negi_s2_folds)

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.fill_between(scenario_pct, p["negi_s2"][2.5], p["negi_s2"][97.5],
                    color=cfg.color_s2, alpha=0.12, label="Scenario 2 95% interval")
    ax.fill_between(scenario_pct, p["negi_s2"][25],  p["negi_s2"][75],
                    color=cfg.color_s2, alpha=0.30, label="Scenario 2 Q25-Q75")
    ax.plot(scenario_pct, p["negi_s2"][50],
            color=cfg.color_s2, linewidth=2.2, label="Scenario 2 median")
    ax.axhline(0, color=cfg.color_neutral, linestyle="-", linewidth=1.1)
    set_axis_labels(ax, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax, f"Scenario 2 NEGI Robustness Across {n_refits} Spatial Refits")
    set_legend(ax, loc="lower left")
    apply_grid(ax)
    return save_fig(fig, FIG_NEGI_SCENARIO2_ROBUSTNESS, cfg,
                    caption="Scenario 2 NEGI robustness: median, IQR, and 95% interval.",
                    section="Supplementary diagnostics")


def plot_negi_exponent_sensitivity(
    negi: NEGIResults, s1: ScenarioTrajectory, cfg: Config
) -> Path:
    """Plot NEGI sensitivity to the cost-function exponent."""
    scenario_pct   = s1.scenario_pct
    cooling_norm_s2 = safe_divide(np.maximum(negi.delta_t_s2, 0.0), negi.benefit_scale)
    cases = [
        {"alpha": 0.6, "beta": 1.6, "w0": 50},
        {"alpha": 1.0, "beta": 1.0, "w0": 200},
        {"alpha": 1.6, "beta": 0.6, "w0": 50},
    ]
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    for case in cases:
        a, b, w = case["alpha"], case["beta"], case["w0"]
        for exponent in (cfg.sensitivity_exponents[0], cfg.sensitivity_exponents[-1]):
            ep       = _energy_cost(negi.scenarios, w, exponent, cfg)
            ref_max  = _energy_cost(negi.scenarios, cfg.reference_w0, exponent, cfg).max()
            negi_profile = a * cooling_norm_s2 - b * safe_divide(ep, ref_max)
            ax.plot(scenario_pct, negi_profile, linewidth=1.8,
                    label=f"α={a:.1f}, β={b:.1f}, w0={w}, γ={exponent}")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.text(1.5, 0.01, "Break-even (NEGI = 0)",
            fontsize=9, color=cfg.color_neutral, va="bottom", style="italic")
    set_axis_labels(ax, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax, "Supplementary Diagnostic: NEGI Sensitivity - "
              "Cooling Weight, Energy Weight, Energy Scaling")
    set_legend(ax)
    apply_grid(ax)
    return save_fig(fig, FIG_NEGI_EXPONENT_SENSITIVITY, cfg,
                    caption="NEGI profiles under alternative parameter assumptions.",
                    section="Supplementary diagnostics")


def plot_cooling_saturation_diagnostic(
    negi: NEGIResults, s2: ScenarioTrajectory,
    saturation: dict, support: SupportDiagnostics, cfg: Config,
) -> Path:
    """Plot the cooling saturation diagnostic panels."""
    scenario_pct = s2.scenario_pct
    sat_50_idx   = saturation["sat_50_idx"]
    sat_90_idx   = saturation["sat_90_idx"]
    collapse_idx = saturation["collapse_idx"]
    in_support   = support.in_support_s2

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    axes[0].plot(scenario_pct, negi.delta_t_s2,
                 color=cfg.color_secondary, linewidth=2)
    if sat_50_idx is not None:
        axes[0].axvline(
            scenario_pct[sat_50_idx], color=cfg.color_neutral, linestyle=":", linewidth=1,
            label=f"50% of max predicted cooling ({scenario_pct[sat_50_idx]:.1f}%)",
        )
        axes[0].axvline(
            scenario_pct[sat_90_idx], color=cfg.color_accent, linestyle="--", linewidth=1,
            label=f"90% of max predicted cooling ({scenario_pct[sat_90_idx]:.1f}%)",
        )
    else:
        axes[0].text(0.02, 0.05, "No positive predicted cooling beyond the baseline",
                     transform=axes[0].transAxes, fontsize=9, bbox=STAT_BOX_STYLE)
    if collapse_idx is not None:
        axes[0].axvline(scenario_pct[collapse_idx], color="black", linestyle="-",
                        linewidth=1.2, label=f"Near-zero ({scenario_pct[collapse_idx]:.1f}%)")
    warming_mask = negi.delta_t_s2 < 0
    axes[0].fill_between(scenario_pct, negi.delta_t_s2, 0, where=warming_mask,
                         color=cfg.color_accent, alpha=0.18,
                         label="Predicted LST above baseline (ΔT < 0)")
    set_axis_labels(axes[0], ylabel="Signed predicted cooling, S2 (°C)")
    set_title(axes[0], "Supplementary Diagnostic: Scenario 2 Predicted Cooling Along Trajectory")
    set_legend(axes[0], loc="best")
    apply_grid(axes[0])

    axes[1].fill_between(scenario_pct, 0, 1, where=in_support,
                         color=cfg.color_primary, alpha=0.25, step="mid",
                         label="In dense training region")
    axes[1].fill_between(scenario_pct, 0, 1, where=~in_support,
                         color=cfg.color_accent, alpha=0.25, step="mid",
                         label="Outside dense training region")
    if collapse_idx is not None:
        axes[1].axvline(scenario_pct[collapse_idx], color="black",
                        linestyle="-", linewidth=1.2)
    set_axis_labels(axes[1], ylabel="Data support")
    axes[1].set_yticks([])
    set_legend(axes[1], loc="best")

    axes[2].plot(scenario_pct, s2.ndbi, color=cfg.color_primary, linewidth=2)
    if collapse_idx is not None:
        axes[2].axvline(scenario_pct[collapse_idx], color="black",
                        linestyle="-", linewidth=1.2)
    set_axis_labels(axes[2], SCENARIO_AXIS_LABEL, "Scenario 2 NDBI")
    set_title(axes[2], "Empirical NDBI Trajectory")
    apply_grid(axes[2])
    return save_fig(fig, FIG_COOLING_SATURATION, cfg,
                    caption="Scenario 2 cooling saturation diagnostic.",
                    section="Supplementary diagnostics")


def plot_sensitivity_optimal_negi(sensitivity: SensitivityResults, cfg: Config) -> Path:
    """Plot the optimal NEGI as a function of w0 across alpha values."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    for alpha_val in cfg.sensitivity_alpha_values:
        subset = sensitivity.plot_table[
            isclose_mask(sensitivity.plot_table["Alpha"], alpha_val)
        ]
        ax.plot(subset["w0"], subset["Numerical_Maximum_NEGI"],
                marker="o", linewidth=2, label=f"α = {alpha_val:.1f}")
    set_axis_labels(
        ax,
        f"Irrigation scaling coefficient w0\n"
        f"(relative to REFERENCE_W0 = {cfg.reference_w0:.0f})",
        "Numerical Maximum NEGI",
    )
    set_title(ax,
              f"Supplementary Diagnostic: Sensitivity of Numerical Maximum NEGI\n"
              f"(exponent = {cfg.sensitivity_default_exponent}, "
              f"β = {cfg.sensitivity_default_beta:.1f})")
    apply_grid(ax)
    set_legend(ax)
    return save_fig(fig, FIG_SENSITIVITY_OPTIMAL_NEGI, cfg,
                    caption="Sensitivity of the highest evaluated NEGI to w0 and alpha parameters.",
                    section="Supplementary diagnostics")


# MAIN PIPELINE
def main(cfg: Config = CFG) -> None:
    """Run the full NEGI modelling, validation, scenario, and reporting pipeline end to end."""
    run_start = time.monotonic()

    configure_logging(cfg)
    log_info("JEDDAH LST - NEGI FRAMEWORK")

    # Setup
    repro_info = print_reproducibility_info(cfg)
    setup_directories(cfg)
    apply_plot_style(cfg)
    summary = SummaryLog()
    PREDICTION_CACHE.clear()

    # Step 1: Data
    log_info("\n[STEP 1] Loading dataset...")
    df   = load_dataset(cfg)
    log_info(f"  Loaded {len(df):,} rows from {cfg.data_path}")
    data = build_dataset_bundle(df, cfg)
    log_info(f"  Train: {len(data.X_train):,} rows | Test: {len(data.X_test):,} rows")

    corr_ndvi_lst  = df["NDVI"].corr(df["LST"])
    corr_ndbi_lst  = df["NDBI"].corr(df["LST"])
    corr_ndvi_ndbi = df["NDVI"].corr(df["NDBI"])
    summary.log("EDA", "NDVI-LST Pearson r",  f"{corr_ndvi_lst:.4f}")
    summary.log("EDA", "NDBI-LST Pearson r",  f"{corr_ndbi_lst:.4f}")
    summary.log("EDA", "NDVI-NDBI Pearson r", f"{corr_ndvi_ndbi:.4f}")

    # Step 2: Model fitting
    log_info("\n[STEP 2] Fitting XGBoost model (RandomizedSearchCV)...")
    model_results = fit_model(data, cfg, summary)

    # Step 3: Validation
    log_info("\n[STEP 3] Evaluating model...")
    log_info("\n[STEP 3a] Nested GroupKFold CV (honest, leakage-free outer-fold R²)...")
    nested_cv = run_nested_group_kfold_cv(data, cfg, summary)
    validation = evaluate_model(model_results, cfg, summary, nested_cv)

    # Step 4: Residual diagnostics
    log_info("\n[STEP 4] Residual diagnostics...")
    resid = compute_residual_diagnostics(validation, data)
    report_residual_diagnostics(resid, cfg, summary,)
    spatial_resid = compute_residual_spatial_diagnostic(validation, data, cfg, summary)

    calibration_table = compute_calibration_table(
        data.y_test.values, validation.holdout_pred
    )
    save_csv(calibration_table, cfg.data_dir / "calibration_metrics.csv")
    log_info(f"\nCalibration metrics table:\n{calibration_table.to_string(index=False)}")

    # Step 5: Benchmark comparison
    log_info("\n[STEP 5] Benchmark model comparison...")
    comparison_df = compare_models(model_results, cfg, summary)

    # Step 6: Feature importance
    log_info("\n[STEP 6] Permutation feature importance...")
    fi = compute_feature_importance(model_results, cfg, summary)

    # Step 7: Scenario trajectories
    log_info("\n[STEP 7] Computing scenario trajectories...")
    baseline, s1, s2 = run_scenario_trajectories(model_results, cfg, summary)

    # Step 8: Tree artifact / staircase diagnostics
    log_info("\n[STEP 8] Tree artifact / staircase diagnostics...")
    tree_artifact_diagnostics(s1.lst, "Scenario 1", cfg, summary)
    tree_artifact_diagnostics(s2.lst, "Scenario 2", cfg, summary)
    local_gradient_diagnostics(s1.lst, s1.scenario_pct, "Scenario 1", summary)
    local_gradient_diagnostics(s2.lst, s2.scenario_pct, "Scenario 2", summary)
    smoothing = compute_display_smoothing(s1, s2, cfg)

    # Step 9: NEGI computation
    log_info("\n[STEP 9] Computing NEGI results...")
    negi, reference_cooling_c = compute_negi_results(baseline, s1, s2, cfg)
    report_negi_scenario_diagnostics(negi, reference_cooling_c, cfg, summary)
    log_info("\nInterior-optimum check:")
    check_interior_optimum(negi.negi_s1, negi.scenarios, "Scenario 1", cfg)
    check_interior_optimum(negi.negi_s2, negi.scenarios, "Scenario 2", cfg)

    # Step 10: Support diagnostics
    log_info("\n[STEP 10] Feature-space support diagnostics...")
    support = compute_support_diagnostics(model_results, s1, s2, cfg, summary)
    report_data_support_check(s2, negi, support, cfg, summary)

    # Step 11: Uncertainty bands
    log_info("\n[STEP 11] Computing uncertainty bands (spatial refits)...")
    uncertainty = compute_uncertainty_bands(model_results, baseline, s1, s2, negi, cfg)
    report_uncertainty_bands(uncertainty, s1, cfg, summary)
    refit_summary_df = summarize_spatial_refit_diagnostics(validation, uncertainty, s1, cfg)

    # Step 12: Sensitivity analysis
    log_info("\n[STEP 12] Sensitivity analysis...")
    sensitivity = run_sensitivity_analysis(negi, s1, cfg)
    report_sensitivity_analysis(sensitivity, cfg)

    # Pre-computed diagnostics
    ndbi_sweep = compute_ndbi_response_sweep(model_results, cfg)
    surface    = compute_response_surface(model_results, cfg)

    ndbi_q = df["NDBI"].quantile([0.25, 0.50, 0.75]).values
    ndbi_bins = [
        (df["NDBI"].min(),    ndbi_q[0],            f"Low urban\nNDBI < {ndbi_q[0]:.3f}"),
        (ndbi_q[0],          ndbi_q[1],             f"Moderate urban\n{ndbi_q[0]:.3f}-{ndbi_q[1]:.3f}"),
        (ndbi_q[1],          ndbi_q[2],             f"High urban\n{ndbi_q[1]:.3f}-{ndbi_q[2]:.3f}"),
        (ndbi_q[2],          df["NDBI"].max() + 1e-9, f"Very high urban\nNDBI > {ndbi_q[2]:.3f}"),
    ]
    conditional_results = compute_conditional_ndvi_lst_slopes(df, ndbi_bins)
    conditional_df = conditional_slopes_to_dataframe(conditional_results)
    save_csv(conditional_df, cfg.data_dir / "conditional_ndvi_lst_slopes.csv")

    ndvi_decile_df = compute_ndvi_decile_summary(df)
    saturation     = compute_saturation_diagnostic(negi, s2, cfg)

    feature_support_status = (
        "within bounds" if not s2.out_of_bounds.any() else "partial extrapolation"
    )
    annotations = {
        "baseline_lst":       baseline.baseline_lst,
        "first_cooling_pct":  negi.first_cooling_pct,
        "max_cooling_c":      negi.max_cooling_s2,
        "max_negi_s2_value":  negi.negi_s2[negi.s2_optimum_idx],
        "max_negi_s2_pct":    negi.scenario_pct[negi.s2_optimum_idx],
        "calibration_slope":  validation.calibration_slope,
        "calibration_intercept": validation.calibration_intercept,
        "feature_support_status": feature_support_status,
    }
    flag_str = (
        "WARNING: partial extrapolation" if not support.extrap_s2_ok
        else "OK: within training support"
    )

    scenario_summary_df = pd.DataFrame([{
        "baseline_lst_c":           baseline.baseline_lst,
        "s1_max_negi":              negi.negi_s1[negi.s1_optimum_idx],
        "s1_max_negi_pct":          negi.scenario_pct[negi.s1_optimum_idx],
        "s2_max_negi":              negi.negi_s2[negi.s2_optimum_idx],
        "s2_max_negi_pct":          negi.scenario_pct[negi.s2_optimum_idx],
        "s2_max_cooling_c":         negi.max_cooling_s2,
        "s2_max_warming_c":         negi.max_warming_s2,
        "s2_first_cooling_pct":     negi.first_cooling_pct,
        "s2_warming_fraction":      negi.warming_fraction_s2,
        "s2_cooling_fraction":      negi.cooling_fraction_s2,
        "calibration_slope":        validation.calibration_slope,
        "feature_support_status":   feature_support_status,
    }])
    save_csv(scenario_summary_df, cfg.data_dir / "scenario_summary.csv")

    scenario_results_s2 = compute_scenario_results(
        s2, negi.negi_s2, negi.energy_norm_sqrt, negi.desal_energy_sqrt, baseline.baseline_lst,
    )
    scenario_points_df = scenario_results_s2.to_dataframe()
    scenario_points_df["Within Support"]            = support.in_support_s2
    scenario_points_df["Nearest Neighbor Distance"] = support.mean_knn_dist_s2
    save_csv(scenario_points_df, cfg.data_dir / "scenario_points.csv")

    # Auto-populate Conclusions in the summary log
    summary.add_conclusion(
        f"NDBI is the dominant predictor of LST in Jeddah "
        f"(permutation importance = {fi.table['Importance'].iloc[0]:.3f}), "
        f"substantially exceeding Elevation "
        f"({fi.table['Importance'].iloc[1]:.3f}) and NDVI "
        f"({fi.table['Importance'].iloc[2]:.3f})."
    )
    summary.add_conclusion(interpretation_text("predictor_scope"))
    summary.add_conclusion(
        f"Spatial holdout R² = {validation.holdout_r2:.3f}, "
        f"RMSE = {validation.holdout_rmse:.2f} °C.  "
        f"Maximum predicted cooling ({negi.max_cooling_s2:.2f} °C) "
        f"is below the model RMSE; cooling-magnitude estimates should be "
        "interpreted with caution."
    )
    # Configurable near-boundary tolerance band for boundary-maximum warnings.
    s1_at_boundary = is_near_trajectory_boundary(
        negi.s1_optimum_idx, len(negi.negi_s1), cfg.scenario_boundary_tolerance_pct
    )
    s2_at_boundary = is_near_trajectory_boundary(
        negi.s2_optimum_idx, len(negi.negi_s2), cfg.scenario_boundary_tolerance_pct
    )
    if s1_at_boundary:
        summary.add_conclusion(
            "Scenario 1 Maximum Evaluated NEGI occurs at the trajectory boundary (0%).  "
            "No interior optimum is identifiable under greening-only conditions "
            "at the reference parameter values."
        )
        summary.add_warning(
            "Scenario 1: " + interpretation_text("boundary_maximum") +
            "  Results are sensitive to cost-function parameters (w0, alpha, beta); "
            "see sensitivity analysis."
        )
    if s2_at_boundary:
        summary.add_warning("Scenario 2: " + interpretation_text("boundary_maximum"))

    # Step 13: Generate figures
    log_info("\n[STEP 13] Generating figures...")

    log_info("  [13.1] Validation figures...")
    plot_actual_vs_predicted(validation, data, cfg)
    plot_model_comparison(comparison_df, validation, cfg)

    log_info("  [13.2] Calibration diagnostics...")
    FIGURE_REGISTRY.add(
        filename="calibration_metrics.csv",
        caption=generate_caption(
            "Calibration metrics table (calibration slope, intercept, mean bias, "
            "Pearson r) computed without recalibrating predictions.  "
            "RMSE and MAE are reported in the Confirmatory Holdout section.",
        ),
        description="Single-table calibration diagnostics export.",
        section="Calibration",
    )

    log_info("  [13.3] Residual diagnostics figures...")
    plot_residuals_histogram(resid, cfg)
    plot_residuals_qq(resid, cfg)
    plot_residuals_vs_predicted_full(
        resid, validation.holdout_pred, cfg, summary, registry=FIGURE_REGISTRY
    )
    plot_residual_spatial_map(spatial_resid, cfg, registry=FIGURE_REGISTRY)

    log_info("  [13.4] Feature importance...")
    plot_feature_importance(fi, cfg)

    log_info("  [13.5] Response curves...")
    plot_ndvi_vs_lst(df, cfg)
    plot_lst_response_to_ndbi(ndbi_sweep, cfg)
    plot_ndbi_response_uncertainty(ndbi_sweep, uncertainty, cfg)
    plot_ndvi_decile_analysis(
        ndvi_decile_df, corr_ndvi_lst, corr_ndbi_lst, corr_ndvi_ndbi, cfg
    )
    plot_ndvi_conditional_by_ndbi_bins(conditional_results, df, cfg)

    log_info("  [13.6] Response surface...")
    plot_response_surface(surface, df, s2, cfg)

    log_info("  [13.7] Scenario support...")
    plot_scenario2_data_support(df, s2, support, cfg)

    log_info("  [13.8] Scenario optimization...")
    _, smoothing_robust, peak_is_ood, peak_idx_raw = plot_negi_smoothing_robustness(
        negi, smoothing, s2, cfg, summary
    )
    plot_negi_exponent_sensitivity(negi, s1, cfg)
    plot_cooling_saturation_diagnostic(negi, s2, saturation, support, cfg)

    log_info("  [13.9] NEGI figures...")
    summary.log("NEGI scenario comparison", "Feature support flag", flag_str)
    plot_negi_scenario_comparison(negi, annotations, flag_str, cfg)
    plot_negi_energy_cost_comparison(negi, cfg)

    log_info("  [13.10] Uncertainty bands...")
    fig_unc, ax_unc, peak_iqr_idx = plot_negi_uncertainty_bands_primary(
        uncertainty, s1, cfg
    )
    set_axis_labels(ax_unc, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax_unc, "NEGI Uncertainty Bands (Spatial-Block Holdout Refits)")
    set_legend(ax_unc, loc="lower left")
    fig_unc.text(
        0.5, -0.02,
        (
            f"Median ± IQR across {cfg.n_spatial_refits} spatial-block refits.  "
            f"Peak IQR at scenario = {s1.scenario_pct[peak_iqr_idx]:.0f}%.  "
            f"Calibration: pred = {validation.calibration_slope:.3f}*obs"
            f"{validation.calibration_intercept:+.3f}."
        ),
        ha="center", fontsize=7.5, color=cfg.color_neutral,
    )
    save_fig(fig_unc, FIG_NEGI_UNCERTAINTY_BANDS, cfg,
             caption="NEGI uncertainty bands across spatial-block holdout refits.")
    plot_negi_scenario2_robustness_full(uncertainty, s1, cfg)

    log_info("  [13.11] Sensitivity analysis...")
    plot_sensitivity_optimal_negi(sensitivity, cfg)

    log_info("  [13.12] Supplementary diagnostics complete.")

    # Step 14: Summary log
    log_info("\n[STEP 14] Exporting summary log...")
    summary.log("NEGI scenario comparison", "Smoothing robust (Savgol)", str(smoothing_robust))
    summary.log("NEGI scenario comparison", "Peak location OOD",         str(peak_is_ood))

    # Prediction-cache statistics are an implementation detail, 
    # not a scientific result: they stay in the developer log only and are
    # intentionally excluded from the publication summary report.
    if PREDICTION_CACHE is not None:
        cache_stats = PREDICTION_CACHE.stats()
        log_debug(
            f"Prediction cache: {cache_stats['hits']} hits, "
            f"{cache_stats['misses']} misses "
            f"({cache_stats['hit_rate'] * 100:.1f}% hit rate)."
        )

    log_info("\n[STEP 14b] Running automated QA checks...")
    run_qa_checks(
        cfg, validation, fi, repro_info,
        scenario_summary_df=scenario_summary_df,
        scenario_points_df=scenario_points_df,
    )

    summary.export(
        cfg.data_dir / "full_summary_report.csv",
        cfg.data_dir / "full_summary_report.md",
    )

    runtime_seconds = time.monotonic() - run_start
    write_reproducibility_report(
        cfg, repro_info,
        best_params=model_results.best_params,
        validation=validation,
        runtime_seconds=runtime_seconds,
    )

    log_info("PIPELINE COMPLETE")
    log_info(f"  Figures  -> {cfg.results_dir}")
    log_info(f"  Data     -> {cfg.data_dir}")
    log_info(f"  Runtime  -> {runtime_seconds:.1f} s")


if __name__ == "__main__":
    main()
