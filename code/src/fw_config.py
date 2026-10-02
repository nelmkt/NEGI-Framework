"""Analysis settings. Revision sensitivities are exploratory, not retrospectively prespecified."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    panel_path: Path = Path("code/gee/panel.csv")
    out_dir: Path = Path(".")
    seed: int = 42

    # periods (summer composites, May–September; see code/gee/export_panel.py)
    pre_years: tuple = (2014, 2015)
    mid_years: tuple = (2018, 2019)
    post_years: tuple = (2024, 2025)

    # setting before the greening: share of the ground within 500 m covered by buildings in 2015 (GHSL)
    builtup_min: float = 0.05                # below: outside the built-up area; at or above: built-up surroundings

    # Legacy pipeline `main` also adjusts for concurrent neighborhood development.
    # The manuscript leads with pre-treatment-only strata and labels that output a sensitivity.
    region_deg: float = 0.10                 # local region, ~10 km: sea breeze, humidity, terrain
    coast_bins_km: tuple = (2.0, 5.0, 10.0)  # distance to the coast
    own_ghsl_bins: tuple = (0.05,)           # the cell's own building cover, 2015
    nb_change_bins: tuple = (0.02, 0.10)     # surroundings: change in Dynamic World built-up probability, 2015 → 2024–25
    emis_bins: tuple = (0.96,)               # the fixed emissivity of the ground (ASTER-based): its material
    min_controls: int = 3                    # a stratum needs this many control cells to be used
    stable_surface_max: float = 0.20         # control and ring cells: own built-up probability changed by less than this

    # dose: greened 30 m pixels out of the cell's 9
    dose_bins: tuple = ((1, 2), (3, 4), (5, 8), (9, 9))
    min_cells_reported: int = 10

    # inference
    block_deg: float = 0.05                  # spatial block for the bootstrap, ~5 km
    n_boot: int = 2000

    # sensitivity
    far_control_m: float = 600.0             # stricter control distance from any greened pixel
    arc_lat_bounds: tuple = (21.3, 21.4)     # exploratory geographic proxy, not a mapped intervention boundary

    # emissivity sensitivity: Landsat band 10 effective wavelength (µm) and second radiation constant (µm K)
    lambda_b10_um: float = 10.9
    c2_umK: float = 14388.0

    # machine-learning model of summer LST across the city (XGBoost), fitted on the 2024–25 cross-section
    sample_fraction: float = 0.10            # cells with rnd < this form a representative sample of the study area
    ml_features: tuple = ("ndvi", "ndbi", "elev", "emis", "coast_km", "lon", "lat")
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 5, "reg_lambda": 1.0})
    cv_folds: int = 5                        # spatial cross-validation: whole 0.05° blocks held out
    n_refits: int = 200                      # joint spatial refits; smaller runs cannot license candidates
    support_k: int = 5                       # feature-space support: k nearest real cells
    support_percentile: float = 95.0         # outside the data = farther than 95% of real cells are from theirs
    strict_exclusion_m: float = 300.0        # stricter test: no training cell within this distance of greening
    validation_margin_C: float | None = None  # decision-relevant tolerance must be supplied; no default pass
    min_validation_cells: int = 10
    min_validation_blocks: int = 5
    min_validation_refits: int = 200
    jobs: int = 4
    region_holdouts: bool = True

    def __post_init__(self):
        if self.n_boot < 2 or self.n_refits < 2 or self.n_refits > self.n_boot:
            raise ValueError("Require 2 <= refits <= bootstrap draws.")
        if self.validation_margin_C is not None and (not 0 < self.validation_margin_C < float('inf')):
            raise ValueError("Validation tolerance must be finite and positive.")

    # Optional allocation of externally supplied, comparable site/intervention options.
    # Resource quantities must all cover the same declared planning period.
    decision_maker: str | None = None  # optional allocation input; no decision-maker is inferred from the panel
    allocation_path: Path | None = None
    water_budget_m3: float | None = None
    wastewater_cap_m3: float | None = None
    energy_budget_mwh: float | None = None
    financial_budget: float | None = None

    # water and energy ledger: (low, central, high)
    kc: tuple = (0.50, 0.60, 0.85)
    efficiency: tuple = (0.90, 0.75, 0.60)
    kwh_per_m3: dict = field(default_factory=lambda: {
        "desalinated seawater (SWRO)": (2.5, 3.25, 4.0),
        "treated wastewater": (0.30, 0.60, 0.93),
    })
    sio_path: Path = HERE / "sio_irrigation_2020_2022_v8.csv"  # rebuilt from unedited 6.xlsx and 8.xlsx


SETTINGS = ("outside the built-up area", "built-up surroundings")

SOURCES = {
    "kc": "Landscape coefficient for warm-season turf 0.6 as used in landscape water budgets (WUCOLS IV, University "
          "of California Cooperative Extension, 2014); 0.5 for mixed moderate-water plantings; 0.85 is the FAO-56 "
          "value for unstressed warm-season turf (Allen et al. 1998, Table 12).",
    "efficiency": "Irrigation efficiency 0.75 for overhead spray and 0.9 for drip; 0.6 for poorly maintained systems "
                  "(California Model Water Efficient Landscape Ordinance, 2015).",
    "et0": "TerraClimate reference evapotranspiration (Penman–Monteith, Abatzoglou et al. 2018), annual total, mean "
           "over the greened land (code/gee/export_panel.py).",
    "swro": "Seawater reverse osmosis, whole plant: 2.5–4.0 kWh/m³ (Voutchkov 2018, Desalination 431, 2–14).",
    "tse": "Municipal wastewater treatment with reuse-grade polishing: 0.30–0.93 kWh/m³ (Plappally & Lienhard 2012, "
           "Renewable and Sustainable Energy Reviews 16, 4818–4848).",
    "sio": "Saudi Irrigation Organization open data, https://sio.gov.sa/OpenData/Primery_Data_en (irrigated area and "
           "water supplied by source, 2020–2022).",
    "ghsl": "GHSL built-up surface 2015 (JRC/GHSL/P2023A/GHS_BUILT_S, Pesaresi & Politis 2023).",
}
