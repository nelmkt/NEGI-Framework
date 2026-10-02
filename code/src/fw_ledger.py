"""Water and energy of keeping one hectare green, set beside the cooling measured on it.

The two sides stay in their own units (°C of surface cooling; m³ of water; kWh of electricity). No conversion of a
surface-temperature change into energy is attempted: nothing in the data measures one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fw_config import Config

LEVELS = ("low", "central", "high")


def reference_et(meta: dict) -> tuple[float, str]:
    v = {k: x for k, x in (meta.get("reference_et_mm_per_year") or {}).items() if x}
    if not v:
        raise SystemExit("the panel's _meta.json has no reference_et_mm_per_year (rerun code/gee/export_panel.py)")
    years = sorted(v)
    return float(np.mean([v[k] for k in years])), ", ".join(years)


def water(cfg: Config, et0_mm: float) -> pd.DataFrame:
    """Irrigation depth = ET0 × Kc / efficiency. Low pairs the low Kc with the high efficiency, and so on."""
    rows = []
    for i, lev in enumerate(LEVELS):
        depth_mm = et0_mm * cfg.kc[i] / cfg.efficiency[i]
        rows.append({"level": lev, "kc": cfg.kc[i], "efficiency": cfg.efficiency[i], "depth_m": depth_mm / 1000,
                     "m3_per_ha_yr": depth_mm * 10})
    return pd.DataFrame(rows)


def energy(cfg: Config, W: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, (lo, mid, hi) in cfg.kwh_per_m3.items():
        rows.append({"source": src, "kwh_per_m3": f"{lo:g}–{hi:g}",
                     "MWh_per_ha_yr_low": W.loc[0, "m3_per_ha_yr"] * lo / 1000,
                     "MWh_per_ha_yr_central": W.loc[1, "m3_per_ha_yr"] * mid / 1000,
                     "MWh_per_ha_yr_high": W.loc[2, "m3_per_ha_yr"] * hi / 1000})
    return pd.DataFrame(rows)


def sio_depths(cfg: Config) -> pd.DataFrame:
    """Water the Saudi Irrigation Organization supplied per irrigated hectare, by branch and year."""
    s = pd.read_csv(cfg.sio_path)
    vol = s[["reclaimed_m3", "groundwater_m3", "agri_drainage_m3"]].fillna(0).sum(axis=1)
    s["supplied_m3"] = vol
    s["reclaimed_share"] = s["reclaimed_m3"].fillna(0) / vol
    s["depth_m"] = vol / (s["irrigated_area_ha"] * 1e4)
    return s[["year", "branch", "irrigated_area_ha", "supplied_m3", "reclaimed_share", "depth_m"]]


def run(cfg: Config, meta: dict) -> dict:
    et0, et0_years = reference_et(meta)
    W = water(cfg, et0)
    E = energy(cfg, W)
    S = sio_depths(cfg)
    q = S["depth_m"].quantile([0.25, 0.5, 0.75]).to_dict()
    y22 = S[S.year == 2022]
    total = None
    if meta.get("greened_area_km2"):
        ha = meta["greened_area_km2"] * 100
        total = {"area_ha": ha, "water_Mm3_yr": {lev: ha * W.loc[i, "m3_per_ha_yr"] / 1e6 for i, lev in enumerate(LEVELS)}}
    return {"et0_mm": et0, "et0_years": et0_years, "water": W, "energy": E, "sio": S,
            "sio_quartiles_m": {"q25": q[0.25], "median": q[0.5], "q75": q[0.75]},
            "sio_reclaimed_share_2022": float((y22["supplied_m3"] * y22["reclaimed_share"]).sum() / y22["supplied_m3"].sum()),
            "total": total}
