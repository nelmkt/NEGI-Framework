"""Optional binary portfolio selection from externally supplied, comparable interventions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

COLUMNS=["option_id","site_id","intervention","cooling_C","benefit_weight","water_m3","wastewater_m3",
         "energy_mwh","cost","period","benefit_metric","currency","evidence_source"]

def allocate(cfg):
    objective="maximize sum of supplied cooling reduction times beneficiary weight; at most one option per site"
    if cfg.allocation_path is None:
        return dict(status="not_evaluated_missing_inputs",objective=objective,selected=pd.DataFrame(columns=COLUMNS),
                    reason="Comparable interventions, beneficiary weights, evidence and four resource limits are required.")
    budgets=[cfg.water_budget_m3,cfg.wastewater_cap_m3,cfg.energy_budget_mwh,cfg.financial_budget]
    if any(x is None or not np.isfinite(x) or x<0 for x in budgets):
        raise ValueError("Allocation needs nonnegative finite water, wastewater, energy and financial limits.")
    d=pd.read_csv(cfg.allocation_path)
    missing=set(COLUMNS)-set(d.columns)
    if missing or not len(d):
        raise ValueError(f"Allocation requires nonempty options and columns {sorted(missing)}")
    if d[COLUMNS].isna().any().any() or d.option_id.duplicated().any():
        raise ValueError("Allocation fields must be complete and option IDs unique.")
    for col in ["option_id","site_id","intervention","period","benefit_metric","currency","evidence_source"]:
        if d[col].astype(str).str.strip().eq("").any(): raise ValueError(f"Empty allocation {col}")
    for col in ["period","benefit_metric","currency"]:
        if d[col].nunique()!=1: raise ValueError(f"All options must share {col}.")
    nums=["cooling_C","benefit_weight","water_m3","wastewater_m3","energy_mwh","cost"]
    a=d[nums].to_numpy(float)
    if not np.isfinite(a).all() or (a<0).any() or (d.wastewater_m3>d.water_m3).any():
        raise ValueError("Resource/benefit inputs must be finite and nonnegative; wastewater cannot exceed total water.")
    benefit=d.cooling_C.to_numpy(float)*d.benefit_weight.to_numpy(float)
    resource=d[["water_m3","wastewater_m3","energy_mwh","cost"]].to_numpy(float).T
    sites=np.vstack([(d.site_id==site).to_numpy(float) for site in d.site_id.unique()])
    matrix=np.vstack([resource,sites]); cap=np.r_[budgets,np.ones(len(sites))]
    result=milp(c=-benefit,integrality=np.ones(len(d)),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,np.full(len(cap),-np.inf),cap),
                options={"time_limit":60})
    if not result.success:
        raise RuntimeError(f"No verified optimal portfolio: {result.message}")
    chosen=result.x>.5
    selected=d[chosen].copy()
    return dict(status="optimal_for_supplied_inputs",objective=objective,selected=selected,
                benefit=float(benefit@chosen),resource_totals=dict(zip(["water_m3","wastewater_m3","energy_mwh","cost"],resource@chosen)),
                limits=dict(zip(["water_m3","wastewater_m3","energy_mwh","cost"],budgets)),
                period=str(d.period.iloc[0]),benefit_metric=str(d.benefit_metric.iloc[0]),currency=str(d.currency.iloc[0]),
                caveat="Input evidence is supplied, not independently validated. Require disjoint benefit footprints or adjusted weights; no inferred health benefit.")

