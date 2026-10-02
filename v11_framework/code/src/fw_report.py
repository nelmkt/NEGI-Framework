"""Generate a claim-limited report from the current run; no narrative assertions of automatic validation."""
from __future__ import annotations
import hashlib
import platform
import numpy as np
import pandas as pd
from fw_config import SETTINGS


def table(data, columns=None):
    if isinstance(data,list): data=pd.DataFrame(data)
    if not len(data): return "_No rows available._"
    if columns is not None: data=data[[c for c in columns if c in data]]
    def cell(v):
        if isinstance(v,(float,np.floating)): return f"{v:.3f}" if np.isfinite(v) else "not available"
        return str(v).replace("|","/").replace("\n"," ")
    return "\n".join(["| "+" | ".join(data.columns)+" |","|"+"---|"*len(data.columns)]+
                      ["| "+" | ".join(cell(v) for v in row)+" |" for row in data.itertuples(index=False,name=None)])


def write_report(res,cfg,path):
    R,V,L,D=res["measured"],res["test"],res["ledger"],res["decision"]
    logic_tables = path.parent.parent / "tables"
    primary_head = (pd.read_csv(logic_tables / "logic_primary_class_estimates.csv")
                    if (logic_tables / "logic_primary_class_estimates.csv").exists() else pd.DataFrame())
    primary_head = primary_head[(primary_head.setting == SETTINGS[0]) &
                                (primary_head.specification == "pre-treatment-only strata") &
                                (primary_head.dose_class == "all")] if len(primary_head) else primary_head
    rows=[]
    for s in SETTINGS:
        for name in ["per_pixel","all_greened","placebo_per_pixel","ring"]:
            rows.append(dict(setting=s,estimate=name,**R[s]["estimates"][name]))
    counts=res["meta"].get("scenes_per_year_month",{})
    missing=[key for key,value in counts.items() if value==0]
    coverage="No month-count metadata supplied." if not counts else ("Missing scene months: "+", ".join(missing) if missing else "All requested year/month combinations have scenes at collection level; pixel-level coverage can still vary.")
    quick=cfg.n_boot<2000 or cfg.n_refits<200
    pieces=[
        "# Urban greening in Jeddah: measured surface change and illustrative resource accounting",
        "**Synthetic test panel**" if res["meta"].get("synthetic") else "Observed Landsat-derived panel; interpretation conditional on the design assumptions below.",
        "**Run status:** "+("reduced diagnostic run; do not use as final inferential output." if quick else "configured for 2,000 or more measured draws and 200 or more joint model refits; consult the provenance below for which outputs were recomputed in this layout.")+" "+res["meta"].get("analysis_provenance","See manuscript/RUN_PROVENANCE.md for which outputs were recomputed in this package layout.")+" Current pre-treatment-only classes, block deletions, exported-cell proximity proxy and SIO arithmetic were recomputed from bundled files.",
        "## Decision scope",
        "An optional allocation tool can maximize supplied cooling reduction times beneficiary weights under water, usable wastewater, electricity and financial limits, selecting at most one intervention per site. It requires externally supplied, comparable options and constraints; the observed panel alone does not provide them.",
        "**This run does not establish the best municipal investment.** Observed NDVI transitions do not establish managed planting or applied irrigation. Much outside-built-up vegetation was described in the source report as wadi vegetation; runoff or discharged wastewater may support it. The irrigation ledger is a hypothetical resource scenario, not observed treatment expenditure.",
        "The pre-treatment-only matching row is the interpretation-leading specification because concurrent neighbourhood built-up change may be affected by a greening project or shared development. The pipeline's `main` label denotes the earlier concurrent-change computation; its figures and Model A/B support screens are sensitivity results on that narrower population, not screens for the added pre-treatment-only cells. Neither match resolves stable-control selection or conditional parallel trends.",
        "## What was measured",
        "Matched changes in summer land-surface temperature (LST), 2014–15 to 2024–25. Negative values mean cooler surfaces relative to matched controls. Effects concern 90 m cells; 30 m greened-pixel counts are optical-area measures, not independent 30 m thermal measurements. The per-pixel statistic is a zero-intercept slope with dose-squared leverage, not a causal marginal return to another planted pixel.",
        "The panel calls a 30 m pixel greened when summer NDVI is <0.15 in both 2014 and 2015 and >=0.30 in both 2024 and 2025. Landsat 8 Collection 2 Level-2 scenes require CLOUD_COVER <20%; cloud, cirrus, shadow, dilated cloud and saturated observations are masked. Each available May-September month is median-composited, then the available monthly medians are averaged by year. No May 2014 scene appears in the source metadata, so the baseline comparison remains unresolved until a common-month analysis is run. Per-pixel availability can differ; no irrigation or planting record verifies these transitions.",
        "The leading pre-treatment-only outside estimate is shown first; the following full measured table is the earlier concurrent-change sensitivity population.",
        table(primary_head, ["setting", "specification", "dose_class", "n_matched", "n_treated_blocks", "per_pixel_C", "per_pixel_lo_C", "per_pixel_hi_C"]),
        table(rows,["setting","estimate","estimate_C","lo_C","hi_C","n_treated_matched","n_boot_valid"]),
        "Intervals use full-panel spatial blocks. The late-greener placebo covers 289/744 concurrent-change matched outside cells (38.8%), and only through 2018–19. It does not test the other 455 or later untreated trends. Exact planting dates, irrigation histories and canopy maturation are not observed.",
        "## Geographic and matching sensitivity",
        table(pd.concat([R[s]["sensitivity"].assign(setting=s) for s in SETTINGS]),["setting","variant","estimate_C","lo_C","hi_C","n_treated_matched"]),
        "Southern-belt exclusion uses 21.3 <= latitude < 21.4 degrees N as an **exploratory proxy**, not a mapped project boundary. Full leave-one-region-out results and dose-specific exclusions are in the tables. Baseline-only strata omit neighborhood development change while retaining other matching variables. Stable-surface control selection remains a separate assumption.",
        table(pd.concat([R[s]["cluster_diagnostics"].assign(setting=s) for s in SETTINGS]).query("estimand == 'per greened pixel'"),
              ["setting","variant","n_treated_matched","n_treated_blocks_matched","n_treated_regions_matched","largest_block_dose_squared_share"]),
        "Block counts and leverage shares are diagnostics, not independent sample sizes. Main matching conditions on neighborhood built-up change during treatment; excluding greened pixels does not prove that covariate is unaffected by associated development. Residual balance is reported in measured_balance.csv.",
        "## Model effect diagnostics",
        f"Spatial absolute-LST cross-validation: R²={res['model']['r2']:.3f}, RMSE={res['model']['rmse_C']:.3f} degrees C. This does not validate intervention effects. A includes surrounding non-greened training cells; B excludes training within {cfg.strict_exclusion_m:g} m of greening. Neither changes the model's counterfactual definition.",
        "The model contrast substitutes pre-period NDVI/NDBI into a post-period cross-sectional model. Its comparability with matched DiD requires assumptions about untreated surface evolution and confounding. Geographic model failure alone does not refute the matched estimate.",
        table(pd.concat([V["by_setting"],V["strict"]]),["setting","test","n_cells","supported_share","measured_C","model_C","difference_C","difference_lo_C","difference_hi_C"]),
        "The model and measured comparison are recomputed within the same spatial draw, including model training, matched controls, and the evaluated treated-cell mix. Joint intervals use the configured model-refit count; measured-only headline intervals use all measured draws. Missing joint draws remain aligned and are counted.",
        "**Practical error tolerance:** "+(f"{cfg.validation_margin_C:g} degrees C, supplied by the caller. This revision is exploratory; it is not retrospectively prespecified." if cfg.validation_margin_C is not None else "not supplied. No model setting is labelled sufficiently accurate and no candidate predictions are released through the gate."),
        "An eligible supported-domain diagnostic requires the entire 95% model-minus-measured interval inside the stated tolerance, at least 10 supported cells, 5 treated blocks, and 200 valid joint refits (and at least 95% valid draws). These are screening requirements, not proof of causal or site-level validity. Support screening is fixed from the primary fit, so these intervals are conditional on that domain.",
        table(V["supported"],["setting","n_cells","n_blocks","supported_share","difference_C","difference_lo_C","difference_hi_C","n_joint_valid","status"]),
        "Candidate exploration additionally requires the supported dose to pass, both baseline and changed features within training support, and resemblance to tested before/after states. Targets use a paired transition from one real observed cell within the dose, rather than unrelated marginal medians. All refits contribute to candidate intervals; no robust cooling is flagged explicitly. These transitions still require a real planting-to-surface-change model before planning use.",
        f"Candidate diagnostic rows released: {len(D['candidates'])}. Regional held-out diagnostics: {len(V['region_holdouts'])} rows. Whole ~10 km regions plus a 300 m bounding buffer are excluded from training. Regional discrepancies are point diagnostics, not equivalence tests, and do not by themselves license local rankings.",
        table(V["region_holdouts"],["setting","region","n_cells","measured_C","model_C","difference_C","supported_share"]),
        "## Conditional water and electricity accounting",
        table(L["water"]),table(L["energy"]),
        "Water is reference evapotranspiration times landscape coefficient divided by efficiency. Its low/central/high cases are scenarios, not a fitted probability distribution. Delivered pumping, marginal source availability and installation/maintenance costs are not inferred from the satellite panel. The SIO branch-year CSV is an agricultural context series; its source/site-to-branch join cannot be verified without the two original workbooks and it is not Jeddah's available supply.",
        "Water per degree refers to annual water for an assumed irrigated area divided by summer cell-average LST reduction. Because irrigation depth is shared, it algebraically inverts cooling per greened area; it is not a separate resource ranking, heat removed, a health benefit, financial cost-effectiveness, or a water-to-cooling dose-response.",
        table(D["summary"],["setting","basis","cells","cooling_C","m3_per_degree","m3_per_degree_low","m3_per_degree_high","scenario_low","scenario_high"]),
        "Central ratio intervals jointly resample matched cooling and mean greened area. Scenario-low/high additionally combine low/high irrigation with the ratio limits; they are **scenario envelopes, not 95% confidence intervals**. If any finite bootstrap draw has nonpositive cooling benefit, finite ratio bounds are withheld. Full cross-products of irrigation case, source and source-energy intensity are in decision_scenarios.csv.",
        "## Optional allocation with externally supplied inputs",
        "The previously computed rank frequencies remain in tables/decision_ranking.csv for traceability only. Under a common irrigation depth they re-express the measured cooling-per-area ordering and add no independent decision evidence. Option-specific irrigation, beneficiary effects and constraints would be needed for a resource trade-off.",
        "**Allocation status:** "+D["allocation"]["status"],
        D["allocation"].get("reason",D["allocation"].get("caveat","")),
        "The optional allocation table accepts greening, shade, roofs or other interventions only when supplied on one comparable benefit scale and planning period. Nonnegative cooling reductions and beneficiary weights are required; water-source variants can be separate options at the same site. Total water, reclaimed-water availability, electricity and money are hard limits. Disjoint benefit footprints or overlap-adjusted weights are necessary. Choosing nothing is the explicit feasible baseline.",
        "## Measurement limitations and remaining data work",
        "- Summer LST near the ~10:50 satellite overpass is not air temperature, night cooling, winter conditions or building demand. Population-weighted LST is a proxy unless calibrated to the intended benefit.",
        "- "+coverage.rstrip(".")+". The existing annual composites cannot reconstruct missing monthly data, and the baseline comparison remains unresolved. The export tool accepts --months 6 7 8 9 for a separate common-month sensitivity; this satellite re-export has not been performed. The separate monthly NDVI and 30 m isolation scripts are prepared but have not produced results.",
        "- Unchanged ST_EMIS is not proof of unbiased retrieval or conservatively estimated cooling. USGS documents vegetation-adjustment and blockiness issues. A conditional emissivity sensitivity is not an independent LST validation. Corrected or independent thermal retrieval remains necessary for that claim. [USGS known issues](https://www.usgs.gov/landsat-missions/landsat-collection-2-known-issues).",
        "- Verify intervention type and water regime; obtain planting dates, meaningful beneficiary weights, comparable alternative interventions, and seasonal supply/cost data. No such data have been invented.",
        "- The code can test practical agreement once an externally justified tolerance is supplied. A nonsignificant difference alone is not equivalence. [Equivalence testing](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/).",
        "## Reproduction",
        f"Panel SHA-256: {hashlib.sha256(cfg.panel_path.read_bytes()).hexdigest()}. Python {platform.python_version()}. Bootstrap seed {cfg.seed}; {cfg.n_boot} measured draws; {cfg.n_refits} joint model refits. Run settings are saved in tables/results.json.",
        str(res["meta"].get("description",res["meta"].get("source",""))),
        "Run from the package directory: python code/src/run_framework.py --panel code/gee/panel.csv --out . See manuscript/README.md and manuscript/WORK_STATUS.md for implemented changes and work needing external data."
    ]
    if (logic_tables / "logic_primary_class_estimates.csv").exists():
        primary = pd.read_csv(logic_tables / "logic_primary_class_estimates.csv")
        blocks = pd.read_csv(logic_tables / "logic_block_summary.csv")
        proxy = pd.read_csv(logic_tables / "logic_panel_proximity_proxy.csv")
        sio = pd.read_csv(logic_tables / "logic_sio_integrity.csv")
        pieces += ["## Current saved-panel logic sensitivities",
                   "The interpretation-leading pre-treatment-only estimates avoid conditioning on concurrent neighbourhood development. Figure 2 is recomputed for this population; saved Model A/B diagnostics and Figures 3b/c and 5 still describe the narrower concurrent-change matched population.",
                   table(primary, ["setting", "specification", "dose_class", "n_matched", "n_treated_blocks", "per_pixel_C", "per_pixel_lo_C", "per_pixel_hi_C"]),
                   "Single-block deletions omit both treated and comparison cells in that block and rematch. Jackknife intervals are exploratory with uneven blocks; leverage-equivalent blocks describe concentration, not the effective number of independent clusters.",
                   table(blocks, ["setting", "specification", "dose_class", "n_deleted_blocks", "deletion_min_C", "deletion_max_C", "jackknife_95_lo_C", "jackknife_95_hi_C", "largest_block_dose_squared_share"]),
                   "The proximity screen sees only other exported eligible greened 90 m cells. It cannot certify a 300 m absence of all greened 30 m pixels or distinguish blur from site selection.",
                   table(proxy, ["setting", "specification", "nearest_other_exported_greened_cell_centre_gt_m", "n_matched", "n_treated_blocks", "per_pixel_C", "lo_C", "hi_C", "difference_from_full_per_pixel_C", "difference_lo_C", "difference_hi_C"]),
                   f"SIO aggregated branch-year depths: {len(sio)} rows, {int(sio.inside_assumed_depth_band.sum())} inside the assumed depth band. The two source workbooks and their site-to-branch join are not bundled, so their join remains unverified. Monthly NDVI persistence cannot be reconstructed from the annual panel; the June–September re-export was left pending at the user's request."]
        if (logic_tables / "logic_primary_figure2_estimates.csv").exists():
            fig2 = pd.read_csv(logic_tables / "logic_primary_figure2_estimates.csv")
            late_n = int(fig2[(fig2.setting == SETTINGS[0]) & (fig2.estimate == "placebo_per_pixel")].iloc[0].n_treated_matched)
            total_n = int(fig2[(fig2.setting == SETTINGS[0]) & (fig2.estimate == "per_pixel")].iloc[0].n_treated_matched)
            pieces += [f"Figure 2's pre-treatment-only placebo applies only to late greeners and ends in 2018–19. It covers {late_n}/{total_n} matched outside cells ({late_n/total_n:.1%}); the other {total_n-late_n} have no comparable tested pre-window.",
                       table(fig2[fig2.estimate == "placebo_per_pixel"],
                             ["setting", "estimate", "n_treated_matched", "estimate_C", "lo_C", "hi_C"])]
        if (logic_tables / "logic_primary_area_mixing.csv").exists():
            primary_mixing = pd.read_csv(logic_tables / "logic_primary_area_mixing.csv")
            primary_coast = pd.read_csv(logic_tables / "logic_primary_coastal_difference.csv")
            pieces += ["## Primary-population area-mixing and coastal diagnostics",
                       "These paired spatial-bootstrap comparisons use the pre-treatment-only population plotted in Figure 2. The area-mixing benchmark inherits any bias in its 9/9 reference. The 5–8-pixel departure excludes zero here, whereas its concurrent-change counterpart below does not. Neither departure identifies thermal spillover.",
                       table(primary_mixing), table(primary_coast),
                       "Coast contrasts include dose and site composition and are not causal gradients. Figure 5b uses the narrower concurrent-change population tabulated separately below."]
    pieces += followup_sections(res,cfg)
    path.write_text("\n\n".join(pieces)+"\n",encoding="utf-8")


def followup_sections(res,cfg):
    V=res["test"]
    p=["## Figure interpretation",
       "Figure 2: pre-treatment-only matched contrasts are plotted and tabulated in logic_primary_figure2_*.csv. Slope and placebo rows are degrees C per greened optical pixel; dose and ring rows are degrees C per 90 m cell. Figure 1 counts all classified greened cells. Figures 3b/c and 5 use the narrower concurrent-change population.",
       f"Figure 3b/c display {cfg.n_refits} joint refits saved with the analysis results; the provenance record states when they were run. With 200 refits, each nominal 2.5% tail contains about five draws, so interval endpoints are approximate. Regional holdouts are separate point diagnostics.",
       "Figure 5 assumes equal irrigation per hectare across settings. Panel a is an algebraic re-expression of concurrent-change matched cooling per greened area, not an independent resource comparison. Envelopes omit structural uncertainty. Filled and hollow model squares summarize supported and unsupported subsets; small filled circles in the setting colour show measured cooling on the same subset beside each square. Support is not accuracy, and these subsets differ from the all-cell target. Some supported subsets are below ten cells, so their squares are point diagnostics only. The outside all-cell crosses mostly extrapolate beyond the screened support domain in the observed panel; the 2-5 km measured estimate is withheld below the reporting minimum.",
       "The model learns responses to post-period NDVI/NDBI from non-greened cells. Transfer to newly irrigated, dense canopy is unverified. Model B's outside cell-level error grows across dose classes; coast-band errors change sign but are confounded by dose and site mix.",
       "The panel selects 2014–15, 2018–19 and 2024–25; other years were not exported. Endpoint-defined treatment mixes adoption timing and maturation. In the concurrent-change sensitivity, the late-greener placebo covers only 289 of 744 matched outside cells and only the first two windows; Figure 2's pre-treatment-only population has a separate placebo row.",
       "Hotter baseline treated cells can inflate apparent cooling if they would have cooled faster without greening. Higher baseline NDVI has ambiguous bias direction. Baseline imbalance alone does not establish either mechanism.",
       "SIO values are supply per irrigated agricultural hectare, not landscape consumption. MWh per degree in the decision tables is a gross scenario-implied supply requirement, not net energy saved; Figure 4 reports energy per hectare."]
    if "support_counts" in V:
        p+=["## Model B support by dose",table(V["support_counts"]),
            "Zero supported cells means no supported-subset discrepancy is estimable. Small cell/block counts do not demonstrate practical accuracy.",
            table(V["supported_by_dose"],["setting","pixels","n_cells","n_blocks","difference_C","difference_lo_C","difference_hi_C","n_joint_valid","status"])]
    if "followup" in res:
        F=res["followup"]
        p+=["## Concurrent-change paired mixing and coastal contrasts",table(F["area_mixing"]),
            "These rows describe the narrower population used by Figures 3b/c and 5. The mixing benchmark scales the observed 9/9 contrast by each class's resampled mean pixel count divided by nine. The paired difference uses the same spatial draw for both contrasts and mean area. It inherits bias in the 9/9 reference; it is not an independent physical model. If the 9/9 magnitude is inflated, the 1–2 benchmark becomes less negative and its observed excess grows. The 5–8 departure has an interval including zero in this population. Negative differences denote excess cooling relative to this benchmark. Separate interval overlap is not the test.",
            table(F["coastal_difference"]),
            "The coastal contrast is at least 10 km minus 5–10 km, retaining common-draw dependence. It includes dose/site composition differences and does not identify a causal coast-distance effect. Exclusion checks do not establish transportability of magnitude.",
            "## Matching specifications and retained populations",
            table(F["population_retention"]) if "population_retention" in F else "Retention diagnostics require run_followup.py.",
            table(F["coastal_block_counts"]) if "coastal_block_counts" in F else "Coastal block diagnostics require run_followup.py.",
            "Figure 1 counts all classified cells; the matched estimates use cells with at least three eligible stable-surface controls within their strata. The dose-retention table shows which cells were lost and how many fully greened cells remain in each caliper. The coastal table includes every matched cell, including bands withheld from effect plots below the reporting minimum, and gives spatial block counts.",
            table(F["matching_specifications"],["setting","specification","estimand","estimate_C","lo_C","hi_C","n_retained","n_controls","per_pixel_C","n_valid","inference"]),
            "Calipers restrict baseline LST, NDVI and elevation to 0.5 or 0.2 pooled baseline SD, and coast distance to 0.5 km, within original strata, requiring at least three controls per treated cell. They retain the original post-period neighborhood-adjustment assumption. Main-on-retained rows distinguish population restriction from changing comparisons. Spatial intervals condition on the baseline caliper graph. The retention table gives dose composition; the specification table gives per-pixel slopes and intervals. Interpret small retained sets as weak evidence, even when a percentile interval excludes zero.",
            "Compare each caliper result with the original method on exactly the same retained treated cells. Their difference diagnoses the effect of changed comparisons within that subset; it cannot establish absence of confounding in the full sample.",
            "The outcome-regression and DR rows target cell ATT on the main matched population, not a per-pixel slope. They use baseline surface/geographic covariates, squared baseline LST/NDVI and coarse-region indicators. Untreated-change OLS and unpenalized logistic propensity models supply the DR score; post-period NDVI/NDBI and development change are excluded from those models. The original population and stable-control selection assumptions remain. Spatial block influence intervals include nuisance estimation but are first-order approximations.",
            "[Panel DR-DiD reference](https://psantanna.com/DRDID/reference/drdid_panel.html). Conditional parallel trends, overlap and a correct propensity or untreated-change working model remain requirements. These exploratory fits do not establish immunity to unobserved confounding.",
            table(F["matching_specifications"],["setting","specification","max_control_ps","control_weight_ess","max_control_weight_share"]),
            "The DR rows retain the main matched treated population, while calipers restrict it. The weight table gives control effective sample size and largest normalized share for each setting. DR cannot validate a dose class outside the model's feature support on its own.",
            "followup_specification_comparison.csv combines estimates, retained counts and all standardized differences. followup_specification_balance.csv gives the long-form audit with means, using one fixed pre-matching SD per setting. DR rows use propensity-odds control weights. Outcome-regression rows show the unweighted estimation pool; regression does not itself balance that pool.",
            "followup_bias_scenarios.csv varies the assumed untreated treated-minus-control temperature change delta: corrected effect = estimate minus delta. Negative delta can inflate apparent cooling. These are additive tipping scenarios, not identified confounding bounds or evidence of plausibility."]
    return p

