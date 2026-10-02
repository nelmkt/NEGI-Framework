# NEGI_LOGIC_AUDIT_V11 (Section A9)

Written 2026-10-02. Source: `code/negi_original/NEGI_Framework.py` (SHA256 02988C8D7F6C449EF3AE727673B62F57D1EF628C3A7E60E2029F844A08A6602B), read by line range and grep only; not executed, not imported. `NF:n` = line n of that file. Every line number below comes from a read performed this session.

This audit describes the metric as implemented. It does not propose or implement a corrected NEGI.

## The implementation in one place

```
delta_t_j(f)      = baseline_lst - lst_j(f)                         NF:4624-4626, NF:9057-9058
benefit_j(f)      = max(delta_t_j(f), 0)                             NF:9090-9091
S                 = max(max_f benefit_1, max_f benefit_2, EPS)       NF:9115
cost_raw(f; w0)   = w0 * desal_energy_intensity * f^p                NF:9026
cost_norm(f)      = cost_raw(f; w0) / max_f cost_raw(f; reference_w0)   NF:9044-9046
NEGI_j(f)         = alpha_weight * benefit_j(f) / S - beta_weight * cost_norm(f)   NF:9120-9121
```

f is the scenario fraction, `np.arange(0.0, 1.0001, 0.0025)` (NF:8321 with `scenario_step` at NF:614), so f runs from 0 to 1 and max f^p = 1. j = 1 is Scenario 1 (NDVI raised, NDBI fixed), j = 2 is Scenario 2 (NDVI and NDBI moved together).

## (a) Same cost array in both pathways: **confirmed**

- One array, `energy_norm_sqrt`, is computed once from `s1.scenario_fraction` (NF:9083-9085).
- It is subtracted in both `negi_s1` (NF:9120) and `negi_s2` (NF:9121).
- The two scenarios share one fraction axis (NF:8321; length equality asserted at NF:9138-9140).

Therefore, point by point, NEGI_s2 - NEGI_s1 = alpha_weight x (benefit_s2_norm - benefit_s1_norm). The cost term cancels in any comparison of the two pathways at the same f. The same holds for the linear-cost variants (NF:9122-9123) and inside every spatial refit (NF:10128-10131, which reuse `negi.energy_norm_sqrt`).

## (b) Cost at `reference_w0`: **confirmed**

- `_energy_cost` returns `w0 * cfg.desal_energy_intensity * (scenario_fraction ** exponent)` (NF:9026).
- `_energy_norm` divides it by the maximum of the same expression evaluated at `cfg.reference_w0` (NF:9044-9046).
- The primary NEGI calls `_energy_norm` with `w0 = cfg.reference_w0` (NF:9083-9085).

So cost_norm(f) = [w0 x eta x f^p] / [w0 x eta x max f^p] = f^p / max(f^p). `reference_w0` (200.0, NF:747) and `desal_energy_intensity` (3.5, NF:748) cancel. The function's own docstring says so (NF:9035-9042): "it and desal_energy_intensity affect results only in the sensitivity analysis".

Inputs to the cost: scenario fraction, w0, exponent, cfg (signature at NF:9023-9025, NF:9029-9031). No irrigated area, no land-cover class, no vegetation type, no pathway identifier and no pixel count enters it. The cost at a given f is identical for the vegetation-only pathway and the coordinated pathway although the two pathways reach different NDVI values at that f (Scenario 1: linear to the 95th percentile NDVI, NF:8334, NF:8384-8386; Scenario 2: the archetype path, NF:8396-8399).

Consequence for wording: at the reference setting the cost term is the shape assumption f^p with p = 0.5 (NF:749), not a measured or estimated energy quantity.

## (c) `cooling_benefit = max(delta_t, 0)` discards predicted warming: **confirmed**

- NF:9090-9091: `np.maximum(delta_t_s1, 0.0)`, `np.maximum(delta_t_s2, 0.0)`. Same clip in every refit (NF:10126-10127) and in the sensitivity sweep (NF:11511).
- Unclipped diagnostics are kept separately: warming fraction, cooling fraction, maximum cooling, maximum warming, first cooling point, Scenario 2 only (NF:9060-9068).

Where it binds, per the code's own comments (not run outputs):

- Scenario 1: at every evaluated point. NF:145-151: delta_t_s1 "is <= 0 across the isolated-NDVI trajectory" and the clip "holds the cooling-benefit term at exactly zero for every evaluated point".
- Scenario 2: wherever delta_t_s2 < 0. The accepted abstract places first cooling at about 69.5% of the trajectory (`Manuscript.docx` body paragraph 11), which implies the clip binds or the benefit is zero over roughly the first 69.5% of points.

Share of points over which it binds, from `scenario_points.csv`: **not found** (the file does not exist; the code's equivalent exports `scenario1_trajectory.csv`, `scenario2_trajectory.csv`, `negi_results.csv` at NF:15883-15920 do not exist either). **blocked: no rerun outputs**.

## (d) Benefit scale depends on both scenarios: **confirmed**

NF:9115: `reference_benefit_scale = max(cooling_benefit_s1.max(), cooling_benefit_s2.max(), EPS)`. The scale is the largest positive predicted cooling over both scenarios, taken from the primary fit, and reused unchanged in every refit (NF:9099-9110, NF:10111-10127).

Consequences that follow from the lines above:

1. Changing either scenario's definition rescales the other scenario's NEGI.
2. The scenario containing the maximum has normalised benefit exactly 1 at that point, so its NEGI there is alpha - beta x f^p, a function of the position f alone. With the defaults and p = 0.5, a maximum at f = 0.99 gives 1 - 0.994987 = 0.005013 for any cooling magnitude.
3. Because the scale is fixed from the primary fit, a refit's normalised benefit can exceed 1, so a refit's NEGI can exceed the primary point estimate. The interval is "conditional on this fixed reference scale" (NF:9231-9236).
4. A second, different scale exists: `reference_cooling_c` = `reference_cooling_scale_factor` x max(|delta_t_s1|, |delta_t_s2|) (NF:9075-9077), which includes warming magnitudes. The comments say it is not used in the primary NEGI (NF:9070-9074).

## (e) Upper bound and sign: bound **confirmed**; sign **partly**

- Defaults: `alpha_weight = 1.0` (NF:745), `beta_weight = 1.0` (NF:746).
- Upper bound: benefit/S is at most 1 in the primary fit and the cost is non-negative, so NEGI is at most alpha_weight. The code asserts it (NF:9133-9137). **confirmed** for the primary fit. It does not hold for refits, where benefit/S can exceed 1 (see (d)3); the assertion is not applied there (NF:10126-10131).
- Lower bound, by the same algebra: -beta_weight (benefit 0 at f = 1).
- Sign: NEGI(f) > 0 exactly when alpha x benefit(f)/S > beta x f^p. alpha and beta do not set the sign alone; they set it together with the normalised benefit and the cost shape. Only the ratio alpha/beta matters for the sign and for the position of the maximum. The code states the same for the sensitivity form: "the maximum position depends on (alpha, beta, w0) only through c", c = (beta/alpha) x (w0/reference_w0) (NF:11587-11590, NF:11937). Hence **partly**: the weights are free parameters that can move the sign at any point where benefit is positive, and at any point where benefit is zero NEGI is negative for every beta > 0 and f > 0.
- No source, calibration or data is given in the file for alpha = beta = 1; they are defaults.

## (f) How `build_scenario2_empirical_trajectory` constructs Pathway 2

Construction (NF:8139-8221, with NF:8002-8060 and NF:8063-8136):

1. All quantities come from the training partition (NF:8319).
2. Baseline = (median NDVI, median NDBI) (NF:8332-8333).
3. Endpoint: the NDBI percentile is fixed at 0.15 (NF:705). `_empirical_cumulative_bin_stats` takes all pixels with NDBI at or below that percentile (NF:8035-8036), computes the subset's median NDVI and median NDBI as a temporary target (NF:8042-8043), and returns the single real pixel nearest to that target in z-scored (NDVI, NDBI) space (NF:8050-8057).
4. Path: 40 NDBI percentiles evenly spaced from the baseline's NDBI percentile to 0.15 (NF:8189-8191, `s2_trajectory_n_bins` at NF:708). For each, the same nearest-real-pixel rule gives one (NDVI, NDBI) pair (NF:8193-8202). Bins with fewer than 30 pixels reuse the previous bin (NF:8197-8200, NF:706).
5. The first bin is overwritten with the baseline (NF:8208-8209). This is the baseline-anchoring fix.
6. The 40 pairs are linearly interpolated onto the 401-point fraction axis (NF:8218-8220).
7. Elevation, ST_EMIS and ST_EMSD are held at training medians (NF:8328-8331).
8. The model is evaluated along the path afterwards (NF:8419-8423).

Does it represent an intervention or an observed co-variation? **Observed co-variation.** Evidence:

- The docstring says the path "directly follows the model's learned response through the observed NDVI-NDBI joint co-variation (moving through real, increasingly-vegetated / less-built-up observed pixels)" (NF:8148-8150).
- Each of the 40 nodes is a different existing pixel chosen from the cross-section because of where it sits in the NDBI distribution. No pixel is followed through time and nothing is changed on any pixel.
- The path is indexed by NDBI percentile, not by vegetation added. NDVI at each node is whatever the selected pixel happens to have. Nothing in NF:8189-8220 forces NDVI to rise along the path.
- Step 6 interpolates linearly between consecutive real pixels, so points between nodes are synthetic blends, although the docstring says the path is "never ... a synthetic linear blend" (NF:8153-8156) and the caller's comment says it "is NOT a linear blend of two endpoint values" (NF:8393-8395). The 40 nodes are real pixels; the 401 evaluated points are not all real pixels. Verdict on the "no synthetic point" claim: **partly**.
- The code labels the scenario "realistic municipal greening intervention" (NF:8227-8228, NF:8235, NF:8461). That label is an interpretation placed on a cross-sectional gradient. The file contains no intervention data (no planting records, no before-and-after observations).

The model's prediction along this path is therefore the fitted cross-sectional LST surface read along an observed NDVI-NDBI gradient. Reading it as the effect of converting built-up land to vegetation requires the assumption that a pixel moved to another pixel's (NDVI, NDBI) would take that pixel's LST with elevation and emissivity unchanged. The file states the non-causal caveat itself (NF:9186-9188: "should not be interpreted as evidence that vegetation causes warming").

## Summary table

| Item | Verdict | Key lines |
|---|---|---|
| (a) shared cost array; difference = alpha x benefit difference | confirmed | NF:9083-9085, 9120-9121 |
| (b) cost reduces to f^p / max(f^p); w0 and intensity cancel; no area or pathway input | confirmed | NF:9026, 9044-9046, 9083-9085 |
| (c) clip discards predicted warming | confirmed | NF:9090-9091 |
| (c) share of points where it binds | not found; blocked: no rerun outputs | - |
| (d) scale depends on both scenarios | confirmed | NF:9115 |
| (e) upper bound alpha_weight (primary fit) | confirmed | NF:9133-9137 |
| (e) alpha and beta set the sign | partly | NF:9120-9121, 11587-11590 |
| (e) defaults | alpha 1.0, beta 1.0 | NF:745-746 |
| (f) Pathway 2 is an observed co-variation path, not an intervention | confirmed | NF:8148-8150, 8189-8220 |
| (f) "never synthetic" claim | partly | NF:8153-8156 versus 8218-8220 |
