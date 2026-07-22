# Full Summary Report

## EDA

| Quantity | Value |
| --- | --- |
| NDVI-LST Pearson r | -0.1511 |
| NDBI-LST Pearson r | 0.5291 |
| NDVI-NDBI Pearson r | -0.6371 |

## Training performance

| Quantity | Value |
| --- | --- |
| Inner tuning score (RandomizedSearchCV mean_test_score, GroupKFold) | 0.5427 |
| Nested GroupKFold CV R² (headline, outer-fold) | 0.5427 ± 0.0190 |
| Nested GroupKFold CV outer folds | 5 |

## Nested spatial CV performance

| Quantity | Value |
| --- | --- |
| Repeated spatial-holdout R² (mean ± 95% CI) | 0.5466 ± 0.0107 |
| Repeated spatial-holdout RMSE (mean ± 95% CI, °C) | 1.7608 ± 0.0101 |
| Repeated spatial-holdout MAE (mean ± 95% CI, °C) | 1.4039 ± 0.0106 |

## Holdout performance

| Quantity | Value |
| --- | --- |
| Single holdout RMSE (°C) | 1.7400 |
| Single holdout R² | 0.5378 |
| Single holdout MAE (°C) | 1.3752 |
| Train R² | 0.5845 |
| Train-test R² gap | 0.0467 |

## Calibration

| Quantity | Value |
| --- | --- |
| Calibration slope | 0.9852 |
| Calibration intercept (°C) | 0.8504 |
| Mean bias (°C) | 0.1100 |
| Recalibration applied | False |
| Reporting note | Calibration diagnostics are reported descriptively only; no post-hoc recalibration was applied.  RMSE and MAE are reported once in the Confirmatory Holdout section and are not repeated here.  MACE is omitted because it is identical to MAE. |

## Bootstrap CI

| Quantity | Value |
| --- | --- |
| rmse (point estimate) | 1.7400 |
| rmse 95% CI | 1.7068-1.7745 |
| mae (point estimate) | 1.3752 |
| mae 95% CI | 1.3476-1.4032 |
| bias (point estimate) | 0.1100 |
| bias 95% CI | 0.0648-0.1579 |
| calibration_slope (point estimate) | 0.9852 |
| calibration_slope 95% CI | 0.9617-1.0106 |
| calibration_intercept (point estimate) | 0.8504 |
| calibration_intercept 95% CI | -0.4365-2.0331 |

## Residual diagnostics

| Quantity | Value |
| --- | --- |
| Residual skewness | -0.1872 |
| Residual kurtosis (excess) | 0.2970 |
| Breusch-Pagan statistic | 57.2008 |
| Breusch-Pagan p-value | 3.93e-14 |
| Breusch-Pagan implementation | auxiliary-regression approximation (statsmodels unavailable) |
| Spearman \|residual\| vs predicted (rho) | -0.0964 |
| Spearman \|residual\| vs predicted (p) | 1.064e-12 |
| Global Moran's I (holdout residuals) | 0.5371 |
| Moran's I p-value | 0 |
| Breusch-Pagan statistic | 57.2008 |
| Breusch-Pagan p-value | 3.93e-14 |

## Interpretive Diagnostics

| Quantity | Value |
| --- | --- |
| Holdout observations (n) | 5431 |
| Approximate effective sample size (n_eff) | 1636 |
| Approximate n_eff reduction | 70% |

## Model comparison

| Quantity | Value |
| --- | --- |
| Benchmark note | XGBoost was optimized using RandomizedSearchCV. Benchmark models are included for contextual comparison and were intentionally not optimized to the same extent. Therefore, benchmark comparisons should be interpreted as contextual rather than evidence of global model superiority. |

## Feature importance

| Quantity | Value |
| --- | --- |
| NDBI permutation importance (mean ± 95% CI) | 0.8129 ± 0.0022 |
| Elevation permutation importance (mean ± 95% CI) | 0.3788 ± 0.0006 |
| NDVI permutation importance (mean ± 95% CI) | 0.1551 ± 0.0007 |

## Supplementary Diagnostics - Staircase

| Quantity | Value |
| --- | --- |
| Scenario 1 mean jump (°C) | 0.00402 |
| Scenario 1 maximum jump (°C) | 0.12455 |
| Scenario 1 jump count (> 0.1°C) | 2 |
| Scenario 2 mean jump (°C) | 0.01052 |
| Scenario 2 maximum jump (°C) | 0.47550 |
| Scenario 2 jump count (> 0.1°C) | 10 |

## Supplementary Diagnostics - Local Gradient

| Quantity | Value |
| --- | --- |
| Scenario 1 median local derivative (°C/scenario%) | 0.00000 |
| Scenario 1 max local derivative (°C/scenario%) | 0.49818 |
| Scenario 1 min local derivative (°C/scenario%) | -0.22737 |
| Scenario 2 median local derivative (°C/scenario%) | 0.00000 |
| Scenario 2 max local derivative (°C/scenario%) | 0.75287 |
| Scenario 2 min local derivative (°C/scenario%) | -1.90201 |

## Scenario diagnostics

| Quantity | Value |
| --- | --- |
| S2 fraction warming (unclipped) | 68.58% |
| S2 fraction cooling (unclipped) | 30.67% |
| S2 maximum cooling (unclipped, °C) | 0.9055 |
| S2 maximum warming (unclipped, °C) | 0.3798 |
| S2 first cooling point (scenario %) | 69.50 |

## Support diagnostics

| Quantity | Value |
| --- | --- |
| Scenario 1 maximum NN distance | 0.1590 |
| Scenario 1 median NN distance | 0.0588 |
| Scenario 1 95th percentile NN dist | 0.1407 |
| Scenario 1 points beyond 95th pct | 20 |
| Scenario 2 maximum NN distance | 0.1264 |
| Scenario 2 median NN distance | 0.0591 |
| Scenario 2 95th percentile NN dist | 0.1039 |
| Scenario 2 points beyond 95th pct | 20 |
| Scenario 1 fully within dense support | True |
| Scenario 2 fully within dense support | True |
| Feature bounds check | PASS |
| kNN support check | PASS |

## Uncertainty diagnostics

| Quantity | Value |
| --- | --- |
| N spatial refits | 10 |
| Peak S2 NEGI SD (scenario %) | 98.2 |
| Peak S2 NEGI SD (value) | 0.1293 |
| Peak S2 NEGI IQR (scenario %) | 99.0 |
| Peak S2 NEGI IQR (Q75-Q25) | 0.2244 |

## Supplementary Diagnostics - Smoothing

| Quantity | Value |
| --- | --- |
| Smoothing robust | True |
| Peak location out-of-feature-bounds | False |

## NEGI scenario comparison

| Quantity | Value |
| --- | --- |
| Feature support flag | OK: within training support |
| Smoothing robust (Savgol) | True |
| Peak location OOD | False |

## Warnings

| Quantity | Value |
| --- | --- |
| Warning 1 | Significant residual spatial autocorrelation detected (Moran's I = 0.5371, p = 0).  Confidence intervals should be interpreted accordingly. |
| Warning 2 | Scenario 1: The highest evaluated NEGI occurs near the boundary of the explored scenario trajectory. This should be interpreted as the best-performing point within the evaluated scenario rather than evidence of a unique interior or global optimum.  Results are sensitive to cost-function parameters (w0, alpha, beta); see sensitivity analysis. |
| Warning 3 | Scenario 2: The highest evaluated NEGI occurs near the boundary of the explored scenario trajectory. This should be interpreted as the best-performing point within the evaluated scenario rather than evidence of a unique interior or global optimum. |

## Conclusions

| Quantity | Value |
| --- | --- |
| Conclusion 1 | Residual spatial dependence remained after spatial-block validation, indicating that some spatially structured variation in LST remains unexplained. Point estimates remain valid, but local prediction errors may exhibit spatial correlation and predictive performance should be interpreted with appropriate caution. |
| Conclusion 2 | NDBI is the dominant predictor of LST in Jeddah (permutation importance = 0.813), substantially exceeding Elevation (0.379) and NDVI (0.155). |
| Conclusion 3 | Only three predictors (NDBI, Elevation, and NDVI) were intentionally used in this model. The remaining unexplained variance likely reflects additional environmental processes not captured by these variables. Omitted variables such as building morphology, wind exposure, and land-use type are presented only as potential future work and do not represent demonstrated deficiencies of the current model. |
| Conclusion 4 | Spatial holdout R² = 0.538, RMSE = 1.74 °C.  Maximum predicted cooling (0.91 °C) is below the model RMSE; cooling-magnitude estimates should be interpreted with caution. |
| Conclusion 5 | Scenario 1 Maximum Evaluated NEGI occurs at the trajectory boundary (0%).  No interior optimum is identifiable under greening-only conditions at the reference parameter values. |

