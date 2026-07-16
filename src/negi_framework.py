import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.signal import savgol_filter
from xgboost import XGBRegressor

# CONFIGURATION
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_CSV = Path(os.environ.get("LST_INPUT_CSV", SCRIPT_DIR / "data" / "input.csv"))
DATA_DIR = Path(os.environ.get("LST_OUTPUT_DATA_DIR", SCRIPT_DIR / "outputs" / "data"))
PLOTS_DIR = Path(os.environ.get("LST_OUTPUT_PLOTS_DIR", SCRIPT_DIR / "outputs" / "plots"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("Input dataset :", INPUT_CSV)
print("Data outputs  :", DATA_DIR)
print("Plot outputs  :", PLOTS_DIR)
print("Scikit-learn version:", sklearn.__version__)

# PLOT STYLE
plt.style.use("seaborn-v0_8-whitegrid")

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
})

COLOR_PRIMARY = "#2E86AB"    # Scenario 1 / NDVI
COLOR_SECONDARY = "#E67E22"  # Scenario 2 / NDBI / Energy costs
COLOR_ACCENT = "#C0392B"     # Optimal markers / trendlines
COLOR_NEUTRAL = "#4D4D4D"    # Reference lines
FIGSIZE = (7, 5)


def confirm_saved(filepath):
    """Confirm that a file was actually written to disk."""
    filepath = str(filepath)
    if os.path.exists(filepath):
        print(f"Saved: {filepath} ({os.path.getsize(filepath)} bytes)")
    else:
        print(f"WARNING: expected file not found after save: {filepath}")


def save_fig(fig, filename):
    """Save a figure with consistent layout, verify it landed on disk, and close it."""
    fig.tight_layout()
    filepath = PLOTS_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    confirm_saved(filepath)
    plt.show()
    plt.close(fig)

# DATA PREPARATION & SPLITTING
df = pd.read_csv(INPUT_CSV).dropna()

X = df[["NDVI", "NDBI", "Elevation"]]
y = df["LST"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# XGBOOST MODEL TRAINING & HYPERPARAMETER TUNING
param_grid = {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1],}

base_model = XGBRegressor(objective="reg:squarederror", random_state=42)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Mean CV R²:", round(grid_search.best_score_, 3))

model = grid_search.best_estimator_

grid_results = pd.DataFrame(grid_search.cv_results_)
grid_results.to_csv(DATA_DIR / "grid_search_results.csv", index=False)
confirm_saved(DATA_DIR / "grid_search_results.csv")

pred = model.predict(X_test)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
print("Cross-validated R²:", cv_scores.mean().round(3))

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print("RMSE:", rmse)
print("R2:", r2)

# MODEL COMPARISON
comparison_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
    "XGBoost": model,
}

comparison_results = []

for name, mdl in comparison_models.items():
    print(f"Running {name}...")
    mdl.fit(X_train, y_train)
    y_pred_model = mdl.predict(X_test)

    comparison_results.append({
        "Model": name,
        "R2": r2_score(y_test, y_pred_model),
        "RMSE (°C)": np.sqrt(mean_squared_error(y_test, y_pred_model)),
        "MAE (°C)": mean_absolute_error(y_test, y_pred_model),
    })

comparison_df = pd.DataFrame(comparison_results).sort_values("R2", ascending=False)

print("\nModel comparison:")
print(comparison_df)

comparison_df.to_csv(DATA_DIR / "model_comparison.csv", index=False)
confirm_saved(DATA_DIR / "model_comparison.csv")

# Model comparison: R² by model
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.bar(
    comparison_df["Model"],
    comparison_df["R2"],
    color=[COLOR_NEUTRAL, COLOR_SECONDARY, COLOR_PRIMARY, COLOR_ACCENT],
)
ax.set_ylabel("R²")
ax.set_title("Machine Learning Model Comparison for LST Prediction")
plt.xticks(rotation=30, ha="right")
save_fig(fig, "figure_model_comparison_r2.png")

# MODEL PERFORMANCE FIGURES

# Feature importance
importance_values = model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance_values,
}).sort_values("Importance", ascending=False)

print("\nFeature importance:")
print(feature_importance)

fig, ax = plt.subplots(figsize=FIGSIZE)
bars = ax.bar(
    feature_importance["Feature"],
    feature_importance["Importance"],
    color=[COLOR_NEUTRAL, COLOR_SECONDARY, COLOR_PRIMARY],
    edgecolor="white",
    linewidth=0.5,
)
for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{height:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )
ax.set_ylabel("Feature Importance (Gain)")
ax.set_title("XGBoost Feature Importance (Gain)")
ax.margins(y=0.12)
save_fig(fig, "figure_feature_importance.png")

# NDVI distribution
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.hist(df["NDVI"], bins=40, color=COLOR_PRIMARY, edgecolor="white", alpha=0.9)
ax.axvline(
    df["NDVI"].median(),
    color=COLOR_ACCENT,
    linestyle="--",
    linewidth=1.5,
    label=f"Median = {df['NDVI'].median():.2f}",
)
ax.set_title("Distribution of NDVI Values")
ax.set_xlabel("NDVI")
ax.set_ylabel("Frequency")
ax.legend()
save_fig(fig, "figure1_ndvi_distribution.png")

# NDBI distribution
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.hist(df["NDBI"], bins=40, color=COLOR_SECONDARY, edgecolor="white", alpha=0.9)
ax.axvline(
    df["NDBI"].median(),
    color=COLOR_ACCENT,
    linestyle="--",
    linewidth=1.5,
    label=f"Median = {df['NDBI'].median():.2f}",
)
ax.set_title("Distribution of NDBI Values")
ax.set_xlabel("NDBI")
ax.set_ylabel("Frequency")
ax.legend()
save_fig(fig, "figure2_ndbi_distribution.png")

# NDVI vs. LST with linear trendline
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter(df["NDVI"], df["LST"], alpha=0.2, s=14, color=COLOR_PRIMARY, edgecolor="none")

reg = LinearRegression()
reg.fit(df[["NDVI"]], df["LST"])
x_range = np.linspace(df["NDVI"].min(), df["NDVI"].max(), 100)
ax.plot(
    x_range,
    reg.predict(pd.DataFrame(x_range, columns=["NDVI"])),
    color=COLOR_ACCENT,
    linewidth=2,
    label="OLS Fit",
)

ax.set_title("Relationship between NDVI and Land Surface Temperature")
ax.set_xlabel("NDVI")
ax.set_ylabel("LST (°C)")
ax.legend()
save_fig(fig, "figure3_ndvi_vs_lst.png")

# Actual vs predicted LST
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter(y_test, pred, alpha=0.4, s=16, color=COLOR_PRIMARY, edgecolor="none")
lims = [y_test.min(), y_test.max()]
ax.plot(lims, lims, color=COLOR_NEUTRAL, linewidth=1.2, label="1:1 line")
ax.set_aspect("equal", adjustable="box")
ax.text(
    0.05, 0.95,
    f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} °C",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.4"),
)
ax.set_title("Observed vs Predicted Land Surface Temperature")
ax.set_xlabel("Observed LST (°C)")
ax.set_ylabel("Predicted LST (°C)")
ax.legend(loc="lower right")
save_fig(fig, "figure4_actual_vs_predicted.png")

# Residuals
residuals = y_test - pred
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter(pred, residuals, alpha=0.3, s=14, color=COLOR_SECONDARY, edgecolor="none")
ax.axhline(0, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.2)
ax.set_title("Residuals of the Optimized XGBoost Model")
ax.set_xlabel("Predicted LST (°C)")
ax.set_ylabel("Residuals (°C)")
ax.set_ylim(-5, 5)  # bound the view to the typical residual range
save_fig(fig, "figure5_residuals.png")

# TWO-SCENARIO GREENING FRAMEWORK & NEGI COMPILATION
scenarios = np.arange(0.05, 0.55, 0.01)

desal_energy_intensity = 4.0  # kWh/m3
w0 = 200
alpha_weight = 10

ndvi_min = df["NDVI"].quantile(0.05)
ndvi_max = df["NDVI"].quantile(0.95)
fixed_elevation = df["Elevation"].median()
baseline_ndbi = df["NDBI"].median()
baseline_ndvi = df["NDVI"].median()


def predict_row(ndvi_val, ndbi_val):
    row = pd.DataFrame({
        "NDVI": [ndvi_val],
        "NDBI": [ndbi_val],
        "Elevation": [fixed_elevation],
    })
    return model.predict(row)[0]


baseline_lst = predict_row(baseline_ndvi, baseline_ndbi)

lst_s1_list = []
lst_s2_list = []

for fvc in scenarios:
    ndvi_val = ndvi_min + fvc * (ndvi_max - ndvi_min)

    # Scenario 1: greening only, NDBI held fixed
    lst_s1 = predict_row(ndvi_val, baseline_ndbi)
    lst_s1_list.append(lst_s1)

    # Scenario 2: urban transformation, NDBI decreases gradually with greening
    ndbi_val = baseline_ndbi * (1 - 0.7 * fvc)
    ndbi_val = np.clip(ndbi_val, df["NDBI"].min(), df["NDBI"].max())
    lst_s2 = predict_row(ndvi_val, ndbi_val)
    lst_s2_list.append(lst_s2)

lst_s1_arr = np.array(lst_s1_list)
lst_s2_arr = np.array(lst_s2_list)

smooth_window = 9 if len(scenarios) >= 9 else (
    len(scenarios) if len(scenarios) % 2 == 1 else len(scenarios) - 1
)
if smooth_window >= 5:
    lst_s1_arr = savgol_filter(lst_s1_arr, window_length=smooth_window, polyorder=2)
    lst_s2_arr = savgol_filter(lst_s2_arr, window_length=smooth_window, polyorder=2)

delta_T_s1 = baseline_lst - lst_s1_arr
delta_T_s2 = baseline_lst - lst_s2_arr

# Shared energy penalty
desalination_energy = (w0 * np.sqrt(scenarios)) * desal_energy_intensity
energy_norm = (desalination_energy - desalination_energy.min()) / (
    desalination_energy.max() - desalination_energy.min()
)

# Relative cooling benefit
benefit_s1 = alpha_weight * np.log1p(np.maximum(delta_T_s1, 0))
benefit_s2 = alpha_weight * np.log1p(np.maximum(delta_T_s2, 0))

max_benefit = max(benefit_s1.max(), benefit_s2.max())
min_benefit = min(benefit_s1.min(), benefit_s2.min())

benefit_s1_norm = (benefit_s1 - min_benefit) / (max_benefit - min_benefit)
benefit_s2_norm = (benefit_s2 - min_benefit) / (max_benefit - min_benefit)

# Normalized NEGI profiles
NEGI_s1 = benefit_s1_norm - energy_norm
NEGI_s2 = benefit_s2_norm - energy_norm

scenario_results = pd.DataFrame({
    "FVC": scenarios,
    "FVC (%)": scenarios * 100,
    "LST_S1": lst_s1_arr,
    "Cooling_S1": delta_T_s1,
    "NEGI_S1": NEGI_s1,
    "LST_S2": lst_s2_arr,
    "Cooling_S2": delta_T_s2,
    "NEGI_S2": NEGI_s2,
    "Relative_Desal_Energy": desalination_energy,
})

scenario_results["Scenario"] = np.where(
    scenario_results["FVC (%)"] <= 25,
    "Low vegetation expansion",
    "High vegetation expansion",
)

scenario_results.to_csv(DATA_DIR / "scenario_results.csv", index=False)
confirm_saved(DATA_DIR / "scenario_results.csv")

# NEGI comparison: Scenario 1 vs Scenario 2
fig, ax = plt.subplots(figsize=FIGSIZE)

ax.plot(
    scenarios * 100,
    NEGI_s1,
    linewidth=2,
    color=COLOR_PRIMARY,
    label="Scenario 1: Greening Only (NDBI Fixed)",
)
ax.plot(
    scenarios * 100,
    NEGI_s2,
    linewidth=2,
    color=COLOR_SECONDARY,
    label="Scenario 2: Urban Transformation (NDBI Decreased)",
)
ax.axhline(0, color=COLOR_NEUTRAL, linestyle="--", linewidth=1)

opt_s2_idx = np.argmax(NEGI_s2)
ax.scatter(
    [scenarios[opt_s2_idx] * 100], [NEGI_s2[opt_s2_idx]],
    marker="*", s=350, color=COLOR_ACCENT, edgecolor="white", zorder=5,
    label=f"Maximum NEGI among evaluated scenarios (FVC = {scenarios[opt_s2_idx] * 100:.1f}%)",
)

ax.set_xlabel("Fractional Vegetation Cover (%)")
ax.set_ylabel("Normalized NEGI (Cooling Benefit - Energy Cost)")
ax.set_title("Net Energy Gain Index (NEGI) Scenario Comparison")
ax.legend(loc="lower left")
save_fig(fig, "figure6_negi_scenario_comparison.png")

print("\nNEGI scenario results")
print(f"Scenario 1 Maximum NEGI: {NEGI_s1.max():.3f} at FVC = {scenarios[np.argmax(NEGI_s1)] * 100:.1f}%")
print(f"Scenario 2 Maximum NEGI: {NEGI_s2.max():.3f} at FVC = {scenarios[opt_s2_idx] * 100:.1f}%")

# NEGI SENSITIVITY ANALYSIS
print("\nRunning sensitivity analysis...")

fvc_values = scenarios * 100
alpha_values = np.arange(5, 35, 5) 
w0_values = np.arange(50, 450, 50)  
energy_exponents = [1.0, 1.25, 1.5, 1.75]

DEFAULT_EXPONENT = 1.5 

sensitivity_results = []

for exponent in energy_exponents:
    for alpha_val in alpha_values:
        for w0_val in w0_values:
            negi_profile = []
            for i, fvc in enumerate(fvc_values):
                cooling = delta_T_s2[i]
                energy_penalty = w0_val * ((fvc / 50) ** exponent)
                negi = (alpha_val * cooling) - energy_penalty
                negi_profile.append(negi)

            best = np.argmax(negi_profile)
            sensitivity_results.append({
                "Exponent": exponent,
                "Alpha": alpha_val,
                "w0": w0_val,
                "Optimal_FVC (%)": fvc_values[best],
                "Maximum_NEGI": negi_profile[best],
            })

sensitivity_df = pd.DataFrame(sensitivity_results)

print("\nSensitivity results (head):")
print(sensitivity_df.head())

sensitivity_df.to_csv(DATA_DIR / "negi_sensitivity_analysis.csv", index=False)
confirm_saved(DATA_DIR / "negi_sensitivity_analysis.csv")

plot_df = sensitivity_df[sensitivity_df["Exponent"] == DEFAULT_EXPONENT]

# Sensitivity: maximum NEGI vs. irrigation coefficient
plt.figure(figsize=(8, 6))
for alpha_val in alpha_values:
    subset = plot_df[plot_df["Alpha"] == alpha_val]
    plt.plot(
        subset["w0"],
        subset["Maximum_NEGI"],
        marker="o",
        linewidth=2,
        label=f"α = {alpha_val}",
    )

plt.xlabel("Irrigation scaling coefficient (w0)")
plt.ylabel("Maximum NEGI")
plt.title(f"Sensitivity of Maximum NEGI to Scaling Coefficients (exponent = {DEFAULT_EXPONENT})")
plt.grid(True, alpha=0.3)
plt.legend()
sensitivity_plot_1_path = PLOTS_DIR / "sensitivity_maximum_negi.png"
plt.savefig(sensitivity_plot_1_path, dpi=300, bbox_inches="tight")
confirm_saved(sensitivity_plot_1_path)
plt.show()
plt.close()

# Sensitivity: optimal FVC heatmap
pivot = plot_df.pivot(index="Alpha", columns="w0", values="Optimal_FVC (%)")

plt.figure(figsize=(8, 5))
plt.imshow(pivot.values, aspect="auto")
plt.colorbar(label="Optimal FVC (%)")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xlabel("Irrigation Scaling Coefficient (w0)")
plt.ylabel("Alpha")
plt.title(f"Optimal FVC under NEGI Parameter Sensitivity (exponent = {DEFAULT_EXPONENT})")

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        plt.text(
            j, i, int(pivot.iloc[i, j]),
            ha="center", va="center", color="white",
        )

plt.tight_layout()
sensitivity_plot_2_path = PLOTS_DIR / "negi_optimal_fvc_heatmap.png"
plt.savefig(sensitivity_plot_2_path, dpi=300, bbox_inches="tight")
confirm_saved(sensitivity_plot_2_path)
plt.show()
plt.close()

# Sensitivity: NEGI response to the energy cost exponent
cases = [
    {"alpha": 10, "w0": 50},
    {"alpha": 15, "w0": 200},
    {"alpha": 30, "w0": 50},
]

plt.figure(figsize=(9, 5))
for case in cases:
    a, w = case["alpha"], case["w0"]
    for exponent in energy_exponents:
        negi_profile = [
            (a * delta_T_s2[i]) - w * ((fvc / 50) ** exponent)
            for i, fvc in enumerate(fvc_values)
        ]
        plt.plot(
            fvc_values,
            negi_profile,
            linewidth=1.8,
            label=f"α={a}, w0={w}, β={exponent}",
        )

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Fractional Vegetation Cover (%)")
plt.ylabel("NEGI")
plt.title("NEGI Sensitivity to Cooling Weight and Energy Scaling")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
sensitivity_plot_3_path = PLOTS_DIR / "negi_exponent_sensitivity.png"
plt.savefig(sensitivity_plot_3_path, dpi=300, bbox_inches="tight")
confirm_saved(sensitivity_plot_3_path)
plt.show()
plt.close()

# SCENARIO EXPORTS
scenario_feature_combos = []
for fvc in scenarios:
    ndvi_scenario = ndvi_min + fvc * (ndvi_max - ndvi_min)
    ndbi_scenario = baseline_ndbi * (1 - 0.7 * fvc)
    ndbi_scenario = np.clip(ndbi_scenario, df["NDBI"].min(), df["NDBI"].max())
    scenario_feature_combos.append({
        "FVC (%)": fvc * 100,
        "NDVI": ndvi_scenario,
        "NDBI": ndbi_scenario,
        "Elevation": fixed_elevation,
    })

scenario_df = pd.DataFrame(scenario_feature_combos)
scenario_df.to_csv(DATA_DIR / "scenario_feature_combinations.csv", index=False)
confirm_saved(DATA_DIR / "scenario_feature_combinations.csv")

# Formatted scenario results for reporting
scenario_table = scenario_results.rename(columns={
    "LST_S1": "S1 Predicted LST (°C)",
    "Cooling_S1": "S1 Cooling (°C)",
    "NEGI_S1": "S1 NEGI",
    "LST_S2": "S2 Predicted LST (°C)",
    "Cooling_S2": "S2 Cooling (°C)",
    "NEGI_S2": "S2 NEGI",
    "Relative_Desal_Energy": "Relative Energy Cost",
    "Scenario": "Expansion Phase",
})[[
    "FVC (%)",
    "Expansion Phase",
    "S1 Predicted LST (°C)",
    "S1 Cooling (°C)",
    "S1 NEGI",
    "S2 Predicted LST (°C)",
    "S2 Cooling (°C)",
    "S2 NEGI",
    "Relative Energy Cost",
]].copy()

for col in scenario_table.columns:
    if col not in ["FVC (%)", "Expansion Phase"]:
        scenario_table[col] = scenario_table[col].round(3)

scenario_table.to_csv(DATA_DIR / "scenario_results_formatted.csv", index=False)
confirm_saved(DATA_DIR / "scenario_results_formatted.csv")

print("\nScenario results (formatted):")
print(scenario_table)

# LST response to NDBI, NDVI and elevation fixed
ndbi_range = np.linspace(df["NDBI"].quantile(0.05), df["NDBI"].quantile(0.95), 50)
X_ndbi_sweep = np.column_stack([
    np.full(50, df["NDVI"].median()),
    ndbi_range,
    np.full(50, fixed_elevation),
])
pred_ndbi = model.predict(X_ndbi_sweep)

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(ndbi_range, pred_ndbi, color=COLOR_SECONDARY, linewidth=2)
ax.set_xlabel("NDBI")
ax.set_ylabel("Predicted LST (°C)")
ax.set_title("LST Response to NDBI (NDVI and Elevation Fixed)")
save_fig(fig, "figure7_lst_response_to_ndbi.png")
