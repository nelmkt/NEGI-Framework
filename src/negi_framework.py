import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from xgboost import XGBRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV
)
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("Scikit-learn version:", sklearn.__version__)

df = pd.read_csv("data/Jeddah_NDVI_NDBI_LST_dataset.csv")

print(df.head())
print(df.columns)

df = df.dropna()

print(df.isnull().sum())

X = df[["NDVI", "NDBI"]]
y = df["LST"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1]
}

base_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("Best Mean CV R²:")
print(round(grid_search.best_score_, 3))

model = grid_search.best_estimator_

grid_results = pd.DataFrame(grid_search.cv_results_)
grid_results.to_csv(
    "data/Grid_Search_Results.csv",
    index=False
)

pred = model.predict(X_test)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
print("Cross-validated R²:", cv_scores.mean().round(3))

mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, pred)

print("RMSE:", rmse)
print("R2:", r2)


importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance", ascending=False)

print("\nFeature Importance")
print(feature_importance)

plt.figure()
plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)
plt.ylabel("Importance Score")
plt.title("XGBoost Feature Importance")
plt.savefig(
    "plots/Figure_Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


#NDVI Distribution Figure
plt.figure()
plt.hist(df["NDVI"], bins=40)
plt.title("NDVI Distribution in Jeddah")
plt.xlabel("NDVI")
plt.ylabel("Frequency")
plt.savefig("plots/Figure1_NDVI_Distribution.png", dpi=300, bbox_inches="tight")
plt.show()

#NDBI Distribution Figure
plt.figure()
plt.hist(df["NDBI"], bins=40)
plt.title("NDBI Distribution in Jeddah")
plt.xlabel("NDBI")
plt.ylabel("Frequency")
plt.savefig("plots/Figure2_NDBI_Distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# NDVI vs LST Figure
plt.figure()
plt.scatter(df["NDVI"], df["LST"], alpha=0.3)
plt.title("NDVI vs Land Surface Temperature (Jeddah)")
plt.xlabel("NDVI")
plt.ylabel("LST (°C)")
plt.savefig("plots/Figure3_NDVI_vs_LST.png", dpi=300, bbox_inches="tight")
plt.show()

# Actual vs Predicted LST Figure
plt.figure()
plt.scatter(y_test, pred, alpha=0.4)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
plt.title("XGBoost Model Performance")
plt.xlabel("Actual LST (°C)")
plt.ylabel("Predicted LST (°C)")
plt.savefig("plots/Figure4_Actual_vs_Predicted.png", dpi=300, bbox_inches="tight")
plt.show()

#Residual Analysis Figure
residuals = y_test - pred

plt.figure()
plt.scatter(pred, residuals, alpha=0.4)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted LST (°C)")
plt.ylabel("Residuals (°C)")
plt.title("Residual Plot")
plt.savefig(
    "plots/Figure5_Residuals.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\nNDVI Statistics")
print("\nCorrelation Matrix")
print(
    df[["NDVI", "NDBI", "LST"]].corr()
)
print(df.columns)


desal_energy_intensity = 4.0  # kWh/m³ (Ghaffour et al., 2013)


ndvi_min = df["NDVI"].min()
ndvi_max = df["NDVI"].max()

df["FVC"] = (df["NDVI"] - ndvi_min) / (ndvi_max - ndvi_min)
df["FVC"] = df["FVC"].clip(0, 1)

print("\nFVC Statistics")
print(df["FVC"].describe())


scenarios = np.array([
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50
])

ndbi_mean = df["NDBI"].mean()

baseline_lst = model.predict(
    np.array([[df["NDVI"].mean(), ndbi_mean]])
)[0]

predicted_lsts = []

for fvc in scenarios:
    ndvi_scenario = ndvi_min + fvc * (ndvi_max - ndvi_min)
    predicted_lst = model.predict(
        np.array([[ndvi_scenario, ndbi_mean]])
    )[0]
    predicted_lsts.append(predicted_lst)

predicted_lsts = np.array(predicted_lsts)

delta_T = baseline_lst - predicted_lsts

w0 = 200  # baseline irrigation coefficient
irrigation_index = scenarios * w0

desalination_energy = irrigation_index * desal_energy_intensity

alpha = 10  # empirical scaling coefficient
cooling_energy_equivalent = delta_T * alpha

#Compute Net Energy Gain Index (NEGI) Figure

NEGI = cooling_energy_equivalent - desalination_energy

scenario_results = pd.DataFrame({
    "FVC": scenarios,
    "Predicted_LST": predicted_lsts,
    "Temperature_Reduction_degC": delta_T,
    "Water_Index": irrigation_index,
    "Relative_Desalination_Energy": desalination_energy,
    "Relative_Cooling_Benefit": cooling_energy_equivalent,
    "NEGI": NEGI
})

scenario_results.to_csv("data/Scenario_Results.csv", index=False)

#Cooling Benefit vs Energy Cost Figure
plt.figure()
plt.plot(
    scenarios * 100,
    cooling_energy_equivalent,
    marker="o",
    label="Cooling Energy Savings"
)
plt.plot(
    scenarios * 100,
    desalination_energy,
    marker="s",
    label="Desalination Energy Cost"
)
plt.xlabel("Fractional Vegetation Cover (%)")
plt.ylabel("Energy")
plt.title("Cooling Benefit vs Desalination Energy Consumption")
plt.legend()
plt.savefig(
    "plots/Figure6_Cooling_vs_Energy.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#NEGI Curve Figure
plt.figure()
plt.plot(scenarios * 100, NEGI, marker="o")
plt.axhline(0, linestyle="--")
plt.xlabel("Fractional Vegetation Cover (%)")
plt.ylabel("NEGI")
plt.title("Net Energy Gain Index")
plt.savefig(
    "plots/Figure7_NEGI_Curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Optimal Threshold Figure
optimal_index = np.argmax(NEGI)
optimal_fvc = scenarios[optimal_index] * 100

plt.figure()
plt.plot(scenarios * 100, NEGI, marker="o")
plt.axvline(
    optimal_fvc,
    linestyle="--",
    label=f"Optimal = {optimal_fvc:.1f}%"
)
plt.axhline(0, linestyle="--")
plt.xlabel("Fractional Vegetation Cover (%)")
plt.ylabel("NEGI")
plt.title("Optimal Vegetation Threshold")
plt.legend()
plt.savefig(
    "plots/Figure8_Optimal_Threshold.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\n=== NEGI RESULTS ===")
print("Optimal Vegetation Threshold (% FVC):", round(optimal_fvc, 2))
print("\nFVC scenarios:")
print(scenarios)
print("\nCooling Energy:")
print(cooling_energy_equivalent)
print("\nDesalination Energy:")
print(desalination_energy)
print("\nNEGI:")
print(NEGI)


alpha_values = [8, 10, 12]
w0_values = [150, 200, 250]

results = []

for alpha in alpha_values:
    cooling = delta_T * alpha
    for w0 in w0_values:
        irr = scenarios * w0
        desal = irr * desal_energy_intensity
        negi = cooling - desal
        optimum = scenarios[np.argmax(negi)] * 100
        results.append({
            "Alpha": alpha,
            "w0": w0,
            "Optimal_FVC": optimum,
            "Maximum NEGI": np.max(negi)
        })

sensitivity_df = pd.DataFrame(results)

print("\nSensitivity Analysis")
print(sensitivity_df)

sensitivity_df.to_csv("data/Sensitivity_Analysis.csv", index=False)

#Sensitivity Analysis Figure
plt.figure(figsize=(8, 5))

for alpha_test in alpha_values:
    subset = sensitivity_df[sensitivity_df["Alpha"] == alpha_test]
    plt.plot(
        subset["w0"],
        subset["Maximum NEGI"],
        marker="o",
        label=f"α = {alpha_test}"
    )

plt.xlabel("Irrigation Scaling Coefficient (w₀)")
plt.ylabel("Maximum NEGI")
plt.title("Sensitivity of NEGI to Scaling Coefficients")
plt.legend()
plt.savefig(
    "plots/Figure9_Sensitivity_Analysis.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
