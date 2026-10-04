"""
=======================================================================
PHASE 2: RANDOM FOREST REGRESSION MODEL
Region-Aware Hybrid Deep Learning for Crop Yield Prediction
=======================================================================
Run after Phase 1. Requires:
  outputs/train_data.csv
  outputs/test_data.csv
  outputs/label_encoders.json
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble          import RandomForestRegressor
from sklearn.metrics           import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection   import GridSearchCV, cross_val_score
import joblib

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# HELPER
# -----------------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def print_metrics(label, y_true, y_pred):
    print(f"\n  [{label}]")
    print(f"    RMSE : {rmse(y_true, y_pred):.4f}")
    print(f"    MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"    R²   : {r2_score(y_true, y_pred):.4f}")

# -----------------------------------------------------------------------
# STEP 1: LOAD DATA
# -----------------------------------------------------------------------
print("=" * 65)
print("STEP 1: Load Preprocessed Data")
print("=" * 65)

train_df = pd.read_csv(f"{OUTPUT_DIR}/train_data.csv")
test_df  = pd.read_csv(f"{OUTPUT_DIR}/test_data.csv")

print(f"  Train : {train_df.shape}")
print(f"  Test  : {test_df.shape}")
print(f"  Columns: {list(train_df.columns)}\n")

# -----------------------------------------------------------------------
# STEP 2: DEFINE FEATURES & TARGET
# -----------------------------------------------------------------------
print("=" * 65)
print("STEP 2: Define Feature Set")
print("=" * 65)

TARGET = "Yield"

# Use scaled versions of numerical features if available
CANDIDATE_FEATURES = [
    # Agricultural
    "Area_scaled", "Production_scaled", "Fertilizer_scaled", "Pesticide_scaled",
    # Climate
    "Annual_Rainfall_scaled", "Temperature_scaled", "Humidity_scaled",
    # Soil
    "N_scaled", "P_scaled", "K_scaled", "pH_scaled", "NPK_total_scaled",
    # Time-series
    "Yield_lag_1_scaled", "Yield_lag_2_scaled", "Yield_lag_3_scaled",
    "Yield_roll_mean3_scaled", "Yield_roll_std3_scaled", "Yield_yoy_change_scaled",
    # Interaction
    "Temp_x_Rain_scaled", "Rain_roll_mean3_scaled",
    # Categorical (encoded)
    "Crop_Enc", "Region_Enc", "State_Name_Enc",
]

# Fallback: use raw columns if scaled not present
FALLBACK_FEATURES = [
    "Area", "Production", "Fertilizer", "Pesticide",
    "Annual_Rainfall", "Temperature", "Humidity",
    "N", "P", "K", "pH", "NPK_total",
    "Yield_lag_1", "Yield_lag_2", "Yield_lag_3",
    "Yield_roll_mean3", "Yield_roll_std3", "Yield_yoy_change",
    "Temp_x_Rain", "Rain_roll_mean3",
    "Crop_Enc", "Region_Enc", "State_Name_Enc",
]

FEATURES = [f for f in CANDIDATE_FEATURES if f in train_df.columns]
if len(FEATURES) < 5:
    FEATURES = [f for f in FALLBACK_FEATURES if f in train_df.columns]

print(f"  Features selected ({len(FEATURES)}):")
for f in FEATURES:
    print(f"    • {f}")
print()

X_train = train_df[FEATURES].fillna(0)
y_train = train_df[TARGET]
X_test  = test_df[FEATURES].fillna(0)
y_test  = test_df[TARGET]

# -----------------------------------------------------------------------
# STEP 3: BASELINE RANDOM FOREST
# -----------------------------------------------------------------------
print("=" * 65)
print("STEP 3: Train Baseline Random Forest")
print("=" * 65)

rf_base = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_base.fit(X_train, y_train)

y_pred_train = rf_base.predict(X_train)
y_pred_test  = rf_base.predict(X_test)

print_metrics("Train (Baseline)", y_train, y_pred_train)
print_metrics("Test  (Baseline)", y_test,  y_pred_test)

# -----------------------------------------------------------------------
# STEP 4: HYPERPARAMETER TUNING (GridSearchCV)
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 4: Hyperparameter Tuning (GridSearchCV)")
print("=" * 65)

param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
    "max_features":      ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"\n  Best params: {best_params}")

# -----------------------------------------------------------------------
# STEP 5: TUNED RANDOM FOREST
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 5: Train Tuned Random Forest")
print("=" * 65)

rf_tuned = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
rf_tuned.fit(X_train, y_train)

y_pred_train_t = rf_tuned.predict(X_train)
y_pred_test_t  = rf_tuned.predict(X_test)

print_metrics("Train (Tuned)", y_train, y_pred_train_t)
print_metrics("Test  (Tuned)", y_test,  y_pred_test_t)

# -----------------------------------------------------------------------
# STEP 6: CROSS-VALIDATION
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 6: 5-Fold Cross-Validation")
print("=" * 65)

cv_scores = cross_val_score(
    rf_tuned, X_train, y_train,
    cv=5, scoring="r2", n_jobs=-1
)
print(f"  CV R² scores : {np.round(cv_scores, 4)}")
print(f"  Mean R²      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -----------------------------------------------------------------------
# STEP 7: REGION-WISE EVALUATION
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 7: Region-Wise Performance")
print("=" * 65)

region_results = {}
for region in sorted(test_df["Region"].unique()):
    mask   = test_df["Region"] == region
    X_r    = test_df.loc[mask, FEATURES].fillna(0)
    y_r    = test_df.loc[mask, TARGET]
    y_pred_r = rf_tuned.predict(X_r)
    region_results[region] = {
        "RMSE": round(rmse(y_r, y_pred_r), 4),
        "MAE":  round(mean_absolute_error(y_r, y_pred_r), 4),
        "R2":   round(r2_score(y_r, y_pred_r), 4),
        "n":    int(mask.sum())
    }
    print(f"  {region:8s} → RMSE={region_results[region]['RMSE']:.4f}  "
          f"MAE={region_results[region]['MAE']:.4f}  "
          f"R²={region_results[region]['R2']:.4f}  (n={region_results[region]['n']})")

# -----------------------------------------------------------------------
# STEP 8: FEATURE IMPORTANCE
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 8: Feature Importance")
print("=" * 65)

importances = pd.Series(rf_tuned.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)
print(importances.to_string())

# -----------------------------------------------------------------------
# STEP 9: SAVE MODEL & PREDICTIONS
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 9: Save Model & Predictions")
print("=" * 65)

# Save model
joblib.dump(rf_tuned, f"{OUTPUT_DIR}/rf_model.pkl")
print("  Saved rf_model.pkl")

# Save predictions
test_df["RF_Predicted_Yield"] = y_pred_test_t
test_df[["State_Name","Crop","Crop_Year","Region","Yield","RF_Predicted_Yield"]].to_csv(
    f"{OUTPUT_DIR}/rf_predictions.csv", index=False)
print("  Saved rf_predictions.csv")

# Save metrics
metrics = {
    "baseline": {
        "train": {"RMSE": round(rmse(y_train, y_pred_train),4),
                  "MAE":  round(mean_absolute_error(y_train, y_pred_train),4),
                  "R2":   round(r2_score(y_train, y_pred_train),4)},
        "test":  {"RMSE": round(rmse(y_test, y_pred_test),4),
                  "MAE":  round(mean_absolute_error(y_test, y_pred_test),4),
                  "R2":   round(r2_score(y_test, y_pred_test),4)},
    },
    "tuned": {
        "train": {"RMSE": round(rmse(y_train, y_pred_train_t),4),
                  "MAE":  round(mean_absolute_error(y_train, y_pred_train_t),4),
                  "R2":   round(r2_score(y_train, y_pred_train_t),4)},
        "test":  {"RMSE": round(rmse(y_test, y_pred_test_t),4),
                  "MAE":  round(mean_absolute_error(y_test, y_pred_test_t),4),
                  "R2":   round(r2_score(y_test, y_pred_test_t),4)},
    },
    "cross_val_r2": {"mean": round(cv_scores.mean(),4), "std": round(cv_scores.std(),4)},
    "region_wise":  region_results,
    "best_params":  best_params,
}
with open(f"{OUTPUT_DIR}/rf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("  Saved rf_metrics.json")

# -----------------------------------------------------------------------
# STEP 10: PLOTS
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 10: Plots")
print("=" * 65)

# Plot 1: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test_t, alpha=0.4, color="steelblue", edgecolors="none", s=20)
lims = [min(y_test.min(), y_pred_test_t.min()), max(y_test.max(), y_pred_test_t.max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Yield"); plt.ylabel("Predicted Yield")
plt.title(f"RF: Actual vs Predicted  (R²={r2_score(y_test, y_pred_test_t):.4f})")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rf_actual_vs_pred.png", dpi=150); plt.close()
print("  Saved rf_actual_vs_pred.png")

# Plot 2: Feature Importance (top 15)
plt.figure(figsize=(10, 6))
top15 = importances.head(15)
sns.barplot(x=top15.values, y=top15.index, palette="viridis")
plt.title("Random Forest — Top 15 Feature Importances")
plt.xlabel("Importance Score"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rf_feature_importance.png", dpi=150); plt.close()
print("  Saved rf_feature_importance.png")

# Plot 3: Region-wise R² bar chart
plt.figure(figsize=(8, 5))
regions = list(region_results.keys())
r2_vals = [region_results[r]["R2"] for r in regions]
colors  = ["#2ecc71" if v >= 0.8 else "#f39c12" if v >= 0.6 else "#e74c3c" for v in r2_vals]
bars = plt.bar(regions, r2_vals, color=colors)
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.ylim(0, 1.1); plt.title("Random Forest R² Score by Region")
plt.ylabel("R² Score"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rf_region_r2.png", dpi=150); plt.close()
print("  Saved rf_region_r2.png")

# Plot 4: Residuals distribution
residuals = y_test.values - y_pred_test_t
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=40, color="steelblue", edgecolor="white")
plt.title("Residuals Distribution"); plt.xlabel("Residual")
plt.subplot(1, 2, 2)
plt.scatter(y_pred_test_t, residuals, alpha=0.3, s=10, color="steelblue")
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Fitted"); plt.xlabel("Fitted"); plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rf_residuals.png", dpi=150); plt.close()
print("  Saved rf_residuals.png")

# Plot 5: CV scores
plt.figure(figsize=(7, 4))
plt.bar(range(1, 6), cv_scores, color="steelblue")
plt.axhline(cv_scores.mean(), color="red", linestyle="--",
            label=f"Mean R² = {cv_scores.mean():.4f}")
plt.xticks(range(1, 6), [f"Fold {i}" for i in range(1, 6)])
plt.title("5-Fold Cross-Validation R² Scores"); plt.ylabel("R²")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rf_cv_scores.png", dpi=150); plt.close()
print("  Saved rf_cv_scores.png")

# -----------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("✅ PHASE 2 COMPLETE — RANDOM FOREST")
print("=" * 65)
t = metrics["tuned"]
print(f"  Train  → RMSE={t['train']['RMSE']}  MAE={t['train']['MAE']}  R²={t['train']['R2']}")
print(f"  Test   → RMSE={t['test']['RMSE']}   MAE={t['test']['MAE']}   R²={t['test']['R2']}")
print(f"  CV R²  → {metrics['cross_val_r2']['mean']} ± {metrics['cross_val_r2']['std']}")
print(f"\n  Saved: rf_model.pkl, rf_predictions.csv, rf_metrics.json")
print(f"  Plots : rf_actual_vs_pred, rf_feature_importance, rf_region_r2, rf_residuals, rf_cv_scores")
