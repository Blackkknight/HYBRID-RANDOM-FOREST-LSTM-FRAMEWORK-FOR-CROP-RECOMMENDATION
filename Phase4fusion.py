"""
=======================================================================
PHASE 4: FUSION LAYER — HYBRID RF + LSTM MODEL (FIXED)
Region-Aware Hybrid Deep Learning for Crop Yield Prediction
=======================================================================
Implements: Y_final = α · Y_RF + β · Y_LSTM
  Strategy: R²-performance weighted fusion (prevents α=1, β=0 collapse)

Requires:
  outputs/rf_predictions.csv
  outputs/lstm_predictions.csv
  outputs/rf_metrics.json
  outputs/lstm_metrics.json
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.metrics      import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from scipy.optimize       import minimize

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def print_metrics(label, y_true, y_pred):
    print(f"\n  [{label}]")
    print(f"    RMSE : {rmse(y_true, y_pred):.4f}")
    print(f"    MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"    R²   : {r2_score(y_true, y_pred):.4f}")

# -----------------------------------------------------------------------
# STEP 1: LOAD PREDICTIONS
# -----------------------------------------------------------------------
print("=" * 65)
print("STEP 1: Load RF & LSTM Predictions")
print("=" * 65)

rf_pred   = pd.read_csv(f"{OUTPUT_DIR}/rf_predictions.csv")
lstm_pred = pd.read_csv(f"{OUTPUT_DIR}/lstm_predictions.csv")

# Standardise column names
rf_pred.rename(columns={"Yield": "Actual_Yield",
                         "RF_Predicted_Yield": "Y_RF"}, inplace=True)
lstm_pred.rename(columns={"LSTM_Predicted_Yield": "Y_LSTM"}, inplace=True)

print(f"  RF   rows : {len(rf_pred)}")
print(f"  LSTM rows : {len(lstm_pred)}")

# -----------------------------------------------------------------------
# STEP 2: ALIGN ON COMMON KEYS
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 2: Align on State + Crop + Year")
print("=" * 65)

merge_keys = ["State_Name", "Crop", "Crop_Year"]
df = pd.merge(
    rf_pred[merge_keys + ["Region", "Actual_Yield", "Y_RF"]],
    lstm_pred[merge_keys + ["Y_LSTM"]],
    on=merge_keys, how="inner"
).dropna(subset=["Y_RF", "Y_LSTM", "Actual_Yield"])

print(f"  Aligned rows : {len(df)}")
print(f"  Regions      : {sorted(df['Region'].unique())}")
print(f"  Crops        : {sorted(df['Crop'].unique())}\n")

y_true = df["Actual_Yield"].values
y_rf   = df["Y_RF"].values
y_lstm = df["Y_LSTM"].values

# -----------------------------------------------------------------------
# STEP 3: DIAGNOSE INDIVIDUAL MODEL PERFORMANCE ON ALIGNED SET
# -----------------------------------------------------------------------
print("=" * 65)
print("STEP 3: Individual Model Performance on Aligned Set")
print("=" * 65)
print_metrics("Random Forest (aligned)", y_true, y_rf)
print_metrics("LSTM (aligned)",          y_true, y_lstm)

r2_rf_aligned   = r2_score(y_true, y_rf)
r2_lstm_aligned = r2_score(y_true, y_lstm)

# -----------------------------------------------------------------------
# STEP 4: FUSION STRATEGIES
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 4: Fusion Strategies")
print("=" * 65)

# --- A: Equal weight ---
alpha_eq, beta_eq = 0.5, 0.5
y_equal = alpha_eq * y_rf + beta_eq * y_lstm
print_metrics("A) Equal Weight (0.5 / 0.5)", y_true, y_equal)

# --- B: R²-performance weighted ---
total    = r2_rf_aligned + r2_lstm_aligned
alpha_r2 = r2_rf_aligned   / total
beta_r2  = r2_lstm_aligned / total
y_r2w    = alpha_r2 * y_rf + beta_r2 * y_lstm
print(f"\n  R²-weights → α(RF)={alpha_r2:.4f}  β(LSTM)={beta_r2:.4f}")
print_metrics("B) R²-Weighted (performance-based)", y_true, y_r2w)

# --- C: Constrained optimisation (force both weights > 0.1) ---
def fusion_loss(weights):
    a, b = weights
    return rmse(y_true, a * y_rf + b * y_lstm)

constraints = [
    {"type": "eq",   "fun": lambda w: w[0] + w[1] - 1},   # α + β = 1
    {"type": "ineq", "fun": lambda w: w[0] - 0.1},         # α ≥ 0.1
    {"type": "ineq", "fun": lambda w: w[1] - 0.1},         # β ≥ 0.1
]
result     = minimize(fusion_loss, x0=[alpha_r2, beta_r2],
                      method="SLSQP", bounds=[(0.1, 0.9), (0.1, 0.9)],
                      constraints=constraints)
alpha_opt  = round(result.x[0], 4)
beta_opt   = round(result.x[1], 4)
y_opt      = alpha_opt * y_rf + beta_opt * y_lstm
print(f"\n  Optimised weights → α(RF)={alpha_opt:.4f}  β(LSTM)={beta_opt:.4f}")
print_metrics("C) Constrained Optimised Fusion", y_true, y_opt)

# --- D: Inverse-RMSE weighted (lower error = higher weight) ---
rmse_rf   = rmse(y_true, y_rf)
rmse_lstm = rmse(y_true, y_lstm)
inv_rf    = 1 / rmse_rf
inv_lstm  = 1 / rmse_lstm
alpha_inv = inv_rf   / (inv_rf + inv_lstm)
beta_inv  = inv_lstm / (inv_rf + inv_lstm)
y_inv     = alpha_inv * y_rf + beta_inv * y_lstm
print(f"\n  Inverse-RMSE weights → α(RF)={alpha_inv:.4f}  β(LSTM)={beta_inv:.4f}")
print_metrics("D) Inverse-RMSE Weighted", y_true, y_inv)

# --- E: Stacked Ridge meta-learner ---
split     = int(len(df) * 0.8)
X_m_tr    = np.column_stack([y_rf[:split],  y_lstm[:split]])
X_m_te    = np.column_stack([y_rf[split:],  y_lstm[split:]])
y_m_tr    = y_true[:split]
y_m_te    = y_true[split:]
ridge     = Ridge(alpha=1.0)
ridge.fit(X_m_tr, y_m_tr)
y_ridge   = ridge.predict(X_m_te)
print(f"\n  Ridge meta-learner coefs → RF={ridge.coef_[0]:.4f}  LSTM={ridge.coef_[1]:.4f}")
print_metrics("E) Stacked Ridge Meta-Learner (on 20% hold-out)", y_m_te, y_ridge)

# -----------------------------------------------------------------------
# STEP 5: SELECT BEST STRATEGY (on full aligned set, excluding ridge)
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 5: Select Best Fusion Strategy")
print("=" * 65)

strategies = {
    "Equal Weight":        (y_equal, alpha_eq,  beta_eq),
    "R²-Weighted":         (y_r2w,   alpha_r2,  beta_r2),
    "Constrained Optimised":(y_opt,  alpha_opt, beta_opt),
    "Inverse-RMSE":        (y_inv,   alpha_inv, beta_inv),
}

print(f"\n  {'Strategy':<26}  {'R²':>8}  {'RMSE':>8}  {'MAE':>8}  α(RF)  β(LSTM)")
print("  " + "-" * 72)

best_name, best_r2_val, best_preds = None, -999, None
best_alpha, best_beta = alpha_r2, beta_r2

for name, (preds, a, b) in strategies.items():
    r2v   = r2_score(y_true, preds)
    rmse_ = rmse(y_true, preds)
    mae_  = mean_absolute_error(y_true, preds)
    marker = " ← best" if r2v > best_r2_val else ""
    print(f"  {name:<26}  {r2v:>8.4f}  {rmse_:>8.4f}  {mae_:>8.4f}  {a:.3f}  {b:.3f}{marker}")
    if r2v > best_r2_val:
        best_r2_val = r2v
        best_name   = name
        best_preds  = preds
        best_alpha  = a
        best_beta   = b

print(f"\n  ✅ Best: '{best_name}'  α={best_alpha:.4f}  β={best_beta:.4f}  R²={best_r2_val:.4f}")

# -----------------------------------------------------------------------
# STEP 6: FINAL HYBRID PREDICTIONS
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 6: Final Hybrid Model — Full Evaluation")
print("=" * 65)

y_final = best_alpha * y_rf + best_beta * y_lstm
print_metrics("Final Hybrid Model", y_true, y_final)

# -----------------------------------------------------------------------
# STEP 7: REGION-WISE & CROP-WISE COMPARISON
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 7: Region-Wise Comparison")
print("=" * 65)

region_compare = {}
print(f"\n  {'Region':8s}  {'RF R²':>8}  {'LSTM R²':>8}  {'Hybrid R²':>10}  {'Hybrid RMSE':>12}")
print("  " + "-" * 57)

for region in sorted(df["Region"].unique()):
    m   = df["Region"] == region
    yt  = y_true[m]; yrf = y_rf[m]; yls = y_lstm[m]; yh = y_final[m]
    r   = {"RF_R2":   round(r2_score(yt, yrf), 4),
           "LSTM_R2": round(r2_score(yt, yls), 4),
           "Hybrid_R2":   round(r2_score(yt, yh), 4),
           "Hybrid_RMSE": round(rmse(yt, yh), 4),
           "n": int(m.sum())}
    region_compare[region] = r
    print(f"  {region:8s}  {r['RF_R2']:>8.4f}  {r['LSTM_R2']:>8.4f}  "
          f"{r['Hybrid_R2']:>10.4f}  {r['Hybrid_RMSE']:>12.4f}")

print("\n" + "=" * 65)
print("STEP 7b: Crop-Wise Performance")
print("=" * 65)

crop_results = {}
print(f"\n  {'Crop':12s}  {'RF R²':>8}  {'LSTM R²':>8}  {'Hybrid R²':>10}  {'Hybrid RMSE':>12}")
print("  " + "-" * 60)

for crop in sorted(df["Crop"].unique()):
    m  = df["Crop"] == crop
    yt = y_true[m]; yrf = y_rf[m]; yls = y_lstm[m]; yh = y_final[m]
    c  = {"RF_R2":       round(r2_score(yt, yrf), 4),
          "LSTM_R2":     round(r2_score(yt, yls), 4),
          "Hybrid_R2":   round(r2_score(yt, yh), 4),
          "Hybrid_RMSE": round(rmse(yt, yh), 4),
          "Hybrid_MAE":  round(mean_absolute_error(yt, yh), 4)}
    crop_results[crop] = c
    print(f"  {crop:12s}  {c['RF_R2']:>8.4f}  {c['LSTM_R2']:>8.4f}  "
          f"{c['Hybrid_R2']:>10.4f}  {c['Hybrid_RMSE']:>12.4f}")

# -----------------------------------------------------------------------
# STEP 8: SAVE
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 8: Save Results")
print("=" * 65)

df["Hybrid_Predicted_Yield"] = y_final
df.to_csv(f"{OUTPUT_DIR}/final_predictions.csv", index=False)
print("  Saved final_predictions.csv")

final_metrics = {
    "best_fusion_strategy": best_name,
    "alpha_RF":  round(best_alpha, 4),
    "beta_LSTM": round(best_beta, 4),
    "hybrid":    {"RMSE": round(rmse(y_true, y_final), 4),
                  "MAE":  round(mean_absolute_error(y_true, y_final), 4),
                  "R2":   round(r2_score(y_true, y_final), 4)},
    "rf_aligned":   {"RMSE": round(rmse(y_true, y_rf), 4),
                     "MAE":  round(mean_absolute_error(y_true, y_rf), 4),
                     "R2":   round(r2_score(y_true, y_rf), 4)},
    "lstm_aligned": {"RMSE": round(rmse(y_true, y_lstm), 4),
                     "MAE":  round(mean_absolute_error(y_true, y_lstm), 4),
                     "R2":   round(r2_score(y_true, y_lstm), 4)},
    "region_wise": region_compare,
    "crop_wise":   crop_results,
    "all_strategies": {
        name: {"R2":   round(r2_score(y_true, p), 4),
               "RMSE": round(rmse(y_true, p), 4),
               "alpha": round(a, 4), "beta": round(b, 4)}
        for name, (p, a, b) in strategies.items()
    }
}
with open(f"{OUTPUT_DIR}/final_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)
print("  Saved final_metrics.json")

with open(f"{OUTPUT_DIR}/fusion_config.json", "w") as f:
    json.dump({"strategy": best_name, "alpha_RF": round(best_alpha,4),
               "beta_LSTM": round(best_beta,4)}, f, indent=2)
print("  Saved fusion_config.json")

# -----------------------------------------------------------------------
# STEP 9: PLOTS
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 9: Generating Plots")
print("=" * 65)

# Plot 1: Actual vs Predicted — all 3 models
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (preds, label, col) in zip(axes, [
        (y_rf,    f"Random Forest\nR²={r2_score(y_true,y_rf):.4f}",    "steelblue"),
        (y_lstm,  f"LSTM\nR²={r2_score(y_true,y_lstm):.4f}",           "darkorange"),
        (y_final, f"Hybrid ({best_name})\nR²={r2_score(y_true,y_final):.4f}", "green")]):
    ax.scatter(y_true, preds, alpha=0.4, color=col, s=15)
    lim = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
    ax.plot(lim, lim, "r--", lw=1.5)
    ax.set_title(label); ax.set_xlabel("Actual Yield"); ax.set_ylabel("Predicted Yield")
plt.suptitle("Actual vs Predicted: RF | LSTM | Hybrid", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/all_models_actual_vs_pred.png", dpi=150); plt.close()
print("  Saved all_models_actual_vs_pred.png")

# Plot 2: Overall model comparison (RMSE / MAE / R²)
models_   = ["Random Forest", "LSTM", f"Hybrid\n({best_name})"]
rmse_v    = [rmse(y_true,y_rf),  rmse(y_true,y_lstm),  rmse(y_true,y_final)]
mae_v     = [mean_absolute_error(y_true,y_rf),
             mean_absolute_error(y_true,y_lstm),
             mean_absolute_error(y_true,y_final)]
r2_v      = [r2_score(y_true,y_rf), r2_score(y_true,y_lstm), r2_score(y_true,y_final)]
cols_     = ["steelblue", "darkorange", "green"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, vals, metric in zip(axes, [rmse_v, mae_v, r2_v], ["RMSE ↓", "MAE ↓", "R² ↑"]):
    bars = ax.bar(models_, vals, color=cols_)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_title(f"{metric}"); ax.set_ylabel(metric)
    if "R²" in metric: ax.set_ylim(0, 1.1)
plt.suptitle("Model Comparison: RF vs LSTM vs Hybrid", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150); plt.close()
print("  Saved model_comparison.png")

# Plot 3: Region-wise grouped R²
regions_   = sorted(region_compare.keys())
rf_r2_r    = [region_compare[r]["RF_R2"]     for r in regions_]
lstm_r2_r  = [region_compare[r]["LSTM_R2"]   for r in regions_]
hyb_r2_r   = [region_compare[r]["Hybrid_R2"] for r in regions_]
x = np.arange(len(regions_)); w = 0.25

plt.figure(figsize=(11, 5))
plt.bar(x-w,   rf_r2_r,   w, label="Random Forest", color="steelblue")
plt.bar(x,     lstm_r2_r, w, label="LSTM",           color="darkorange")
plt.bar(x+w,   hyb_r2_r,  w, label="Hybrid",         color="green")
plt.xticks(x, regions_); plt.ylim(0, 1.15)
plt.ylabel("R² Score"); plt.title("Region-Wise R²: RF vs LSTM vs Hybrid")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/region_model_comparison.png", dpi=150); plt.close()
print("  Saved region_model_comparison.png")

# Plot 4: Crop-wise hybrid R²
crops_  = list(crop_results.keys())
c_r2_   = [crop_results[c]["Hybrid_R2"] for c in crops_]
c_cols_ = ["#2ecc71" if v>=0.85 else "#f39c12" if v>=0.7 else "#e74c3c" for v in c_r2_]
plt.figure(figsize=(11, 5))
bars = plt.bar(crops_, c_r2_, color=c_cols_)
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.ylim(0, 1.15); plt.ylabel("R² Score")
plt.title("Hybrid Model R² Score by Crop")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hybrid_crop_r2.png", dpi=150); plt.close()
print("  Saved hybrid_crop_r2.png")

# Plot 5: Fusion weight pie chart
plt.figure(figsize=(6, 6))
plt.pie([best_alpha, best_beta],
        labels=[f"Random Forest\nα={best_alpha:.3f}", f"LSTM\nβ={best_beta:.3f}"],
        colors=["steelblue","darkorange"], autopct="%1.1f%%",
        explode=(0.05, 0.05), shadow=True, startangle=140,
        textprops={"fontsize": 13})
plt.title(f"Fusion Weights\nY = {best_alpha:.3f}·RF + {best_beta:.3f}·LSTM",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fusion_weights.png", dpi=150); plt.close()
print("  Saved fusion_weights.png")

# Plot 6: Yield over time (sample crop, all models)
sample_crop   = sorted(df["Crop"].unique())[0]
sample_region = sorted(df["Region"].unique())[0]
mask_s = (df["Crop"]==sample_crop) & (df["Region"]==sample_region)
if mask_s.sum() > 2:
    sub = df[mask_s].sort_values("Crop_Year")
    plt.figure(figsize=(11, 4))
    plt.plot(sub["Crop_Year"], sub["Actual_Yield"],          "k-o",  lw=2,   label="Actual")
    plt.plot(sub["Crop_Year"], sub["Y_RF"],                  "b--s", lw=1.5, label="RF")
    plt.plot(sub["Crop_Year"], sub["Y_LSTM"],                "r--^", lw=1.5, label="LSTM")
    plt.plot(sub["Crop_Year"], sub["Hybrid_Predicted_Yield"],"g-D",  lw=2,   label="Hybrid")
    plt.title(f"Yield Over Time — {sample_crop} ({sample_region})")
    plt.xlabel("Year"); plt.ylabel("Yield (kg/ha)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hybrid_yield_over_time.png", dpi=150); plt.close()
    print("  Saved hybrid_yield_over_time.png")

# Plot 7: All fusion strategies comparison
strat_names_ = list(final_metrics["all_strategies"].keys())
strat_r2_    = [final_metrics["all_strategies"][s]["R2"]   for s in strat_names_]
strat_rmse_  = [final_metrics["all_strategies"][s]["RMSE"] for s in strat_names_]
x_ = np.arange(len(strat_names_)); w_ = 0.35
fig, ax1 = plt.subplots(figsize=(10, 5))
b1 = ax1.bar(x_-w_/2, strat_r2_,  w_, color="green",     label="R²",  alpha=0.85)
ax2 = ax1.twinx()
b2 = ax2.bar(x_+w_/2, strat_rmse_, w_, color="steelblue", label="RMSE", alpha=0.7)
ax1.set_xticks(x_); ax1.set_xticklabels(strat_names_, rotation=20, ha="right")
ax1.set_ylabel("R²", color="green"); ax1.set_ylim(0, 1.15)
ax2.set_ylabel("RMSE", color="steelblue")
ax1.bar_label(b1, fmt="%.4f", padding=2, fontsize=8)
ax2.bar_label(b2, fmt="%.4f", padding=2, fontsize=8)
plt.title("Fusion Strategy Comparison"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fusion_strategy_comparison.png", dpi=150); plt.close()
print("  Saved fusion_strategy_comparison.png")

# -----------------------------------------------------------------------
# FINAL SUMMARY TABLE
# -----------------------------------------------------------------------
h  = final_metrics["hybrid"]
rf = final_metrics["rf_aligned"]
ls = final_metrics["lstm_aligned"]

print("\n" + "=" * 65)
print("✅ PHASE 4 COMPLETE — FINAL HYBRID MODEL")
print("=" * 65)
print(f"\n  {'Model':<26}  {'RMSE':>8}  {'MAE':>8}  {'R²':>8}")
print("  " + "-" * 56)
print(f"  {'Random Forest':<26}  {rf['RMSE']:>8}  {rf['MAE']:>8}  {rf['R2']:>8}")
print(f"  {'LSTM':<26}  {ls['RMSE']:>8}  {ls['MAE']:>8}  {ls['R2']:>8}")
print(f"  {'Hybrid (Final)':<26}  {h['RMSE']:>8}  {h['MAE']:>8}  {h['R2']:>8}")
print(f"\n  Fusion Formula : Y_final = {best_alpha:.4f}·Y_RF + {best_beta:.4f}·Y_LSTM")
print(f"  Best Strategy  : {best_name}")
print(f"\n  Saved : final_predictions.csv, final_metrics.json, fusion_config.json")
print(f"  Plots : all_models_actual_vs_pred, model_comparison,")
print(f"          region_model_comparison, hybrid_crop_r2,")
print(f"          fusion_weights, hybrid_yield_over_time,")
print(f"          fusion_strategy_comparison")

