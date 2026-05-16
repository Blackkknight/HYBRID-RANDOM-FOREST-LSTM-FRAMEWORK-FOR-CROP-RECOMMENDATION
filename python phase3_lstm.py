"""
=======================================================================
PHASE 3: LSTM TEMPORAL LEARNING MODEL
Region-Aware Hybrid Deep Learning for Crop Yield Prediction
=======================================================================
Run after Phase 1 & 2. Requires:
  outputs/train_data.csv
  outputs/test_data.csv
  outputs/rf_metrics.json
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing   import MinMaxScaler
import joblib

import tensorflow as tf
from tensorflow.keras.models    import Sequential, load_model
from tensorflow.keras.layers    import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
print(f"  Test  : {test_df.shape}\n")

# -----------------------------------------------------------------------
# STEP 2: DEFINE FEATURES
# -----------------------------------------------------------------------
print("=" * 65)
print("STEP 2: Define Feature Set")
print("=" * 65)

TARGET = "Yield"

CANDIDATE_FEATURES = [
    "Area_scaled", "Production_scaled", "Fertilizer_scaled", "Pesticide_scaled",
    "Annual_Rainfall_scaled", "Temperature_scaled", "Humidity_scaled",
    "N_scaled", "P_scaled", "K_scaled", "pH_scaled", "NPK_total_scaled",
    "Yield_lag_1_scaled", "Yield_lag_2_scaled", "Yield_lag_3_scaled",
    "Yield_roll_mean3_scaled", "Yield_roll_std3_scaled", "Yield_yoy_change_scaled",
    "Temp_x_Rain_scaled", "Rain_roll_mean3_scaled",
    "Crop_Enc", "Region_Enc", "State_Name_Enc",
]
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

# -----------------------------------------------------------------------
# STEP 3: BUILD SEQUENCES FOR LSTM
# -----------------------------------------------------------------------
print("=" * 65)
print("STEP 3: Build Time-Series Sequences (Sliding Window)")
print("=" * 65)

SEQUENCE_LEN = 3   # use 3 consecutive years to predict next year

def build_sequences(df, features, target, seq_len):
    """
    For each (State, Crop) group, build overlapping windows of
    seq_len years → predict next year's yield.
    """
    X_list, y_list, meta_list = [], [], []

    for (state, crop), group in df.sort_values("Crop_Year").groupby(["State_Name", "Crop"]):
        vals   = group[features].fillna(0).values
        labels = group[target].values
        region = group["Region"].iloc[0]

        for i in range(len(vals) - seq_len):
            X_list.append(vals[i : i + seq_len])          # shape (seq_len, n_features)
            y_list.append(labels[i + seq_len])             # next year's yield
            meta_list.append({
                "State_Name": state, "Crop": crop,
                "Region": region,
                "Crop_Year": group["Crop_Year"].iloc[i + seq_len]
            })

    return np.array(X_list), np.array(y_list), pd.DataFrame(meta_list)

X_train_seq, y_train_seq, meta_train = build_sequences(train_df, FEATURES, TARGET, SEQUENCE_LEN)
X_test_seq,  y_test_seq,  meta_test  = build_sequences(test_df,  FEATURES, TARGET, SEQUENCE_LEN)

print(f"  Sequence length : {SEQUENCE_LEN} years")
print(f"  X_train shape   : {X_train_seq.shape}  (samples, timesteps, features)")
print(f"  X_test  shape   : {X_test_seq.shape}")
print(f"  y_train shape   : {y_train_seq.shape}")
print(f"  y_test  shape   : {y_test_seq.shape}\n")

# Scale target separately for LSTM
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train_seq.reshape(-1, 1)).ravel()
y_test_scaled  = y_scaler.transform(y_test_seq.reshape(-1, 1)).ravel()

# -----------------------------------------------------------------------
# STEP 4: BUILD LSTM MODEL
# -----------------------------------------------------------------------
print("=" * 65)
print("STEP 4: Build LSTM Architecture")
print("=" * 65)

n_features = X_train_seq.shape[2]

model = Sequential([
    # Layer 1 — LSTM with return sequences for stacking
    LSTM(128, return_sequences=True,
         input_shape=(SEQUENCE_LEN, n_features)),
    BatchNormalization(),
    Dropout(0.3),

    # Layer 2 — LSTM
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    # Layer 3 — LSTM (final, no return sequences)
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),

    # Dense output layers
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(16, activation="relu"),
    Dense(1)   # regression output
], name="CropYield_LSTM")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="huber",          # robust to outliers vs MSE
    metrics=["mae"]
)

model.summary()

# -----------------------------------------------------------------------
# STEP 5: TRAIN MODEL
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 5: Train LSTM Model")
print("=" * 65)

callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=15,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=7, min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        filepath=f"{OUTPUT_DIR}/lstm_best.keras",
        monitor="val_loss", save_best_only=True, verbose=0
    ),
]

history = model.fit(
    X_train_seq, y_train_scaled,
    epochs          = 100,
    batch_size      = 32,
    validation_split= 0.15,
    callbacks       = callbacks,
    verbose         = 1,
)

print(f"\n  Training stopped at epoch {len(history.history['loss'])}")

# -----------------------------------------------------------------------
# STEP 6: EVALUATE
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 6: Evaluate LSTM Model")
print("=" * 65)

y_pred_train_scaled = model.predict(X_train_seq, verbose=0).ravel()
y_pred_test_scaled  = model.predict(X_test_seq,  verbose=0).ravel()

# Inverse transform back to original yield scale
y_pred_train_lstm = y_scaler.inverse_transform(y_pred_train_scaled.reshape(-1,1)).ravel()
y_pred_test_lstm  = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1,1)).ravel()

print_metrics("Train (LSTM)", y_train_seq, y_pred_train_lstm)
print_metrics("Test  (LSTM)", y_test_seq,  y_pred_test_lstm)

# -----------------------------------------------------------------------
# STEP 7: REGION-WISE EVALUATION
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 7: Region-Wise LSTM Performance")
print("=" * 65)

region_results_lstm = {}
for region in sorted(meta_test["Region"].unique()):
    mask = meta_test["Region"] == region
    yt   = y_test_seq[mask]
    yp   = y_pred_test_lstm[mask]
    region_results_lstm[region] = {
        "RMSE": round(rmse(yt, yp), 4),
        "MAE":  round(mean_absolute_error(yt, yp), 4),
        "R2":   round(r2_score(yt, yp), 4),
        "n":    int(mask.sum())
    }
    r = region_results_lstm[region]
    print(f"  {region:8s} → RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  R²={r['R2']:.4f}  (n={r['n']})")

# -----------------------------------------------------------------------
# STEP 8: SAVE MODEL & PREDICTIONS
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 8: Save Model & Predictions")
print("=" * 65)

model.save(f"{OUTPUT_DIR}/lstm_model.keras")
joblib.dump(y_scaler, f"{OUTPUT_DIR}/lstm_y_scaler.pkl")
print("  Saved lstm_model.keras")
print("  Saved lstm_y_scaler.pkl")

meta_test["LSTM_Predicted_Yield"] = y_pred_test_lstm
meta_test["Actual_Yield"]         = y_test_seq
meta_test.to_csv(f"{OUTPUT_DIR}/lstm_predictions.csv", index=False)
print("  Saved lstm_predictions.csv")

# Save metrics
lstm_metrics = {
    "train": {
        "RMSE": round(rmse(y_train_seq, y_pred_train_lstm), 4),
        "MAE":  round(mean_absolute_error(y_train_seq, y_pred_train_lstm), 4),
        "R2":   round(r2_score(y_train_seq, y_pred_train_lstm), 4),
    },
    "test": {
        "RMSE": round(rmse(y_test_seq, y_pred_test_lstm), 4),
        "MAE":  round(mean_absolute_error(y_test_seq, y_pred_test_lstm), 4),
        "R2":   round(r2_score(y_test_seq, y_pred_test_lstm), 4),
    },
    "region_wise": region_results_lstm,
    "epochs_run":  len(history.history["loss"]),
}
with open(f"{OUTPUT_DIR}/lstm_metrics.json", "w") as f:
    json.dump(lstm_metrics, f, indent=2)
print("  Saved lstm_metrics.json")

# -----------------------------------------------------------------------
# STEP 9: PLOTS
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 9: Plots")
print("=" * 65)

# Plot 1: Training history
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("LSTM Training & Validation Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Huber Loss")
axes[0].legend()
axes[1].plot(history.history["mae"],     label="Train MAE")
axes[1].plot(history.history["val_mae"], label="Val MAE")
axes[1].set_title("LSTM Training & Validation MAE")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/lstm_training_history.png", dpi=150); plt.close()
print("  Saved lstm_training_history.png")

# Plot 2: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_seq, y_pred_test_lstm, alpha=0.4, color="darkorange", s=20)
lims = [min(y_test_seq.min(), y_pred_test_lstm.min()),
        max(y_test_seq.max(), y_pred_test_lstm.max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Yield"); plt.ylabel("Predicted Yield")
plt.title(f"LSTM: Actual vs Predicted  (R²={r2_score(y_test_seq, y_pred_test_lstm):.4f})")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/lstm_actual_vs_pred.png", dpi=150); plt.close()
print("  Saved lstm_actual_vs_pred.png")

# Plot 3: Region-wise R² comparison with RF
plt.figure(figsize=(10, 5))
regions  = sorted(region_results_lstm.keys())
lstm_r2  = [region_results_lstm[r]["R2"] for r in regions]

# Load RF region results if available
rf_r2 = []
if os.path.exists(f"{OUTPUT_DIR}/rf_metrics.json"):
    with open(f"{OUTPUT_DIR}/rf_metrics.json") as f:
        rf_met = json.load(f)
    rf_r2 = [rf_met["region_wise"].get(r, {}).get("R2", 0) for r in regions]

x = np.arange(len(regions))
w = 0.35
plt.bar(x - w/2, rf_r2  if rf_r2 else lstm_r2, w, label="Random Forest", color="steelblue")
plt.bar(x + w/2, lstm_r2, w, label="LSTM", color="darkorange")
plt.xticks(x, regions); plt.ylim(0, 1.1)
plt.ylabel("R² Score"); plt.title("Region-Wise R²: RF vs LSTM")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/lstm_vs_rf_region_r2.png", dpi=150); plt.close()
print("  Saved lstm_vs_rf_region_r2.png")

# Plot 4: Predicted yield over time for one crop-region sample
sample_crop   = sorted(meta_test["Crop"].unique())[0]
sample_region = sorted(meta_test["Region"].unique())[0]
mask_s = (meta_test["Crop"] == sample_crop) & (meta_test["Region"] == sample_region)
if mask_s.sum() > 2:
    sub = meta_test[mask_s].sort_values("Crop_Year")
    plt.figure(figsize=(10, 4))
    plt.plot(sub["Crop_Year"], sub["Actual_Yield"],         "b-o", label="Actual")
    plt.plot(sub["Crop_Year"], sub["LSTM_Predicted_Yield"], "r--s", label="LSTM Predicted")
    plt.title(f"Yield Over Time — {sample_crop} ({sample_region} Region)")
    plt.xlabel("Year"); plt.ylabel("Yield (kg/ha)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lstm_yield_over_time.png", dpi=150); plt.close()
    print("  Saved lstm_yield_over_time.png")

# Plot 5: Residuals
residuals = y_test_seq - y_pred_test_lstm
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=40, color="darkorange", edgecolor="white")
plt.title("LSTM Residuals Distribution"); plt.xlabel("Residual")
plt.subplot(1, 2, 2)
plt.scatter(y_pred_test_lstm, residuals, alpha=0.3, s=10, color="darkorange")
plt.axhline(0, color="red", linestyle="--")
plt.title("LSTM Residuals vs Fitted"); plt.xlabel("Fitted"); plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/lstm_residuals.png", dpi=150); plt.close()
print("  Saved lstm_residuals.png")

# -----------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------
print("\n" + "=" * 65)
print("✅ PHASE 3 COMPLETE — LSTM MODEL")
print("=" * 65)
print(f"  Train → RMSE={lstm_metrics['train']['RMSE']}  "
      f"MAE={lstm_metrics['train']['MAE']}  R²={lstm_metrics['train']['R2']}")
print(f"  Test  → RMSE={lstm_metrics['test']['RMSE']}  "
      f"MAE={lstm_metrics['test']['MAE']}  R²={lstm_metrics['test']['R2']}")
print(f"  Epochs run : {lstm_metrics['epochs_run']}")
print(f"\n  Saved: lstm_model.keras, lstm_predictions.csv, lstm_metrics.json")
print(f"  Plots : training_history, actual_vs_pred, vs_rf_region_r2,")
print(f"          yield_over_time, residuals")
print(f"\n  👉 Next: Run phase4_fusion.py  (RF + LSTM → Final Hybrid Model)")