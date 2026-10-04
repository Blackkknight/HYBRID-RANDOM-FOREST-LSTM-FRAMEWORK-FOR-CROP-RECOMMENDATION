"""
=======================================================================
RENDER.COM DEPLOYMENT — CROP YIELD PREDICTION
Region-Aware Hybrid Deep Learning for Crop Yield Prediction
=======================================================================
FastAPI app:
  GET  /          → serves index.html
  POST /predict   → returns yield prediction JSON
  GET  /options   → returns crops, states lists
  GET  /health    → health check
  GET  /regions   → region-wise model performance
=======================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------------------
OUTPUT_DIR = "outputs"

rf_model = None
lstm_model = None
lstm_scaler = None
label_encoders = {}
ALPHA, BETA = 1.0, 0.0

try:
    rf_model = joblib.load(f"{OUTPUT_DIR}/rf_model.pkl")
    print("✅ RF model loaded")
except Exception as e:
    print(f"❌ RF model error: {e}")

try:
    from tensorflow.keras.models import load_model as tf_load_model
    lstm_model  = tf_load_model(f"{OUTPUT_DIR}/lstm_model.keras")
    lstm_scaler = joblib.load(f"{OUTPUT_DIR}/lstm_y_scaler.pkl")
    print("✅ LSTM model loaded")
except Exception as e:
    print(f"ℹ️  LSTM not loaded (RF-only mode): {e}")

try:
    with open(f"{OUTPUT_DIR}/fusion_config.json") as f:
        cfg = json.load(f)
    ALPHA = cfg.get("alpha_RF",  0.9)
    BETA  = cfg.get("beta_LSTM", 0.1)
    print(f"✅ Fusion config: α={ALPHA}, β={BETA}")
except Exception as e:
    print(f"ℹ️  Fusion config not found, using RF-only: {e}")

try:
    with open(f"{OUTPUT_DIR}/label_encoders.json") as f:
        label_encoders = json.load(f)
    print("✅ Label encoders loaded")
except Exception as e:
    print(f"ℹ️  Label encoders not found: {e}")

try:
    with open(f"{OUTPUT_DIR}/rf_metrics.json") as f:
        rf_metrics = json.load(f)
except:
    rf_metrics = {}

# -----------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------
REGION_MAP = {
    "Uttar Pradesh":"North","Punjab":"North","Haryana":"North",
    "Rajasthan":"North","Himachal Pradesh":"North","Uttarakhand":"North",
    "Jammu And Kashmir":"North","Delhi":"North","Uttaranchal":"North",
    "Tamil Nadu":"South","Karnataka":"South","Kerala":"South",
    "Andhra Pradesh":"South","Telangana":"South","Puducherry":"South",
    "West Bengal":"East","Bihar":"East","Odisha":"East","Orissa":"East",
    "Jharkhand":"East","Assam":"East","Chhattisgarh":"East",
    "Meghalaya":"East","Manipur":"East","Nagaland":"East",
    "Tripura":"East","Arunachal Pradesh":"East","Mizoram":"East","Sikkim":"East",
    "Maharashtra":"West","Gujarat":"West","Madhya Pradesh":"West","Goa":"West",
}

CROP_DEFAULTS = {
    "Rice":      {"N":80,  "P":40, "K":40, "pH":6.5, "Temperature":23.0, "Humidity":82.0},
    "Wheat":     {"N":120, "P":60, "K":30, "pH":6.5, "Temperature":15.0, "Humidity":65.0},
    "Maize":     {"N":100, "P":50, "K":40, "pH":6.2, "Temperature":22.0, "Humidity":65.0},
    "Cotton":    {"N":120, "P":50, "K":30, "pH":6.5, "Temperature":25.0, "Humidity":60.0},
    "Sugarcane": {"N":100, "P":40, "K":50, "pH":6.5, "Temperature":27.0, "Humidity":80.0},
    "Jute":      {"N":80,  "P":40, "K":40, "pH":6.5, "Temperature":26.0, "Humidity":80.0},
    "Rapeseed":  {"N":80,  "P":40, "K":40, "pH":6.0, "Temperature":15.0, "Humidity":65.0},
    "Soyabean":  {"N":40,  "P":60, "K":40, "pH":6.5, "Temperature":28.0, "Humidity":65.0},
    "Bajra":     {"N":80,  "P":40, "K":20, "pH":6.0, "Temperature":28.0, "Humidity":55.0},
}

STATES = sorted(REGION_MAP.keys())
CROPS  = sorted(CROP_DEFAULTS.keys())

# -----------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------
app = FastAPI(
    title="Crop Yield Prediction API",
    description="Region-Aware Hybrid Deep Learning for Crop Yield Prediction — Indian Agro-Climatic Zones",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# -----------------------------------------------------------------------
class PredictRequest(BaseModel):
    state:       str   = Field(...,  example="West Bengal")
    crop:        str   = Field(...,  example="Rice")
    year:        int   = Field(2024, example=2024)
    area:        float = Field(...,  example=500.0,  description="Hectares")
    rainfall:    float = Field(...,  example=1200.0, description="Annual mm")
    fertilizer:  float = Field(50.0, example=50.0,  description="kg/ha")
    N:           Optional[float] = Field(None, description="Nitrogen kg/ha")
    P:           Optional[float] = Field(None, description="Phosphorus kg/ha")
    K:           Optional[float] = Field(None, description="Potassium kg/ha")
    pH:          Optional[float] = Field(None, description="Soil pH")
    temperature: Optional[float] = Field(None, description="°C")
    humidity:    Optional[float] = Field(None, description="%")

class PredictResponse(BaseModel):
    yield_kg_ha:       float
    production_tonnes: float
    region:            str
    confidence:        str
    model_used:        str
    inputs_used:       dict

# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
def encode_label(column: str, value: str) -> int:
    classes = label_encoders.get(column, [])
    val = str(value).strip().title()
    return classes.index(val) if val in classes else 0


def build_prediction(req: PredictRequest) -> PredictResponse:
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    state  = str(req.state).strip().title()
    crop   = str(req.crop).strip().title()
    region = REGION_MAP.get(state, "East")
    defs   = CROP_DEFAULTS.get(crop, list(CROP_DEFAULTS.values())[0])

    N    = req.N    if req.N    is not None else defs["N"]
    P    = req.P    if req.P    is not None else defs["P"]
    K    = req.K    if req.K    is not None else defs["K"]
    pH   = req.pH   if req.pH   is not None else defs["pH"]
    temp = req.temperature if req.temperature is not None else defs["Temperature"]
    hum  = req.humidity    if req.humidity    is not None else defs["Humidity"]

    npk      = N + P + K
    t_x_r    = temp * req.rainfall
    # Simple lag proxy — the actual trained lags aren't available at inference
    yield_est = req.rainfall * 2.5 + N * 1.2
    lag1, lag2, lag3 = yield_est * 0.95, yield_est * 0.90, yield_est * 0.85
    roll_m   = (lag1 + lag2 + lag3) / 3
    roll_s   = float(np.std([lag1, lag2, lag3]))

    raw_feature_map = {
        "Area":               req.area,
        "Annual_Rainfall":    req.rainfall,
        "Fertilizer":         req.fertilizer,
        "Pesticide":          1.5,
        "N": N, "P": P, "K": K, "pH": pH,
        "Temperature":        temp,
        "Humidity":           hum,
        "NPK_total":          npk,
        "Yield_lag_1":        lag1,
        "Yield_lag_2":        lag2,
        "Yield_lag_3":        lag3,
        "Yield_roll_mean3":   roll_m,
        "Yield_roll_std3":    roll_s,
        "Yield_yoy_change":   lag1 - lag2,
        "Rain_roll_mean3":    req.rainfall,
        "Rain_yoy_change":    0.0,
        "Temp_x_Rain":        t_x_r,
        "State_Name_Enc":     encode_label("State_Name", state),
        "Crop_Enc":           encode_label("Crop", crop),
        "Region_Enc":         encode_label("Region", region),
    }

    # Match exact feature order the model was trained on
    rf_features = (list(rf_model.feature_names_in_)
                   if hasattr(rf_model, "feature_names_in_")
                   else list(raw_feature_map.keys()))
    X = np.array([[raw_feature_map.get(f, 0.0) for f in rf_features]])

    y_rf   = float(rf_model.predict(X)[0])
    y_lstm = None

    if lstm_model is not None and lstm_scaler is not None:
        try:
            seq   = np.array([[X[0] * (0.9 + 0.05 * i)] for i in range(3)]).reshape(1, 3, len(rf_features))
            y_s   = lstm_model.predict(seq, verbose=0)[0][0]
            y_lstm = float(lstm_scaler.inverse_transform([[y_s]])[0][0])
        except Exception:
            y_lstm = None

    if y_lstm is not None:
        y_final   = max(0.0, ALPHA * y_rf + BETA * y_lstm)
        model_lbl = f"Hybrid RF+LSTM (α={ALPHA:.2f}, β={BETA:.2f})"
    else:
        y_final   = max(0.0, y_rf)
        model_lbl = "Random Forest"

    production = (y_final * req.area) / 1000.0
    confidence = "High" if y_final > 1000 else "Medium"

    return PredictResponse(
        yield_kg_ha=round(y_final, 2),
        production_tonnes=round(production, 3),
        region=f"{region} India",
        confidence=confidence,
        model_used=model_lbl,
        inputs_used={"N": N, "P": P, "K": K, "pH": pH,
                     "Temperature": temp, "Humidity": hum}
    )

# -----------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    html_path = "index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>Crop Yield Prediction API is running. Use POST /predict</h2>")


@app.post("/predict", response_model=PredictResponse, summary="Predict crop yield")
async def predict(req: PredictRequest):
    """
    Predict crop yield (kg/ha) and estimated production (tonnes).

    - **state**: Indian state name
    - **crop**: Crop type
    - **year**: Target year (used for display only)
    - **area**: Cultivation area in hectares
    - **rainfall**: Annual rainfall in mm
    - **fertilizer**: Fertilizer application in kg/ha
    - **N/P/K/pH/temperature/humidity**: Optional soil & climate values (auto-filled from ICAR defaults if omitted)
    """
    return build_prediction(req)


@app.get("/options", summary="Get available states and crops")
async def options():
    return {"states": STATES, "crops": CROPS}


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "rf_model": rf_model is not None,
        "lstm_model": lstm_model is not None,
        "fusion_alpha": ALPHA,
        "fusion_beta":  BETA,
    }


@app.get("/regions", summary="Region-wise model performance")
async def regions():
    rw = rf_metrics.get("region_wise", {})
    return {"region_performance": rw}


# -----------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
