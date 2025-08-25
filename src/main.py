# src/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Dict
import os
import joblib
import numpy as np

APP_TITLE = "NVDA Forecast API"
APP_VERSION = "1.0.0"

# ---- Models loader (regression only, LightGBM .pkl) -------------------------
def load_regression_models(model_dir: str) -> Dict[str, object]:
    """
    Load LightGBM regression models for tags A, B, AFF if present.
    Filenames expected:
      - nvda_A_reg_lgb.pkl
      - nvda_B_reg_lgb.pkl
      - nvda_AFF_reg_lgb.pkl
    """
    tag_to_file = {
        "A":   "nvda_A_reg_lgb.pkl",
        "B":   "nvda_B_reg_lgb.pkl",
        "AFF": "nvda_AFF_reg_lgb.pkl",
    }
    models = {}
    for tag, fname in tag_to_file.items():
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            models[tag] = joblib.load(path)
    return models

# ---- FastAPI app -------------------------------------------------------------
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# CORS: allow your GitHub Pages + localhost
cors_env = os.getenv("CORS_ALLOW_ORIGINS", "")
origins = [o.strip() for o in cors_env.split(",") if o.strip()]
if not origins:
    # safe defaults for local/dev + GH Pages
    origins = [
        "http://localhost:5173",
        "https://nidarifda.github.io",
        "https://nidarifda.github.io/nvdia_stock_price_prediction",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_TAG = os.getenv("DEFAULT_TAG", "B")
MODEL_DIR   = os.getenv("MODEL_DIR", "models")

# Pydantic I/O schemas
class PredictRequest(BaseModel):
    tag: Literal["A", "B", "AFF"] = Field(default=DEFAULT_TAG)
    framework: Literal["lgbm"] = Field(default="lgbm")  # server supports lgbm only
    X: List[List[float]]  # 2D list [T, F]

class PredictResponse(BaseModel):
    framework: str
    tag: str
    y_pred: float
    scaled: bool = False  # for clarity; no inverse-scaling here

# Load models at startup
@app.on_event("startup")
def _startup():
    app.state.reg_models = load_regression_models(MODEL_DIR)
    print("✅ Models loaded.")

# Health
@app.get("/health")
def health():
    return {"status": "ok"}

# Regression
@app.post("/predict/regression", response_model=PredictResponse)
def predict_regression(body: PredictRequest):
    if body.framework != "lgbm":
        raise HTTPException(status_code=400, detail="Only 'lgbm' is supported on the server.")
    models = getattr(app.state, "reg_models", {})
    if body.tag not in models:
        raise HTTPException(status_code=400, detail=f"No regression model for tag '{body.tag}'.")

    X = np.asarray(body.X, dtype=float)
    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="X must be a 2D array [T, F].")

    # LightGBM: use last row
    x_last = X[-1].reshape(1, -1)
    model = models[body.tag]
    y = model.predict(x_last)
    y_val = float(y[0]) if hasattr(y, "__len__") else float(y)

    return PredictResponse(framework="lgbm", tag=body.tag, y_pred=y_val, scaled=False)
