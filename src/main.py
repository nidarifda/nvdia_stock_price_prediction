import os
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .schemas import (
    HealthResponse,
    RegressionRequest, RegressionResponse,
    ClassificationRequest, ClassificationResponse
)
from .loaders import load_all_models
from .infer import to_np, last_step, inverse_y_if_possible, prepare_seq_for_keras

load_dotenv()
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
THRESH = float(os.getenv("THRESHOLD_UP", "0.5"))
DEFAULT_TAG = os.getenv("DEFAULT_TAG", "B")
DEFAULT_FRAMEWORK = os.getenv("DEFAULT_FRAMEWORK", "lgbm")

app = FastAPI(title="NVDA Forecast API", version="1.0.0")

# CORS — add your Netlify URL when deployed
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # "https://<your-netlify-site>.netlify.app",
        "*",  # loosen during dev; tighten in prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
MODELS = load_all_models(MODEL_DIR)
Y_SCALER = MODELS.get("y_scaler", None)

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()

def _ensure_available(framework: str, tag: str, kind: str):
    mdl = MODELS.get(framework, {}).get(tag, {}).get(kind)
    if mdl is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {framework}/{tag}/{kind}")
    return mdl

@app.post("/predict/regression", response_model=RegressionResponse)
def predict_regression(req: RegressionRequest):
    fw = req.framework
    tag = req.tag
    x = to_np(req.X)  # [T,F] or [1,F]

    if fw == "lgbm":
        x_last = last_step(x)  # [1,F]
        reg = _ensure_available("lgbm", tag, "reg")
        y_scaled = float(reg.predict(x_last)[0])
        y_pred, scaled_flag = inverse_y_if_possible(y_scaled, Y_SCALER)
        return RegressionResponse(tag=tag, framework=fw, y_pred=y_pred, scaled=scaled_flag,
                                  note=None if not scaled_flag else "Returned in scaled space; y_scaler.pkl missing")

    elif fw == "lstm":
        reg = _ensure_available("lstm", tag, "reg")
        x_btf = prepare_seq_for_keras(x)  # [1,T,F]
        y_scaled = float(reg.predict(x_btf, verbose=0).ravel()[0])
        y_pred, scaled_flag = inverse_y_if_possible(y_scaled, Y_SCALER)
        return RegressionResponse(tag=tag, framework=fw, y_pred=y_pred, scaled=scaled_flag,
                                  note=None if not scaled_flag else "Returned in scaled space; y_scaler.pkl missing")

    elif fw == "bilstm":
        reg = _ensure_available("bilstm", tag, "reg")
        x_btf = prepare_seq_for_keras(x)
        y_scaled = float(reg.predict(x_btf, verbose=0).ravel()[0])
        y_pred, scaled_flag = inverse_y_if_possible(y_scaled, Y_SCALER)
        return RegressionResponse(tag=tag, framework=fw, y_pred=y_pred, scaled=scaled_flag,
                                  note=None if not scaled_flag else "Returned in scaled space; y_scaler.pkl missing")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown framework: {fw}")

@app.post("/predict/classification", response_model=ClassificationResponse)
def predict_classification(req: ClassificationRequest):
    fw = req.framework
    tag = req.tag
    x = to_np(req.X)

    if fw == "lgbm":
        x_last = last_step(x)
        cls = _ensure_available("lgbm", tag, "cls")
        p_up = float(cls.predict_proba(x_last)[:, 1][0])

    elif fw == "lstm":
        cls = _ensure_available("lstm", tag, "cls")
        x_btf = prepare_seq_for_keras(x)
        p_up = float(cls.predict(x_btf, verbose=0).ravel()[0])

    elif fw == "bilstm":
        cls = _ensure_available("bilstm", tag, "cls")
        x_btf = prepare_seq_for_keras(x)
        p_up = float(cls.predict(x_btf, verbose=0).ravel()[0])

    else:
        raise HTTPException(status_code=400, detail=f"Unknown framework: {fw}")

    label = int(p_up >= THRESH)
    return ClassificationResponse(tag=tag, framework=fw, p_up=p_up, label=label, threshold=THRESH)
