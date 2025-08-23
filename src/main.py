# src/main.py
import os
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
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

# globals filled at startup
MODELS: Dict = {}
Y_SCALER = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS, Y_SCALER
    try:
        MODELS = load_all_models(MODEL_DIR)
        Y_SCALER = MODELS.get("y_scaler", None)
        print(f"[startup] Loaded models from {MODEL_DIR.resolve()}")
    except Exception as e:
        # Don’t crash the server — endpoints will 404 if a model is missing
        print(f"[startup] WARNING: failed to load models: {e!r}")
        MODELS = {}
        Y_SCALER = None
    yield
    # (optional) clean up

app = FastAPI(title="NVDA Forecast API", version="1.0.0", lifespan=lifespan)

# CORS — tighten allow_origins in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://<your-username>.github.io",
        "https://<your-username>.github.io/<repo-name>",
        "*",  # dev only
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
