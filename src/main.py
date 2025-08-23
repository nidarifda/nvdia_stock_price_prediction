# src/main.py
import os
from pathlib import Path
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .schemas import HealthResponse, RegressionRequest, RegressionResponse, ClassificationRequest, ClassificationResponse
from .loaders import load_all_models
from .infer import to_np, last_step, inverse_y_if_possible, prepare_seq_for_keras

load_dotenv()
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
THRESH = float(os.getenv("THRESHOLD_UP", "0.5"))
DEFAULT_TAG = os.getenv("DEFAULT_TAG", "B")
DEFAULT_FRAMEWORK = os.getenv("DEFAULT_FRAMEWORK", "lgbm")

MODELS: Dict = {}
Y_SCALER = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS, Y_SCALER
    try:
        MODELS = load_all_models(MODEL_DIR)
        Y_SCALER = MODELS.get("y_scaler")
        print("✅ Models loaded.")
    except Exception as e:
        import traceback; traceback.print_exc()
        MODELS = {}
        Y_SCALER = None
        # keep app running; health will show not-ready
    yield

app = FastAPI(title="NVDA Forecast API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173","*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health():
    # You can enhance HealthResponse to include a simple “ready” flag/message
    if not MODELS:
        return HealthResponse(message="loaded: false")
    return HealthResponse(message="ok")
