# src/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import pickle
import joblib

TAGS = ("A", "B", "AFF")


def _load_pickle(path: Path):
    """Load a pickled model with joblib (fallback to pickle)."""
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _maybe_import_tf():
    """Import TensorFlow only if present; return None otherwise."""
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except Exception:
        return None


def _load_keras(path: Path):
    """Load a .keras model; raise a friendly error if TF isn't installed."""
    tf = _maybe_import_tf()
    if tf is None:
        raise RuntimeError(
            f"TensorFlow is required to load {path.name} but is not installed. "
            "Either add tensorflow==2.15.0 to requirements or remove the .keras model."
        )
    from tensorflow.keras.models import load_model  # type: ignore
    return load_model(path)


def load_all_models(model_dir: str | Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns:
      {
        "lgbm":   { "A": {"reg": ..., "cls": ...}, "B": {...}, "AFF": {...} },
        "lstm":   { ... }      # only if .keras files exist
        "bilstm": { ... }      # only if .keras files exist
        "y_scaler": <optional>
      }
    """
    model_dir = Path(model_dir)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {"lgbm": {}, "lstm": {}, "bilstm": {}}

    # ---- LightGBM pickles (.pkl) ----
    for tag in TAGS:
        reg_pkl = model_dir / f"nvda_{tag}_reg_lgb.pkl"
        cls_pkl = model_dir / f"nvda_{tag}_cls_lgb.pkl"
        bucket: Dict[str, Any] = {}
        if reg_pkl.exists():
            bucket["reg"] = _load_pickle(reg_pkl)
        if cls_pkl.exists():
            bucket["cls"] = _load_pickle(cls_pkl)
        if bucket:
            out["lgbm"][tag] = bucket

    # ---- Optional LSTM (.keras) ----
    for tag in TAGS:
        reg_k = model_dir / f"nvda_LSTM_{tag}_reg.keras"
        cls_k = model_dir / f"nvda_LSTM_{tag}_cls.keras"
        bucket: Dict[str, Any] = {}
        if reg_k.exists():
            bucket["reg"] = _load_keras(reg_k)
        if cls_k.exists():
            bucket["cls"] = _load_keras(cls_k)
        if bucket:
            out["lstm"][tag] = bucket

    # ---- Optional BiLSTM+Attention (.keras) ----
    for tag in TAGS:
        reg_k = model_dir / f"nvda_BiLSTM_Attn_{tag}_reg.keras"
        cls_k = model_dir / f"nvda_BiLSTM_Attn_{tag}_cls.keras"
        bucket: Dict[str, Any] = {}
        if reg_k.exists():
            bucket["reg"] = _load_keras(reg_k)
        if cls_k.exists():
            bucket["cls"] = _load_keras(cls_k)
        if bucket:
            out["bilstm"][tag] = bucket

    # ---- Optional y scaler ----
    y_scaler = None
    for name in ("y_scaler.pkl", "nvda_y_scaler.pkl"):
        p = model_dir / name
        if p.exists():
            y_scaler = _load_pickle(p)
            break
    if y_scaler is not None:
        # put at top level to match how main.py reads it
        out["y_scaler"] = y_scaler  # type: ignore[assignment]

    return out
