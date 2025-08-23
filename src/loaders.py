from pathlib import Path
from typing import Dict, Any
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense

# --- Custom layer used by your BiLSTM+Attention models ---
class SoftAttention(tf.keras.layers.Layer):
    """Additive soft attention over time: input (B,T,F) -> context (B,F)."""
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.proj = Dense(units, activation="tanh")
        self.score = Dense(1, activation=None)

    def call(self, h, mask=None, training=None):
        e = self.score(self.proj(h))  # (B,T,1)
        if mask is not None:
            m = tf.cast(mask[:, :, tf.newaxis], tf.float32)
            e = e + (1.0 - m) * (-1e9)
        a = tf.nn.softmax(e, axis=1)  # (B,T,1)
        return tf.reduce_sum(a * h, axis=1)  # (B,F)

def _load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

def load_all_models(model_dir: Path) -> Dict[str, Any]:
    """
    Loads available artifacts from model_dir.

    Expected (optional) files:
      - LightGBM: nvda_{TAG}_reg_lgb.pkl, nvda_{TAG}_cls_lgb.pkl
      - LSTM: nvda_LSTM_{TAG}_{reg|cls}.keras
      - BiLSTM+Attn: nvda_BiLSTM_Attn_{TAG}_{reg|cls}.keras
      - y_scaler.pkl  (shared scaler for inverse-transform)
    """
    store: Dict[str, Any] = {
        "lgbm": {"A": {}, "B": {}, "AFF": {}},
        "lstm": {"A": {}, "B": {}, "AFF": {}},
        "bilstm": {"A": {}, "B": {}, "AFF": {}},
        "y_scaler": None,
    }

    # y_scaler (optional)
    y_scaler_pkl = model_dir / "y_scaler.pkl"
    if y_scaler_pkl.exists():
        store["y_scaler"] = _load_pickle(y_scaler_pkl)

    # LightGBM (sklearn wrappers)
    for tag in ["A", "B", "AFF"]:
        reg_pkl = model_dir / f"nvda_{tag}_reg_lgb.pkl"
        cls_pkl = model_dir / f"nvda_{tag}_cls_lgb.pkl"
        if reg_pkl.exists():
            store["lgbm"][tag]["reg"] = _load_pickle(reg_pkl)
        if cls_pkl.exists():
            store["lgbm"][tag]["cls"] = _load_pickle(cls_pkl)

    # LSTM (.keras)
    for tag in ["A", "B", "AFF"]:
        reg_keras = model_dir / f"nvda_LSTM_{tag}_reg.keras"
        cls_keras = model_dir / f"nvda_LSTM_{tag}_cls.keras"
        if reg_keras.exists():
            store["lstm"][tag]["reg"] = tf.keras.models.load_model(reg_keras)
        if cls_keras.exists():
            store["lstm"][tag]["cls"] = tf.keras.models.load_model(cls_keras)

    # BiLSTM+Attention (.keras) — need custom_objects
    custom = {"SoftAttention": SoftAttention}
    for tag in ["A", "B", "AFF"]:
        reg_bi = model_dir / f"nvda_BiLSTM_Attn_{tag}_reg.keras"
        cls_bi = model_dir / f"nvda_BiLSTM_Attn_{tag}_cls.keras"
        if reg_bi.exists():
            store["bilstm"][tag]["reg"] = tf.keras.models.load_model(reg_bi, custom_objects=custom, compile=False)
        if cls_bi.exists():
            store["bilstm"][tag]["cls"] = tf.keras.models.load_model(cls_bi, custom_objects=custom, compile=False)

    return store
