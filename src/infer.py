import numpy as np
from typing import List, Tuple

def to_np(x: List[List[float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def last_step(x_seq: np.ndarray) -> np.ndarray:
    """
    From [T,F] -> [1,F] last timestep. If already [1,F], return as-is.
    """
    if x_seq.ndim != 2:
        raise ValueError("X must be 2D [T,F] or [1,F].")
    return x_seq[-1:, :]

def inverse_y_if_possible(y_scaled: float, y_scaler) -> Tuple[float, bool]:
    """
    Try inverse-transform to original space using provided scaler.
    Returns (value, scaled_flag). scaled_flag=False if inverse succeeded.
    """
    if y_scaler is None:
        return float(y_scaled), True
    inv = y_scaler.inverse_transform(np.array([[y_scaled]])).ravel()[0]
    return float(inv), False

def prepare_seq_for_keras(x_seq: np.ndarray) -> np.ndarray:
    """
    Keras expects [B,T,F]; we provide B=1 for single-request inference.
    """
    if x_seq.ndim != 2:
        raise ValueError("Keras expects 2D [T,F] before batching.")
    return x_seq[np.newaxis, :, :]
