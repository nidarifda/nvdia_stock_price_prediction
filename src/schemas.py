from typing import List, Literal, Optional
from pydantic import BaseModel, Field

Tag = Literal["A", "B", "AFF"]
Framework = Literal["lgbm", "lstm", "bilstm"]

class HealthResponse(BaseModel):
    status: str = "ok"

class RegressionRequest(BaseModel):
    tag: Tag = "B"
    framework: Framework = "lgbm"
    # For sequence models: X is [T,F]; for LGBM you can still send [T,F],
    # the server will use the last row.
    X: List[List[float]] = Field(..., description="Sequence [T,F] or last-step [1,F]")

class RegressionResponse(BaseModel):
    tag: Tag
    framework: Framework
    y_pred: float
    scaled: bool = False   # True if inverse-transform not applied
    note: Optional[str] = None

class ClassificationRequest(BaseModel):
    tag: Tag = "B"
    framework: Framework = "lgbm"
    X: List[List[float]]

class ClassificationResponse(BaseModel):
    tag: Tag
    framework: Framework
    p_up: float
    label: int
    threshold: float
