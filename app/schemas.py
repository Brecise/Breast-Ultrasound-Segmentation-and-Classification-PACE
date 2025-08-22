# /breast-cancer-api/app/schemas.py

from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    filename: str
    content_type: str
    predicted_class_id: int
    predicted_class_name: str
    mask_base64: str
    error: Optional[str] = None

    class Config:
        orm_mode = True
        