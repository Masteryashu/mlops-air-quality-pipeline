"""
Pydantic Schemas for FastAPI
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class PredictionInput(BaseModel):
    """Input schema for prediction requests"""
    PT08_S1CO: float = Field(..., description="PT08.S1(CO) sensor value")
    NMHCGT: float = Field(..., description="NMHC(GT) concentration")
    C6H6GT: float = Field(..., description="C6H6(GT) Benzene concentration")
    PT08_S2NMHC: float = Field(..., description="PT08.S2(NMHC) sensor value")
    NOxGT: float = Field(..., description="NOx(GT) concentration")
    PT08_S3NOx: float = Field(..., description="PT08.S3(NOx) sensor value")
    NO2GT: float = Field(..., description="NO2(GT) concentration")
    PT08_S4NO2: float = Field(..., description="PT08.S4(NO2) sensor value")
    PT08_S5O3: float = Field(..., description="PT08.S5(O3) sensor value")
    T: float = Field(..., description="Temperature")
    RH: float = Field(..., description="Relative Humidity")
    AH: float = Field(..., description="Absolute Humidity")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, description="Is weekend (0=No, 1=Yes)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "PT08_S1CO": 1200.0,
                "NMHCGT": 150.0,
                "C6H6GT": 10.5,
                "PT08_S2NMHC": 950.0,
                "NOxGT": 200.0,
                "PT08_S3NOx": 800.0,
                "NO2GT": 120.0,
                "PT08_S4NO2": 1500.0,
                "PT08_S5O3": 1100.0,
                "T": 20.0,
                "RH": 50.0,
                "AH": 1.0,
                "hour": 12,
                "day_of_week": 3,
                "month": 6,
                "is_weekend": 0
            }
        }

class PredictionOutput(BaseModel):
    """Output schema for prediction responses"""
    prediction: float = Field(..., description="Predicted CO(GT) concentration")
    model_name: str = Field(..., description="Name of the model used")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
class ModelInfo(BaseModel):
    """Model information schema"""
    model_name: str
    model_type: str
    version: Optional[str] = None
    status: str
