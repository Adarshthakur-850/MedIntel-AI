from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

class PatientTabularData(BaseModel):
    patient_id: str
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: str = Field(..., description="Sex: M or F")
    blood_pressure_sys: float = Field(..., ge=60, le=250, description="Systolic Blood Pressure (mmHg)")
    blood_pressure_dia: float = Field(..., ge=40, le=150, description="Diastolic Blood Pressure (mmHg)")
    heart_rate: float = Field(..., ge=30, le=220, description="Heart Rate (bpm)")
    temperature: float = Field(..., ge=34.0, le=43.0, description="Body Temperature (°C)")
    glucose: float = Field(..., ge=40, le=500, description="Blood Glucose (mg/dL)")
    cholesterol: float = Field(..., ge=80, le=600, description="Total Cholesterol (mg/dL)")
    hemoglobin: float = Field(..., ge=5.0, le=22.0, description="Hemoglobin (g/dL)")
    creatinine: float = Field(..., ge=0.2, le=15.0, description="Serum Creatinine (mg/dL)")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index")
    smoking_status: str = Field(..., description="Never, Former, or Current")
    diabetes_history: int = Field(..., ge=0, le=1, description="0 or 1")
    cardiovascular_risk: Optional[int] = Field(None, description="Target label 0 (low) or 1 (high)")

    @field_validator('sex')

    def validate_sex(cls, v):
        if v not in ['M', 'F']:
            raise ValueError("Sex must be 'M' or 'F'")
        return v

    @field_validator('smoking_status')
    def validate_smoking(cls, v):
        if v not in ['Never', 'Former', 'Current']:
            raise ValueError("Smoking status must be 'Never', 'Former', or 'Current'")
        return v

class MultimodalPatientInput(BaseModel):
    tabular: PatientTabularData
    clinical_note: str = Field(..., description="Clinical note text")
    image_path: Optional[str] = Field(None, description="Path to patient chest X-ray image")
