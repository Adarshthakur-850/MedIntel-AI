from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PatientCreateSchema(BaseModel):
    patient_id: str
    age: float = Field(..., ge=0, le=120)
    sex: str
    blood_pressure_sys: float = Field(..., ge=60, le=250)
    blood_pressure_dia: float = Field(..., ge=40, le=150)
    heart_rate: float = Field(..., ge=30, le=220)
    temperature: float = Field(..., ge=34.0, le=43.0)
    glucose: float = Field(..., ge=40, le=500)
    cholesterol: float = Field(..., ge=80, le=600)
    hemoglobin: float = Field(..., ge=5.0, le=22.0)
    creatinine: float = Field(..., ge=0.2, le=15.0)
    bmi: float = Field(..., ge=10.0, le=60.0)
    smoking_status: str
    diabetes_history: int

class SingleModalityRequest(BaseModel):
    patient_id: str
    tabular_data: Optional[PatientCreateSchema] = None
    clinical_note: Optional[str] = None
    image_path: Optional[str] = None

class MultimodalPredictRequest(BaseModel):
    patient: PatientCreateSchema
    clinical_note: str = Field(..., description="Patient symptoms or clinical history note")
    image_path: Optional[str] = Field(None, description="Chest X-ray image path")
    fusion_strategy: str = Field("neural", description="weighted_late or neural")

class MultimodalPredictResponse(BaseModel):
    patient_id: str
    tabular_risk: float
    image_risk: float
    nlp_risk: float
    fused_risk: float
    fusion_strategy: str
    disclaimer: str

class ExplanationResponse(BaseModel):
    patient_id: str
    shap_explanation: Dict[str, Any]
    nlp_explanation: Dict[str, Any]
    vision_heatmap_path: Optional[str] = None

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Medical research or clinical question")
    top_k: int = Field(3, ge=1, le=10)

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    disclaimer: str

class ModelMetadataResponse(BaseModel):
    tabular_model: str
    vision_model: str
    nlp_model: str
    multimodal_fusion: str
    rag_vector_db: str
    metrics: Dict[str, Any]
