from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from datetime import datetime
from apps.backend.database.session import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(20), default="researcher")
    created_at = Column(DateTime, default=datetime.utcnow)

class PatientRecord(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True, nullable=False)
    age = Column(Float, nullable=False)
    sex = Column(String(10), nullable=False)
    blood_pressure_sys = Column(Float, nullable=False)
    blood_pressure_dia = Column(Float, nullable=False)
    heart_rate = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    glucose = Column(Float, nullable=False)
    cholesterol = Column(Float, nullable=False)
    hemoglobin = Column(Float, nullable=False)
    creatinine = Column(Float, nullable=False)
    bmi = Column(Float, nullable=False)
    smoking_status = Column(String(20), nullable=False)
    diabetes_history = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionLog(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), index=True, nullable=False)
    tabular_risk = Column(Float, nullable=False)
    vision_risk = Column(Float, nullable=False)
    nlp_risk = Column(Float, nullable=False)
    fused_risk = Column(Float, nullable=False)
    fusion_method = Column(String(30), nullable=False)
    explanation_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(100), nullable=False)
    endpoint = Column(String(100), nullable=False)
    status_code = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
