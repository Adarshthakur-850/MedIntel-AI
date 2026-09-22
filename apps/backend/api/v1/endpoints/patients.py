from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from apps.backend.database.session import get_db
from apps.backend.database.models import PatientRecord
from apps.backend.schemas.api_schemas import PatientCreateSchema

router = APIRouter()

@router.post("", status_code=status.HTTP_201_CREATED)
def create_patient_record(patient: PatientCreateSchema, db: Session = Depends(get_db)):
    existing = db.query(PatientRecord).filter(PatientRecord.patient_id == patient.patient_id).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Patient ID {patient.patient_id} already exists")

    db_patient = PatientRecord(**patient.dict())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return {"message": "Patient record saved successfully", "patient_id": db_patient.patient_id}

@router.get("/{patient_id}")
def get_patient_record(patient_id: str, db: Session = Depends(get_db)):
    record = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Patient record not found")
    return record
