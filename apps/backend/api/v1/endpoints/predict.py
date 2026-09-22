import os
import sys
import joblib
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from apps.backend.database.session import get_db
from apps.backend.database.models import PredictionLog
from apps.backend.schemas.api_schemas import (
    MultimodalPredictRequest, MultimodalPredictResponse, SingleModalityRequest
)

from ml.tabular.preprocessing import TabularPreprocessor
from ml.vision.models import build_vision_model
from ml.nlp.classifier import ClinicalNLPClassifier
from ml.multimodal.fusion import MultimodalFusionEngine

router = APIRouter()

# Lazy load ML models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

tabular_model_path = os.path.join(BASE_DIR, "ml", "tabular", "checkpoints", "best_tabular_model.joblib")
preprocessor_path = os.path.join(BASE_DIR, "ml", "tabular", "checkpoints", "tabular_preprocessor.joblib")
vision_model_path = os.path.join(BASE_DIR, "ml", "vision", "checkpoints", "best_vision_model.pth")

preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
tabular_model = joblib.load(tabular_model_path) if os.path.exists(tabular_model_path) else None

nlp_classifier = ClinicalNLPClassifier()
fusion_engine = MultimodalFusionEngine()

vision_model = build_vision_model(arch="resnet18", pretrained=False)
if os.path.exists(vision_model_path):
    vision_model.load_state_dict(torch.load(vision_model_path, map_location=torch.device('cpu')))
vision_model.eval()

img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@router.post("/multimodal", response_model=MultimodalPredictResponse)
def predict_multimodal(req: MultimodalPredictRequest, db: Session = Depends(get_db)):
    try:
        # 1. Tabular Risk
        patient_df = pd.DataFrame([req.patient.dict()])
        if preprocessor and tabular_model:
            X_trans = preprocessor.transform(patient_df)
            tab_prob = float(tabular_model.predict_proba(X_trans)[0, 1])
            tab_embed = X_trans[0]
        else:
            tab_prob = 0.65
            tab_embed = np.zeros(15)

        # 2. NLP Risk
        nlp_res = nlp_classifier.predict_risk(req.clinical_note)
        nlp_prob = nlp_res['text_risk_score']
        text_embed = nlp_classifier.extract_embedding(req.clinical_note)

        # 3. Vision Risk
        if req.image_path and os.path.exists(req.image_path):
            img = Image.open(req.image_path).convert('RGB')
            img_tensor = img_tf(img).unsqueeze(0)
            with torch.no_grad():
                logits = vision_model(img_tensor)
                vision_prob = float(torch.sigmoid(logits)[0, 0].item())
                img_embed = vision_model.extract_features(img_tensor).squeeze(0).numpy()
        else:
            vision_prob = 0.50
            img_embed = np.zeros(512)

        # 4. Multimodal Fusion
        fused = fusion_engine.fuse_all_modalities(
            tabular_risk=tab_prob,
            image_risk=vision_prob,
            nlp_risk=nlp_prob,
            tab_embed=tab_embed,
            img_embed=img_embed,
            text_embed=text_embed,
            method=req.fusion_strategy
        )

        # Log prediction
        log_entry = PredictionLog(
            patient_id=req.patient.patient_id,
            tabular_risk=tab_prob,
            vision_risk=vision_prob,
            nlp_risk=nlp_prob,
            fused_risk=fused['final_fused_risk'],
            fusion_method=req.fusion_strategy
        )
        db.add(log_entry)
        db.commit()

        return MultimodalPredictResponse(
            patient_id=req.patient.patient_id,
            tabular_risk=round(tab_prob, 4),
            image_risk=round(vision_prob, 4),
            nlp_risk=round(nlp_prob, 4),
            fused_risk=fused['final_fused_risk'],
            fusion_strategy=req.fusion_strategy,
            disclaimer="MedIntel AI research decision-support risk estimate. Not a diagnosis."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tabular")
def predict_tabular_only(req: SingleModalityRequest):
    if not req.tabular_data:
        raise HTTPException(status_code=400, detail="tabular_data is required")
    patient_df = pd.DataFrame([req.tabular_data.dict()])
    if preprocessor and tabular_model:
        X_trans = preprocessor.transform(patient_df)
        prob = float(tabular_model.predict_proba(X_trans)[0, 1])
    else:
        prob = 0.65
    return {"patient_id": req.patient_id, "tabular_risk": round(prob, 4)}
