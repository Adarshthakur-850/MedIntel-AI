import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from apps.backend.schemas.api_schemas import MultimodalPredictRequest, ExplanationResponse

from ml.explainability.shap_explain import TabularSHAPExplainer
from ml.explainability.nlp_explain import NLPTokenExplainer
from ml.explainability.gradcam_explain import GradCAMExplainer
from apps.backend.api.v1.endpoints.predict import (
    tabular_model, preprocessor, vision_model, img_tf
)

router = APIRouter()

nlp_explainer = NLPTokenExplainer()

@router.post("", response_model=ExplanationResponse)
def explain_multimodal(req: MultimodalPredictRequest):
    try:
        # 1. SHAP Tabular Explanation
        patient_df = pd.DataFrame([req.patient.dict()])
        if preprocessor and tabular_model:
            X_trans = preprocessor.transform(patient_df)
            feature_names = preprocessor.feature_names
            shap_explainer = TabularSHAPExplainer(tabular_model, feature_names)
            shap_res = shap_explainer.explain_sample(X_trans[0])
        else:
            shap_res = {"local_explanation": [], "top_positive_contributors": [], "top_negative_contributors": []}

        # 2. NLP Highlight Explanation
        nlp_res = nlp_explainer.explain_text(req.clinical_note)

        # 3. Vision Grad-CAM Explanation
        heatmap_path = None
        if req.image_path and os.path.exists(req.image_path):
            gradcam = GradCAMExplainer(vision_model)
            from PIL import Image
            import torch
            img = Image.open(req.image_path).convert('RGB')
            img_tensor = img_tf(img).unsqueeze(0)
            heatmap = gradcam.generate_heatmap(img_tensor)
            
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "data", "processed", "heatmaps")
            out_file = os.path.join(output_dir, f"heatmap_{req.patient.patient_id}.png")
            heatmap_path = gradcam.overlay_heatmap(req.image_path, heatmap, out_file)

        return ExplanationResponse(
            patient_id=req.patient.patient_id,
            shap_explanation=shap_res,
            nlp_explanation=nlp_res,
            vision_heatmap_path=heatmap_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
