import os
import json
from fastapi import APIRouter
from apps.backend.schemas.api_schemas import ModelMetadataResponse

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
tabular_metrics_path = os.path.join(BASE_DIR, "ml", "tabular", "checkpoints", "tabular_metrics.json")
vision_metrics_path = os.path.join(BASE_DIR, "ml", "vision", "checkpoints", "vision_metrics.json")

@router.get("", response_model=ModelMetadataResponse)
def get_model_information():
    tab_metrics = {}
    if os.path.exists(tabular_metrics_path):
        with open(tabular_metrics_path, "r") as f:
            tab_metrics = json.load(f)

    vis_metrics = {}
    if os.path.exists(vision_metrics_path):
        with open(vision_metrics_path, "r") as f:
            vis_metrics = json.load(f)

    return ModelMetadataResponse(
        tabular_model=tab_metrics.get("best_model", "LogisticRegression / XGBoost"),
        vision_model="ResNet18 (PyTorch Transfer Learning)",
        nlp_model="SentenceTransformers / Clinical TF-IDF NER",
        multimodal_fusion="Late Weighted Fusion & PyTorch Neural Fusion",
        rag_vector_db="FAISS / TF-IDF Vector Store",
        metrics={
            "tabular": tab_metrics.get("all_results", {}),
            "vision": vis_metrics
        }
    )
