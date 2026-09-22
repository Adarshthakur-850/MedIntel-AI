import pytest
from fastapi.testclient import TestClient
from apps.backend.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "disclaimer" in data

def test_model_info_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "tabular_model" in data
    assert "vision_model" in data

def test_rag_query_endpoint():
    payload = {"query": "What are hypertension guidelines?", "top_k": 2}
    response = client.post("/api/v1/rag/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data

def test_end_to_end_multimodal_predict():
    payload = {
        "patient": {
            "patient_id": "TEST_PAT_99",
            "age": 55,
            "sex": "M",
            "blood_pressure_sys": 140,
            "blood_pressure_dia": 90,
            "heart_rate": 78,
            "temperature": 37.0,
            "glucose": 130,
            "cholesterol": 210,
            "hemoglobin": 14.2,
            "creatinine": 1.1,
            "bmi": 28.0,
            "smoking_status": "Never",
            "diabetes_history": 0
        },
        "clinical_note": "Patient reports mild chest tightness.",
        "fusion_strategy": "neural"
    }
    response = client.post("/api/v1/predict/multimodal", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "TEST_PAT_99"
    assert 0.0 <= data["fused_risk"] <= 1.0
    assert "disclaimer" in data
