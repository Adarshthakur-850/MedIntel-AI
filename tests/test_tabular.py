import pytest
import numpy as np
import pandas as pd

from ml.tabular.preprocessing import prepare_splits, TabularPreprocessor
from ml.tabular.models import build_tabular_models
from ml.tabular.evaluator import evaluate_classifier

def test_tabular_preprocessor():
    data = {
        'age': [50, 60],
        'blood_pressure_sys': [120, 140],
        'blood_pressure_dia': [80, 90],
        'heart_rate': [70, 80],
        'temperature': [36.6, 37.0],
        'glucose': [100, 150],
        'cholesterol': [200, 240],
        'hemoglobin': [14.0, 13.0],
        'creatinine': [1.0, 1.2],
        'bmi': [25.0, 30.0],
        'sex': ['M', 'F'],
        'smoking_status': ['Never', 'Current'],
        'diabetes_history': [0, 1],
        'cardiovascular_risk': [0, 1]
    }
    df = pd.DataFrame(data)
    preprocessor = TabularPreprocessor()
    X_trans, y = preprocessor.fit_transform(df)
    assert X_trans.shape[0] == 2
    assert len(preprocessor.feature_names) > 10
    assert y.tolist() == [0, 1]

def test_tabular_model_predictions():
    X = np.random.randn(20, 15)
    y = np.random.choice([0, 1], 20)
    models = build_tabular_models(input_dim=15)
    for name, model in models.items():
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        assert probs.shape[0] == 20
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_evaluator_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.9, 0.3, 0.7])
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate_classifier(y_true, y_pred, y_prob)
    assert metrics['accuracy'] == 1.0
    assert metrics['f1_score'] == 1.0
