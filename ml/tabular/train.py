import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import joblib
import pandas as pd

from ml.tabular.preprocessing import prepare_splits
from ml.tabular.models import build_tabular_models
from ml.tabular.evaluator import evaluate_classifier

def run_tabular_training():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "raw", "synthetic_medical_dataset.csv")
    checkpoint_dir = os.path.join(base_dir, "ml", "tabular", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run pipelines/data_generator.py first.")

    df = pd.read_csv(data_path)
    splits = prepare_splits(df)

    preprocessor = splits['preprocessor']
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']

    models = build_tabular_models(input_dim=X_train.shape[1])
    
    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    print(f"--- Training Tabular Models on {X_train.shape[0]} samples ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Predict on Test set
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate_classifier(y_test, y_pred, y_prob)
        results[name] = metrics
        print(f"Model: {name:<20} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model_name = name
            best_model_obj = model

    print(f"\nTop performing tabular model: {best_model_name} (F1 = {best_f1:.4f})")

    # Save artifacts
    joblib.dump(preprocessor, os.path.join(checkpoint_dir, "tabular_preprocessor.joblib"))
    joblib.dump(best_model_obj, os.path.join(checkpoint_dir, "best_tabular_model.joblib"))
    with open(os.path.join(checkpoint_dir, "tabular_metrics.json"), "w") as f:
        json.dump({
            'best_model': best_model_name,
            'all_results': results
        }, f, indent=2)

    print(f"Artifacts successfully saved to: {checkpoint_dir}")
    return results

if __name__ == "__main__":
    run_tabular_training()
