import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, brier_score_loss
)

def evaluate_classifier(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)
    
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    
    metrics = {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'sensitivity': round(float(sensitivity), 4),
        'specificity': round(float(specificity), 4),
        'brier_score_calibration': round(float(brier_score_loss(y_true, y_prob)), 4),
        'confusion_matrix': cm.tolist()
    }
    return metrics
