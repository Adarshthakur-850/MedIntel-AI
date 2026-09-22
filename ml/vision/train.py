import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from ml.vision.dataset import ChestXRayDataset, get_vision_transforms
from ml.vision.models import build_vision_model
from ml.tabular.evaluator import evaluate_classifier

def run_vision_training(epochs=5, batch_size=16, lr=0.0003, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "raw", "synthetic_medical_dataset.csv")
    checkpoint_dir = os.path.join(base_dir, "ml", "vision", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['cardiovascular_risk'])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=seed, stratify=train_df['cardiovascular_risk'])

    train_tf, val_tf = get_vision_transforms(img_size=224)

    train_ds = ChestXRayDataset(train_df['image_path'].tolist(), train_df['cardiovascular_risk'].tolist(), transform=train_tf)
    val_ds = ChestXRayDataset(val_df['image_path'].tolist(), val_df['cardiovascular_risk'].tolist(), transform=val_tf)
    test_ds = ChestXRayDataset(test_df['image_path'].tolist(), test_df['cardiovascular_risk'].tolist(), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Computer Vision (ResNet18) on {device} ---")

    model = build_vision_model(arch="resnet18", pretrained=False).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, "best_vision_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # Test set evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(np.atleast_1d(probs).tolist())
            all_preds.extend(np.atleast_1d(preds).tolist())
            all_targets.extend(labels.numpy().tolist())

    metrics = evaluate_classifier(np.array(all_targets), np.array(all_preds), np.array(all_probs))
    print(f"\nVision Model Test Metrics: Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    metrics_path = os.path.join(checkpoint_dir, "vision_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Vision model saved to {best_model_path}")
    return metrics

if __name__ == "__main__":
    run_vision_training()
