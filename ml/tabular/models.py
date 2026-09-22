import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

class PyTorchTabularNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class NeuralNetworkWrapper:
    def __init__(self, input_dim, lr=0.001, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = PyTorchTabularNN(input_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            probs = self.model(X_tensor).numpy()
            return np.hstack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

def build_tabular_models(input_dim, seed=42):
    models = {
        'logistic_regression': LogisticRegression(random_state=seed, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=8),
        'xgboost': XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=seed, eval_metric='logloss'),
        'neural_network': NeuralNetworkWrapper(input_dim=input_dim, lr=0.002, epochs=40)
    }
    
    if LGBMClassifier is not None:
        models['lightgbm'] = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=seed, verbose=-1)

    return models
