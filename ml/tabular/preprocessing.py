import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERICAL_FEATURES = [
    'age', 'blood_pressure_sys', 'blood_pressure_dia', 'heart_rate',
    'temperature', 'glucose', 'cholesterol', 'hemoglobin', 'creatinine', 'bmi'
]

CATEGORICAL_FEATURES = ['sex', 'smoking_status', 'diabetes_history']

TARGET = 'cardiovascular_risk'

class TabularPreprocessor:
    def __init__(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERICAL_FEATURES),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
            ]
        )
        self.feature_names = None

    def fit_transform(self, df: pd.DataFrame):
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        y = df[TARGET].values if TARGET in df.columns else None
        
        X_trans = self.preprocessor.fit_transform(X)
        
        # Get feature names after one-hot encoding
        cat_encoder = self.preprocessor.named_transformers_['cat']
        cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names = NUMERICAL_FEATURES + cat_feature_names
        
        return X_trans, y

    def transform(self, df: pd.DataFrame):
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        X_trans = self.preprocessor.transform(X)
        return X_trans

def prepare_splits(df: pd.DataFrame, test_size=0.2, val_size=0.1, seed=42):
    preprocessor = TabularPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    # First split into train_val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Relative validation ratio
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=seed, stratify=y_train_val
    )
    
    return {
        'preprocessor': preprocessor,
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': preprocessor.feature_names
    }
