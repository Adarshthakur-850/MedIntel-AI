import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from data.schemas.patient_schema import PatientTabularData

class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate_schema_and_types(self):
        required_cols = [
            'patient_id', 'age', 'sex', 'blood_pressure_sys', 'blood_pressure_dia',
            'heart_rate', 'temperature', 'glucose', 'cholesterol', 'hemoglobin',
            'creatinine', 'bmi', 'smoking_status', 'diabetes_history', 'cardiovascular_risk'
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Data validation failed: Missing required columns {missing}")
        return True

    def validate_null_values(self):
        null_counts = self.df.isnull().sum().to_dict()
        total_nulls = sum(null_counts.values())
        if total_nulls > 0:
            raise ValueError(f"Data validation failed: Found null values: {null_counts}")
        return True

    def validate_pydantic_records(self, sample_size=50):
        records = self.df.sample(min(sample_size, len(self.df))).to_dict(orient='records')
        for record in records:
            PatientTabularData(**record)
        return True

    def validate_class_distribution(self):
        counts = self.df['cardiovascular_risk'].value_counts()
        if len(counts) < 2:
            raise ValueError("Data validation failed: Target class must have at least 2 classes")
        ratio = counts.min() / counts.max()
        if ratio < 0.05:
            raise ValueError(f"Severe class imbalance detected: ratio {ratio:.2f}")
        return counts.to_dict()

    def run_all_checks(self):
        self.validate_schema_and_types()
        self.validate_null_values()
        self.validate_pydantic_records()
        dist = self.validate_class_distribution()
        print("All data quality validation checks passed successfully!")
        print(f"Target Distribution: {dist}")
        return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "raw", "synthetic_medical_dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        validator = DataValidator(df)
        validator.run_all_checks()
    else:
        print("Synthetic dataset CSV not found. Please run data_generator.py first.")
