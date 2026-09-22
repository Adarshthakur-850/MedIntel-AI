import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

def generate_synthetic_tabular_data(n_samples=500, seed=42):
    np.random.seed(seed)
    patient_ids = [f"PAT_{i:04d}" for i in range(1, n_samples + 1)]
    
    age = np.random.uniform(18, 85, n_samples).round(1)
    sex = np.random.choice(['M', 'F'], n_samples)
    bp_sys = np.random.normal(125, 18, n_samples).clip(85, 200).round(1)
    bp_dia = np.random.normal(80, 12, n_samples).clip(55, 120).round(1)
    heart_rate = np.random.normal(75, 12, n_samples).clip(45, 140).round(1)
    temperature = np.random.normal(36.8, 0.6, n_samples).clip(35.5, 40.5).round(1)
    glucose = np.random.normal(105, 30, n_samples).clip(60, 350).round(1)
    cholesterol = np.random.normal(200, 40, n_samples).clip(110, 450).round(1)
    hemoglobin = np.random.normal(14.0, 1.8, n_samples).clip(7.0, 19.0).round(1)
    creatinine = np.random.normal(1.0, 0.4, n_samples).clip(0.4, 5.0).round(2)
    bmi = np.random.normal(27.0, 5.0, n_samples).clip(15.0, 50.0).round(1)
    smoking_status = np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.5, 0.3, 0.2])
    diabetes_history = (glucose > 140).astype(int)
    
    # Calculate realistic synthetic risk score based on clinical factors
    risk_score = (
        (age - 50) * 0.03 +
        (bp_sys - 120) * 0.02 +
        (glucose - 100) * 0.015 +
        (cholesterol - 200) * 0.01 +
        (bmi - 25) * 0.02 +
        (1 if sex[0] == 'M' else 0) * 0.2 +
        diabetes_history * 0.5 +
        np.random.normal(0, 0.3, n_samples)
    )
    risk_prob = 1 / (1 + np.exp(-risk_score))
    target = (risk_prob > 0.5).astype(int)

    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': age,
        'sex': sex,
        'blood_pressure_sys': bp_sys,
        'blood_pressure_dia': bp_dia,
        'heart_rate': heart_rate,
        'temperature': temperature,
        'glucose': glucose,
        'cholesterol': cholesterol,
        'hemoglobin': hemoglobin,
        'creatinine': creatinine,
        'bmi': bmi,
        'smoking_status': smoking_status,
        'diabetes_history': diabetes_history,
        'cardiovascular_risk': target
    })
    return df

def generate_synthetic_image(is_abnormal=False, save_path=""):
    # Generate realistic synthetic chest X-ray silhouette
    img = Image.new('L', (224, 224), color=20)
    draw = ImageDraw.Draw(img)
    
    # Draw chest cavity / rib cage outline
    draw.ellipse([30, 20, 194, 204], fill=50, outline=90)
    # Draw lung fields
    draw.ellipse([45, 40, 100, 180], fill=120)
    draw.ellipse([124, 40, 179, 180], fill=120)
    # Draw heart silhouette
    draw.ellipse([90, 100, 145, 170], fill=60)
    
    if is_abnormal:
        # Add opacity / infiltrate patch on right lung field
        draw.ellipse([55, 80, 95, 130], fill=190)
        draw.ellipse([60, 90, 85, 115], fill=220)
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
    else:
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

def generate_synthetic_nlp_notes(df):
    notes = []
    for idx, row in df.iterrows():
        if row['cardiovascular_risk'] == 1:
            note = (
                f"Patient {row['patient_id']} presents with persistent shortness of breath, chest tightness, "
                f"and fatigue for past 5 days. History of hypertension (BP {row['blood_pressure_sys']}/{row['blood_pressure_dia']} mmHg) "
                f"and elevated glucose ({row['glucose']} mg/dL). Currently taking Metformin and Lisinopril. Suspected pulmonary edema or cardiovascular stress."
            )
        else:
            note = (
                f"Patient {row['patient_id']} routine follow-up examination. Vital signs stable with heart rate {row['heart_rate']} bpm "
                f"and body temperature {row['temperature']} C. No acute distress reported. Patient denies shortness of breath or chest pain. Continue current wellness plan."
            )
        notes.append(note)
    return notes

def main():
    print("Generating MedIntel AI synthetic dataset...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    images_dir = os.path.join(raw_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    df = generate_synthetic_tabular_data(n_samples=300)
    notes = generate_synthetic_nlp_notes(df)
    df['clinical_note'] = notes

    image_paths = []
    for idx, row in df.iterrows():
        img_name = f"{row['patient_id']}.png"
        img_path = os.path.join(images_dir, img_name)
        is_abnormal = (row['cardiovascular_risk'] == 1)
        generate_synthetic_image(is_abnormal=is_abnormal, save_path=img_path)
        image_paths.append(img_path)

    df['image_path'] = image_paths

    csv_path = os.path.join(raw_dir, "synthetic_medical_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully created at: {csv_path}")
    print(f"Total samples: {len(df)}, Features: {df.shape[1]}")

if __name__ == "__main__":
    main()
