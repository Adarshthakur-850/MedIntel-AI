import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from ml.nlp.preprocessing import clean_clinical_text
from ml.nlp.ner_extractor import ClinicalNERExtractor

class ClinicalNLPClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=250, stop_words='english')
        self.model = LogisticRegression(random_state=42)
        self.ner = ClinicalNERExtractor()
        self.is_fitted = False

    def fit(self, texts, labels):
        cleaned_texts = [clean_clinical_text(t) for t in texts]
        X = self.vectorizer.fit_transform(cleaned_texts)
        self.model.fit(X, labels)
        self.is_fitted = True

    def predict_risk(self, text: str):
        if not self.is_fitted:
            # Rule-based fallback if not fitted yet
            entities = self.ner.extract_entities(text)
            risk = 0.2 + 0.2 * len(entities['symptoms']) + 0.25 * len(entities['conditions'])
            prob = min(max(risk, 0.05), 0.95)
        else:
            cleaned = clean_clinical_text(text)
            X = self.vectorizer.transform([cleaned])
            prob = float(self.model.predict_proba(X)[0, 1])

        entities = self.ner.extract_entities(text)
        return {
            'text_risk_score': round(prob, 4),
            'extracted_entities': entities
        }

    def extract_embedding(self, text: str, output_dim=64):
        # Generate dense text representation from TF-IDF or hashing
        cleaned = clean_clinical_text(text)
        if self.is_fitted:
            vec = self.vectorizer.transform([cleaned]).toarray()[0]
            if len(vec) >= output_dim:
                return vec[:output_dim]
            else:
                return np.pad(vec, (0, output_dim - len(vec)))
        else:
            # Deterministic pseudo-embedding based on hash
            np.random.seed(abs(hash(text)) % (2**32))
            return np.random.normal(0, 1, output_dim)
