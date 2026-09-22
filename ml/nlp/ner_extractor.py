import re
from typing import Dict, List

COMMON_SYMPTOMS = [
    'shortness of breath', 'cough', 'fever', 'chest tightness', 'chest pain',
    'fatigue', 'dizziness', 'palpitations', 'nausea', 'edema', 'headache', 'sweating'
]

COMMON_CONDITIONS = [
    'hypertension', 'diabetes', 'pulmonary edema', 'cardiovascular stress',
    'pneumonia', 'heart failure', 'arrhythmia', 'coronary artery disease'
]

COMMON_MEDICATIONS = [
    'metformin', 'lisinopril', 'aspirin', 'atorvastatin', 'amlodipine',
    'metoprolol', 'furosemide', 'albuterol', 'insulin'
]

class ClinicalNERExtractor:
    def __init__(self):
        pass

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        
        symptoms = [s for s in COMMON_SYMPTOMS if s in text_lower]
        conditions = [c for c in COMMON_CONDITIONS if c in text_lower]
        medications = [m for m in COMMON_MEDICATIONS if m in text_lower]
        
        # Regex for duration pattern: e.g. "5 days", "2 weeks", "past 3 months"
        duration_matches = re.findall(r'(\d+\s*(?:days?|weeks?|months?|years?)|past \d+ (?:days?|weeks?|months?))', text_lower)
        
        return {
            "symptoms": symptoms,
            "durations": duration_matches,
            "conditions": conditions,
            "medications": medications
        }
