import re
from typing import List, Dict, Any

HIGH_RISK_KEYWORDS = {
    'shortness of breath': 0.85,
    'chest tightness': 0.80,
    'chest pain': 0.90,
    'edema': 0.70,
    'hypertension': 0.65,
    'elevated glucose': 0.60,
    'pulmonary edema': 0.88,
    'cardiovascular stress': 0.82,
    'fatigue': 0.40,
    'fever': 0.45,
    'cough': 0.35
}

class NLPTokenExplainer:
    def explain_text(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        highlights = []

        for kw, weight in HIGH_RISK_KEYWORDS.items():
            if kw in text_lower:
                highlights.append({
                    'phrase': kw,
                    'risk_weight': weight,
                    'impact': 'high_positive_risk' if weight > 0.6 else 'moderate_risk'
                })

        highlights.sort(key=lambda x: x['risk_weight'], reverse=True)

        return {
            'text': text,
            'highlighted_phrases': highlights,
            'explanation_summary': f"Identified {len(highlights)} key clinical risk indicator phrases in patient text note."
        }
