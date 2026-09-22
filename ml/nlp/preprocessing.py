import re
import string

def clean_clinical_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
