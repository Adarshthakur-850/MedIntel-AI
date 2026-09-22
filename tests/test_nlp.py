from ml.nlp.preprocessing import clean_clinical_text
from ml.nlp.ner_extractor import ClinicalNERExtractor
from ml.nlp.classifier import ClinicalNLPClassifier

def test_nlp_text_cleaning():
    text = "  Patient HAS   fever!  "
    cleaned = clean_clinical_text(text)
    assert cleaned == "patient has fever!"

def test_ner_entity_extraction():
    extractor = ClinicalNERExtractor()
    note = "Patient has shortness of breath and fever for 5 days. Taking Metformin."
    entities = extractor.extract_entities(note)
    assert "shortness of breath" in entities['symptoms']
    assert "fever" in entities['symptoms']
    assert "metformin" in entities['medications']

def test_nlp_classifier_risk_bounds():
    classifier = ClinicalNLPClassifier()
    res = classifier.predict_risk("Patient complains of chest pain and dizziness.")
    assert 0.0 <= res['text_risk_score'] <= 1.0
    emb = classifier.extract_embedding("Test clinical note")
    assert emb.shape[0] == 64
