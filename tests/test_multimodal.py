import numpy as np
from ml.multimodal.fusion import MultimodalFusionEngine

def test_late_fusion_calculation():
    engine = MultimodalFusionEngine(tabular_weight=0.5, vision_weight=0.3, nlp_weight=0.2)
    score = engine.predict_late_fusion(0.8, 0.6, 0.4)
    expected = (0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4) / 1.0
    assert abs(score - expected) < 1e-4

def test_neural_fusion_forward():
    engine = MultimodalFusionEngine()
    t_emb = np.random.randn(15)
    i_emb = np.random.randn(512)
    x_emb = np.random.randn(64)
    
    score = engine.predict_neural_fusion(t_emb, i_emb, x_emb)
    assert 0.0 <= score <= 1.0

def test_fuse_all_modalities():
    engine = MultimodalFusionEngine()
    t_emb = np.random.randn(15)
    i_emb = np.random.randn(512)
    x_emb = np.random.randn(64)
    
    res = engine.fuse_all_modalities(0.7, 0.8, 0.6, t_emb, i_emb, x_emb, method="neural")
    assert 'final_fused_risk' in res
    assert 'disclaimer' in res
