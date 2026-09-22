from rag.pipeline import MedicalRAGPipeline

def test_rag_retrieval_and_citation():
    pipeline = MedicalRAGPipeline()
    res = pipeline.generate_answer("What is normal blood pressure threshold?", top_k=2)
    assert 'query' in res
    assert 'answer' in res
    assert 'citations' in res
    assert 'disclaimer' in res
    assert len(res['citations']) > 0
    assert "DISCLAIMER" in res['disclaimer']
