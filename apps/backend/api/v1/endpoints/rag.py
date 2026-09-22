from fastapi import APIRouter, HTTPException
from apps.backend.schemas.api_schemas import RAGQueryRequest, RAGQueryResponse
from rag.pipeline import MedicalRAGPipeline

router = APIRouter()
rag_pipeline = MedicalRAGPipeline()

@router.post("/query", response_model=RAGQueryResponse)
def query_medical_rag(req: RAGQueryRequest):
    try:
        res = rag_pipeline.generate_answer(query=req.query, top_k=req.top_k)
        return RAGQueryResponse(
            query=res['query'],
            answer=res['answer'],
            citations=res['citations'],
            disclaimer=res['disclaimer']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
