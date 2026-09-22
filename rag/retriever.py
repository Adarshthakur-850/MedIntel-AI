from typing import List, Dict, Any
from rag.vector_store import MedicalVectorStore

class MedicalRAGRetriever:
    def __init__(self, vector_store: MedicalVectorStore = None):
        self.vector_store = vector_store or MedicalVectorStore()

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, top_k=top_k)
