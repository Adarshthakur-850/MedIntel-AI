from typing import Dict, Any, List
from rag.retriever import MedicalRAGRetriever

SAFETY_DISCLAIMER = (
    "DISCLAIMER: MedIntel AI Knowledge Assistant is an educational and research decision-support tool. "
    "It does not provide definitive medical diagnosis, prescribe medications, or replace a qualified healthcare professional. "
    "Please consult a licensed doctor for personal medical advice."
)

class MedicalRAGPipeline:
    def __init__(self):
        self.retriever = MedicalRAGRetriever()

    def generate_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        retrieved_chunks = self.retriever.retrieve_context(query, top_k=top_k)
        
        if not retrieved_chunks:
            answer = (
                "Based on the available curated medical knowledge base, I could not find a sufficiently high-confidence match for your query. "
                "Please consult clinical practice guidelines or a certified healthcare provider."
            )
            citations = []
        else:
            context_str = "\n".join([f"[{c['doc_title']}]: {c['content']}" for c in retrieved_chunks])
            citations = [
                {
                    'doc_id': c['doc_id'],
                    'doc_title': c['doc_title'],
                    'category': c['category'],
                    'snippet': c['content'],
                    'relevance_score': c['score']
                }
                for c in retrieved_chunks
            ]
            
            # Synthesize structured medical response
            answer = (
                f"Based on curated clinical guidelines ({', '.join(set([c['doc_title'] for c in citations]))}):\n\n"
                f"{context_str}\n\n"
                "Summary: The evidence highlights key physiological indicators and clinical guidelines relevant to your query. "
                "Note that individual clinical evaluation by a physician is necessary to contextualize patient-specific risk factors."
            )

        return {
            'query': query,
            'answer': answer,
            'citations': citations,
            'disclaimer': SAFETY_DISCLAIMER
        }
