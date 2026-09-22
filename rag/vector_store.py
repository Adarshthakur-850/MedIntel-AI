import os
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag.knowledge_base import MEDICAL_KNOWLEDGE_DOCUMENTS

class MedicalVectorStore:
    def __init__(self):
        self.documents = MEDICAL_KNOWLEDGE_DOCUMENTS
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._build_index()

    def _build_index(self):
        self.chunks = []
        for doc in self.documents:
            # Chunking document into sentences / paragraphs
            paragraphs = [p.strip() for p in doc['content'].split('. ') if len(p.strip()) > 10]
            for idx, p in enumerate(paragraphs):
                self.chunks.append({
                    'chunk_id': f"{doc['id']}_c{idx}",
                    'doc_id': doc['id'],
                    'doc_title': doc['title'],
                    'category': doc['category'],
                    'content': p if p.endswith('.') else p + '.'
                })
        
        texts = [c['content'] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score > 0.05:
                chunk = self.chunks[idx].copy()
                chunk['score'] = round(score, 4)
                results.append(chunk)
        return results
