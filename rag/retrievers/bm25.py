from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import numpy as np

class BM25Retriever:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        # Tokenize documents
        self.tokenized_corpus = [doc.get("text", "").lower().split(" ") for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "id": self.documents[idx].get("id", str(idx)),
                "score": float(scores[idx]),
                "text": self.documents[idx].get("text", ""),
                "metadata": self.documents[idx].get("metadata", {})
            })
            
        return results
