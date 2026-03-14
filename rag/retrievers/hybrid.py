from typing import List, Dict, Any
from rag.retrievers.dense import DenseRetriever
from rag.retrievers.bm25 import BM25Retriever
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, alpha: float = 0.5):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha  # Weight for dense score

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return results
            
        scores = [res["score"] for res in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score > min_score:
            for res in results:
                res["normalized_score"] = (res["score"] - min_score) / (max_score - min_score)
        else:
            for res in results:
                res["normalized_score"] = 1.0
                
        return results

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Retrieve more from each to ensure overlap
        k_retrieve = max(top_k * 2, 20)
        
        dense_results = self.dense_retriever.retrieve(query, top_k=k_retrieve)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=k_retrieve)
        
        dense_results = self._normalize_scores(dense_results)
        bm25_results = self._normalize_scores(bm25_results)
        
        # Merge results combining scores
        combined_scores = {}
        result_map = {}
        
        for res in dense_results:
            doc_id = res.get("id", res.get("text", "")[:50]) # Fallback id
            combined_scores[doc_id] = self.alpha * res["normalized_score"]
            result_map[doc_id] = res
            
        for res in bm25_results:
            doc_id = res.get("id", res.get("text", "")[:50])
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1.0 - self.alpha) * res["normalized_score"]
            else:
                combined_scores[doc_id] = (1.0 - self.alpha) * res["normalized_score"]
                result_map[doc_id] = res
                
        # Sort and select top_k
        sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        
        final_results = []
        for doc_id in sorted_ids:
            doc = result_map[doc_id]
            doc["score"] = combined_scores[doc_id]
            doc["retrieval_method"] = "hybrid"
            final_results.append(doc)
            
        return final_results
