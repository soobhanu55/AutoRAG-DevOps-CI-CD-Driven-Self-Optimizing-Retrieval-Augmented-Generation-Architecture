from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
from config.settings import settings
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DocumentReranker:
    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        if not documents:
            return []
            
        if top_k is None:
            top_k = len(documents)
            
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.get("text", "")] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents and sort
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
            
        sorted_documents = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        return sorted_documents[:top_k]
