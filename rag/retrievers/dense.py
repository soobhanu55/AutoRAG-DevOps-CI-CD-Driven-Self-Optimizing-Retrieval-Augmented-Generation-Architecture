from typing import List, Dict, Any
from vectorstore.qdrant_client import QdrantStore
from ingestion.embedder import BGEEmbedder

class DenseRetriever:
    def __init__(self, vectorstore: QdrantStore, embedder: BGEEmbedder):
        self.vectorstore = vectorstore
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)
        scored_points = self.vectorstore.search(query_vector=query_vector, limit=top_k)
        
        results = []
        for point in scored_points:
            results.append({
                "id": point.id,
                "score": point.score,
                "text": point.payload.get("text", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "text"}
            })
            
        return results
