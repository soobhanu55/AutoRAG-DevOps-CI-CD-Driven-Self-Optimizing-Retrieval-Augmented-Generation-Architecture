from qdrant_client import QdrantClient
from qdrant_client.http import models
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    def upsert_documents(self, points: list[models.PointStruct]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def search(self, query_vector: list[float], limit: int = 5) -> list[models.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
