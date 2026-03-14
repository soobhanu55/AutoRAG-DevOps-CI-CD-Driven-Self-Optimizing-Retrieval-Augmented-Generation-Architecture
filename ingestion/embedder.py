from sentence_transformers import SentenceTransformer
from typing import List
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class BGEEmbedder:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> List[float]:
        # BGE models typically require 'Represent this sentence for searching relevant passages:' for queries
        # Adding specifically for bge-*-en-v1.5
        instruction = "Represent this sentence for searching relevant passages: "
        if "bge" in settings.EMBEDDING_MODEL_NAME.lower():
            query = instruction + query
        return self.model.encode(query).tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        # Documents don't need instruction
        return self.model.encode(documents).tolist()
