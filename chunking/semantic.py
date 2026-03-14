from langchain_experimental.text_splitter import SemanticChunker as LangchainSemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config.settings import settings
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SemanticChunker:
    def __init__(self, breakpoint_threshold_type: str = "percentile"):
        # We can use our BAAI/bge-small-en-v1.5 here too via SentenceTransformers wrapped for Langchain
        # Setting up the embeddings
        try:
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            self.splitter = LangchainSemanticChunker(
                embeddings, 
                breakpoint_threshold_type=breakpoint_threshold_type
            )
        except Exception as e:
            logger.error(f"Failed to initialize semantic chunker: {e}")
            raise

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        chunks = []
        for doc in documents:
            try:
                # Semantic Chunker from experimental takes docs or text. Let's pass text.
                texts = self.splitter.split_text(doc["text"])
                for i, text in enumerate(texts):
                    chunks.append({
                        "text": text,
                        "metadata": {
                            **doc.get("metadata", {}),
                            "chunk_index": i,
                            "chunk_strategy": "semantic"
                        }
                    })
            except Exception as e:
                logger.error(f"Error semantic chunking doc: {e}")
                # Fallback directly to the whole text
                chunks.append({
                    "text": doc["text"],
                    "metadata": {**doc.get("metadata", {}), "chunk_strategy": "semantic_fallback"}
                })
        return chunks
