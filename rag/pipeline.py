from typing import Dict, Any, List, Optional
from rag.retrievers.dense import DenseRetriever
from rag.retrievers.bm25 import BM25Retriever
from rag.retrievers.hybrid import HybridRetriever
from reranker.cross_encoder import DocumentReranker
from generator.llm import LLMGenerator
import logging
import time

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(
        self,
        retriever_type: str,
        retriever_instance: Any,
        generator: LLMGenerator,
        reranker: Optional[DocumentReranker] = None,
        top_k_retrieve: int = 10,
        top_k_rerank: int = 5
    ):
        self.retriever_type = retriever_type
        self.retriever = retriever_instance
        self.generator = generator
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

    def query(self, query_text: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Retrieval
        retrieval_start = time.time()
        docs = self.retriever.retrieve(query_text, top_k=self.top_k_retrieve)
        retrieval_time = time.time() - retrieval_start
        
        # 2. Reranking (Optional)
        rerank_time = 0.0
        if self.reranker and len(docs) > 0:
            rerank_start = time.time()
            docs = self.reranker.rerank(query=query_text, documents=docs, top_k=self.top_k_rerank)
            rerank_time = time.time() - rerank_start
        else:
            docs = docs[:self.top_k_rerank]
            
        # 3. Generation
        gen_start = time.time()
        answer = self.generator.generate(query=query_text, context_docs=docs)
        gen_time = time.time() - gen_start
        
        total_time = time.time() - start_time
        
        return {
            "query": query_text,
            "answer": answer,
            "context": docs,
            "metrics": {
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "generation_time": gen_time,
                "total_time": total_time,
                "retriever_type": self.retriever_type,
                "reranker_used": self.reranker is not None,
                "num_context_docs": len(docs)
            }
        }
