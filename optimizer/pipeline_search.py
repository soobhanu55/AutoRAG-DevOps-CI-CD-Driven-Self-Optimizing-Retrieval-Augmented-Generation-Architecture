from typing import List, Dict, Any
from evaluation.ragas_eval import RagasEvaluator
from rag.pipeline import RAGPipeline
from rag.retrievers.dense import DenseRetriever
from rag.retrievers.bm25 import BM25Retriever
from rag.retrievers.hybrid import HybridRetriever
from reranker.cross_encoder import DocumentReranker
from generator.llm import LLMGenerator
import logging
import json
import os

logger = logging.getLogger(__name__)

class PipelineOptimizer:
    def __init__(self, vectorstore, documents, embedder):
        self.vectorstore = vectorstore
        self.documents = documents
        self.embedder = embedder
        
        self.generator = LLMGenerator()
        self.reranker = DocumentReranker()
        self.evaluator = RagasEvaluator()

    def run_optimization(self, eval_questions: List[str], eval_ground_truths: List[str]) -> Dict[str, Any]:
        """
        Runs a grid search over pipeline configurations and returns the best one.
        """
        configs = [
            {"retriever": "dense", "use_reranker": False, "top_k": 5},
            {"retriever": "dense", "use_reranker": True, "top_k": 10},
            {"retriever": "bm25", "use_reranker": False, "top_k": 5},
            {"retriever": "bm25", "use_reranker": True, "top_k": 10},
            {"retriever": "hybrid", "use_reranker": False, "top_k": 5},
            {"retriever": "hybrid", "use_reranker": True, "top_k": 10},
        ]
        
        # Initialize retrievers
        dense_ret = DenseRetriever(self.vectorstore, self.embedder)
        bm25_ret = BM25Retriever(self.documents)
        hybrid_ret = HybridRetriever(dense_ret, bm25_ret)
        
        retriever_map = {
            "dense": dense_ret,
            "bm25": bm25_ret,
            "hybrid": hybrid_ret
        }
        
        results = []
        best_score = -1.0
        best_config = None
        
        for config in configs:
            logger.info(f"Evaluating config: {config}")
            retriever_inst = retriever_map[config["retriever"]]
            rerank_inst = self.reranker if config["use_reranker"] else None
            
            pipeline = RAGPipeline(
                retriever_type=config["retriever"],
                retriever_instance=retriever_inst,
                generator=self.generator,
                reranker=rerank_inst,
                top_k_retrieve=config["top_k"],
                top_k_rerank=5
            )
            
            # Generate answers and contexts
            answers = []
            contexts = []
            for q in eval_questions:
                res = pipeline.query(q)
                answers.append(res["answer"])
                contexts.append([doc.get("text", "") for doc in res["context"]])
                
            # Evaluate
            metrics = self.evaluator.evaluate_pipeline(
                questions=eval_questions,
                ground_truths=eval_ground_truths,
                answers=answers,
                contexts=contexts
            )
            
            # Calculate composite score
            if metrics:
                # Average of available metrics
                score = sum(metrics.values()) / len(metrics)
            else:
                score = 0.0
                
            config_result = {
                "config": config,
                "metrics": metrics,
                "composite_score": score
            }
            results.append(config_result)
            
            if score > best_score:
                best_score = score
                best_config = config_result
                
        # Save results
        self._save_results(results, best_config)
                
        return best_config
        
    def _save_results(self, all_results, best_config):
        output = {
            "all_results": all_results,
            "best_config": best_config
        }
        os.makedirs("data", exist_ok=True)
        with open("data/optimization_results.json", "w") as f:
            json.dump(output, f, indent=4)
