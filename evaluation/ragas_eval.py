from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config.settings import settings
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self):
        # RAGAS needs LLM and Embeddings to evaluate
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]
        
        # Override RAGAS default models if needed via their settings, but usually env vars work
        # For evaluation, we generally use OpenAI models
        pass

    def evaluate_pipeline(self, questions: List[str], ground_truths: List[str], answers: List[str], contexts: List[List[str]]) -> Dict[str, float]:
        """
        Evaluates the RAG pipeline using RAGAS.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        try:
            logger.info("Running RAGAS evaluation...")
            result = evaluate(
                dataset,
                metrics=self.metrics,
                llm=ChatOpenAI(model_name=settings.LLM_MODEL_NAME, temperature=0),
                embeddings=OpenAIEmbeddings()
            )
            return dict(result)
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {e}")
            return {}
