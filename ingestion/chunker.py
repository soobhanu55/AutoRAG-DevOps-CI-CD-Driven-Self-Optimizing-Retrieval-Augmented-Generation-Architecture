from typing import List, Dict, Optional
from chunking.fixed import FixedChunker
from chunking.semantic import SemanticChunker
from chunking.sliding_window import SlidingWindowChunker
import logging

logger = logging.getLogger(__name__)

class MasterChunker:
    def __init__(self):
        self.strategies = {
            "fixed": FixedChunker,
            "semantic": SemanticChunker,
            "sliding_window": SlidingWindowChunker
        }

    def chunk_documents(self, documents: List[Dict], strategy: str = "fixed", **kwargs) -> List[Dict]:
        if strategy not in self.strategies:
            logger.warning(f"Strategy {strategy} not found. Defaulting to fixed.")
            strategy = "fixed"
            
        chunker_class = self.strategies[strategy]
        
        # Instantiate with kwargs based on strategy
        try:
            if strategy == "semantic":
                # Semantic chunker takes breakpoint_threshold_type
                threshold = kwargs.get("breakpoint_threshold_type", "percentile")
                chunker = chunker_class(breakpoint_threshold_type=threshold)
            else:
                chunk_size = kwargs.get("chunk_size", 500)
                chunk_overlap = kwargs.get("chunk_overlap", int(chunk_size * 0.1))
                chunker = chunker_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
            logger.info(f"Chunking {len(documents)} documents using {strategy} strategy.")
            return chunker.split_documents(documents)
            
        except Exception as e:
            logger.error(f"Error during chunking setup: {e}")
            raise
