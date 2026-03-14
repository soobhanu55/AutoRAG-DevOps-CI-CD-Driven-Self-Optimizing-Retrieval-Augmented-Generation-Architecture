from langchain.text_splitter import TokenTextSplitter
from typing import List, Dict

class SlidingWindowChunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 150):
        # Sliding window is basically fixed token chunker with a large overlap (e.g. 50%)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = TokenTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        chunks = []
        for doc in documents:
            texts = self.splitter.split_text(doc["text"])
            for i, text in enumerate(texts):
                chunks.append({
                    "text": text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": i,
                        "chunk_strategy": "sliding_window"
                    }
                })
        return chunks
