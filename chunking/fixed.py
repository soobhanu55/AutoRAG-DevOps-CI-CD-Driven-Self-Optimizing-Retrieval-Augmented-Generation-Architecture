from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

class FixedChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
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
                        "chunk_strategy": "fixed"
                    }
                })
        return chunks
