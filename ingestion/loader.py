import os
import fitz  # PyMuPDF
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: str) -> List[Dict]:
        doc = fitz.open(file_path)
        documents = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": file_path,
                        "page": page_num + 1,
                        "type": "pdf"
                    }
                })
        return documents

    @staticmethod
    def load_txt(file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [{
            "text": text,
            "metadata": {
                "source": file_path,
                "type": "txt"
            }
        }]

    @staticmethod
    def load_markdown(file_path: str) -> List[Dict]:
        # Simple extraction for now, can be improved with markdown parsing
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        return [{
            "text": text,
            "metadata": {
                "source": file_path,
                "type": "markdown"
            }
        }]

    @staticmethod
    def load_file(file_path: str) -> List[Dict]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return DocumentLoader.load_pdf(file_path)
        elif ext == '.txt':
            return DocumentLoader.load_txt(file_path)
        elif ext in ['.md', '.markdown']:
            return DocumentLoader.load_markdown(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return []
            
    @staticmethod
    def load_directory(dir_path: str) -> List[Dict]:
        all_docs = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                docs = DocumentLoader.load_file(file_path)
                all_docs.extend(docs)
        return all_docs
