from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil
import json

from config.settings import settings
from vectorstore.qdrant_client import QdrantStore
from ingestion.loader import DocumentLoader
from ingestion.embedder import BGEEmbedder
from ingestion.chunker import MasterChunker
from rag.pipeline import RAGPipeline
from rag.retrievers.dense import DenseRetriever
from rag.retrievers.bm25 import BM25Retriever
from rag.retrievers.hybrid import HybridRetriever
from reranker.cross_encoder import DocumentReranker
from generator.llm import LLMGenerator
from optimizer.pipeline_search import PipelineOptimizer

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Global instances (in a real app, use dependency injection)
vectorstore = None
embedder = None
chunker = None
generator = None
reranker = None
bm25_retriever = None # Requires documents to build
active_pipeline = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class OptimizeRequest(BaseModel):
    questions: List[str]
    ground_truths: List[str]

@app.on_event("startup")
async def startup_event():
    global vectorstore, embedder, chunker, generator, reranker, active_pipeline
    
    # Initialize Core Components
    vectorstore = QdrantStore()
    embedder = BGEEmbedder()
    chunker = MasterChunker()
    generator = LLMGenerator()
    try:
        reranker = DocumentReranker()
    except Exception as e:
        print(f"Warning: Reranker failed to load: {e}")
        reranker = None
        
    # Set a default active pipeline (Dense)
    dense_ret = DenseRetriever(vectorstore, embedder)
    active_pipeline = RAGPipeline(
        retriever_type="dense",
        retriever_instance=dense_ret,
        generator=generator,
        reranker=reranker,
        top_k_retrieve=10,
        top_k_rerank=5
    )


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...), strategy: str = "fixed"):
    try:
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        file_path = os.path.join(settings.DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        docs = DocumentLoader.load_file(file_path)
        chunks = chunker.chunk_documents(docs, strategy=strategy)
        
        # Build vectors and upsert
        from qdrant_client.http import models
        points = []
        import uuid
        
        for chunk in chunks:
            vector = embedder.embed_query(chunk["text"])
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk["text"], **chunk["metadata"]}
            )
            points.append(point)
            
        vectorstore.upsert_documents(points)
        
        # Need to rebuild BM25 index if we use it, but keeping it simple.
        
        return {"message": "Success", "chunks_processed": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_pipeline(request: QueryRequest):
    if not active_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    result = active_pipeline.query(request.query)
    return result


@app.post("/evaluate")
async def evaluate_pipeline():
    # Placeholder for evaluating against a static dataset
    return {"message": "Evaluation endpoint triggered"}


@app.post("/optimize")
async def optimize_pipeline(request: OptimizeRequest, background_tasks: BackgroundTasks):
    global active_pipeline
    
    if not request.questions or not request.ground_truths or len(request.questions) != len(request.ground_truths):
        raise HTTPException(status_code=400, detail="Questions and ground_truths must be provided and have same length.")
        
    # Load all documents for BM25 (simplified, ideally retrieve from Qdrant scroll API)
    # For now, we mock passing documents
    all_docs = [{"id": str(i), "text": f"Doc {i}"} for i in range(10)]
    
    optimizer = PipelineOptimizer(vectorstore, all_docs, embedder)
    
    # Run optimizer
    best_config = optimizer.run_optimization(request.questions, request.ground_truths)
    
    return {"message": "Optimization completed.", "best_config": best_config}


@app.get("/config")
async def get_config():
    return {
        "active_retriever": active_pipeline.retriever_type if active_pipeline else "None",
        "reranker_enabled": active_pipeline.reranker is not None if active_pipeline else False
    }


@app.get("/metrics")
async def get_metrics():
    # Return last optimization results if available
    results_path = "data/optimization_results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return {"message": "No metrics available yet."}
