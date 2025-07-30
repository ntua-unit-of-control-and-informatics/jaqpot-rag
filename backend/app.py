from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import List, Optional
import logging

from rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jaqpot RAG API",
    description="RAG system for Jaqpot documentation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = None

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    retrieved_chunks: List[dict]

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    try:
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline()
        await rag_pipeline.initialize()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pinecone_host": os.getenv("PINECONE_HOST"),
        "ollama_host": os.getenv("OLLAMA_HOST"),
        "pipeline_initialized": rag_pipeline is not None
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        logger.info(f"Processing query: {request.question}")
        result = await rag_pipeline.query(
            question=request.question,
            max_results=request.max_results
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            retrieved_chunks=result["retrieved_chunks"]
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get statistics about the knowledge base"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        stats = await rag_pipeline.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("BACKEND_PORT", 8000)),
        reload=True
    )