from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import uvicorn
from contextlib import asynccontextmanager

from rag_pipeline import RAGPipeline
from config import Config

# Global RAG pipeline instance
rag_pipeline: RAGPipeline = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = Config.TOP_K_RESULTS

class SearchResponse(BaseModel):
    documents: List[Dict[str, Any]]

async def initialize_rag_system():
    """Initialize RAG system and load documents on startup."""
    global rag_pipeline
    
    print("🚀 Initializing WanderNest RAG System...")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        
        # Check if documents folder exists
        documents_folder = Config.DOCUMENTS_FOLDER
        if os.path.exists(documents_folder) and os.listdir(documents_folder):
            print(f"📁 Loading documents from: {documents_folder}")
            try:
                rag_pipeline.load_and_index_documents(documents_folder)
                print("✅ Vector database initialized successfully!")
            except Exception as e:
                print(f"❌ Error initializing vector database: {e}")
        else:
            print(f"⚠️  Documents folder not found or empty: {documents_folder}")
            print("📝 Create the folder and add documents, then restart the server.")
    
    except Exception as e:
        print(f"❌ Critical error during RAG system initialization: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_rag_system()
    yield
    # Shutdown (cleanup if needed)
    print("🛑 Shutting down WanderNest RAG System...")

# Create FastAPI app with lifespan events
app = FastAPI(
    title="WanderNest RAG API",
    description="Fast RAG system for WanderNest travel assistance",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "WanderNest RAG API is running!",
        "status": "healthy",
        "vector_db_initialized": rag_pipeline is not None and rag_pipeline.qa_chain is not None
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system with a question."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_pipeline.query(request.question)
        return QueryResponse(answer=result["answer"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for similar documents without generating an answer."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        docs = rag_pipeline.get_similar_documents(request.query, k=request.top_k)
        
        # Format documents for API response
        formatted_docs = []
        for doc in docs:
            formatted_docs.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None)
            })
        
        return SearchResponse(documents=formatted_docs)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.post("/reload-documents")
async def reload_documents(background_tasks: BackgroundTasks):
    """Reload documents from the documents folder."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    def reload_task():
        documents_folder = Config.DOCUMENTS_FOLDER
        if os.path.exists(documents_folder) and os.listdir(documents_folder):
            try:
                rag_pipeline.load_and_index_documents(documents_folder)
                print("✅ Documents reloaded successfully!")
            except Exception as e:
                print(f"❌ Error reloading documents: {e}")
    
    background_tasks.add_task(reload_task)
    return {"message": "Document reload started in background"}

@app.get("/status")
async def get_status():
    """Get system status and statistics."""
    if rag_pipeline is None:
        return {
            "rag_initialized": False,
            "documents_folder": Config.DOCUMENTS_FOLDER,
            "vector_db_ready": False
        }
    
    vector_db_ready = rag_pipeline.qa_chain is not None
    doc_count = 0
    
    if rag_pipeline.vector_store.vectorstore is not None:
        try:
            doc_count = rag_pipeline.vector_store.vectorstore._collection.count()
        except:
            doc_count = 0
    
    return {
        "rag_initialized": True,
        "vector_db_ready": vector_db_ready,
        "documents_folder": Config.DOCUMENTS_FOLDER,
        "indexed_chunks": doc_count,
        "config": {
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.LLM_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "top_k_results": Config.TOP_K_RESULTS
        }
    }

if __name__ == "__main__":
    print("🏠 Starting WanderNest RAG Server...")
    uvicorn.run(
        "main:app",
        host=Config.SERVER_HOST,
        port=Config.SERVER_PORT,
        reload=Config.AUTO_RELOAD
    )
