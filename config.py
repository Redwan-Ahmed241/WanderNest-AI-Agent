import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Google Gemini Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Chroma Configuration
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "e:/WanderNest-Agent/chroma_db")
    COLLECTION_NAME = "wandernest_documents"
    
    # Document Processing - optimized for speed
    CHUNK_SIZE = 800  # Smaller chunks for faster processing
    CHUNK_OVERLAP = 100  # Reduced overlap
    
    # Retrieval Settings - optimized for speed
    TOP_K_RESULTS = 3  # Fewer results for faster retrieval
    
    # Model Settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast local model
    LLM_MODEL = "gemini-2.5-flash"  # Correct Gemini model name
    TEMPERATURE = 0.3  # Lower temperature for faster, more focused responses
    
    # Embedding Configuration
    USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    
    # FastAPI Configuration
    DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "e:/WanderNest-Agent/documents")
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
    AUTO_RELOAD = os.getenv("AUTO_RELOAD", "true").lower() == "true"
