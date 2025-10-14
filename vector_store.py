from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from typing import List, Optional
import chromadb
from config import Config

class VectorStore:
    def __init__(self, persist_directory: str = Config.CHROMA_PERSIST_DIR, 
                 collection_name: str = Config.COLLECTION_NAME):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings based on configuration
        if Config.USE_LOCAL_EMBEDDINGS:
            print("🔧 Using local HuggingFace embeddings (no quota limits)")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
        else:
            print("🔧 Using Gemini embeddings")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=Config.GOOGLE_API_KEY
            )
        
        # Initialize or load Chroma vector store
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store."""
        try:
            # Try to load existing vectorstore
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print(f"Loaded existing vector store with {self.vectorstore._collection.count()} documents")
        except Exception as e:
            print(f"Creating new vector store: {e}")
            self.vectorstore = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            print("No documents to add")
            return
        
        if self.vectorstore is None:
            # Create new vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
        else:
            # Add to existing vectorstore
            self.vectorstore.add_documents(documents)
        
        # Persist the vectorstore
        self.vectorstore.persist()
        print(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = Config.TOP_K_RESULTS) -> List[Document]:
        """Perform similarity search."""
        if self.vectorstore is None:
            print("Vector store not initialized")
            return []
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_retriever(self, k: int = Config.TOP_K_RESULTS):
        """Get retriever for the vector store."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
