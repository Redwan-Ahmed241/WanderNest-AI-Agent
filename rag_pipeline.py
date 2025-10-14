from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_store import VectorStore
from document_processor import DocumentProcessor
from config import Config
from typing import Dict, Any

class RAGPipeline:
    def __init__(self):
        self.config = Config()
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore()
        
        # Initialize Gemini LLM with optimized settings
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.LLM_MODEL,
            temperature=self.config.TEMPERATURE,
            google_api_key=self.config.GOOGLE_API_KEY
        )
        
        # Create WanderNest-specific prompt template (simplified)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a WanderNest travel AI assistant. Answer briefly and helpfully based on the context.

Context: {context}

Question: {question}

Answer (keep it concise):"""
        )
        
        self.qa_chain = None
        self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Set up the QA chain with retriever - optimized for speed."""
        if self.vector_store.vectorstore is not None:
            retriever = self.vector_store.get_retriever(k=self.config.TOP_K_RESULTS)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": self.prompt_template,
                    "verbose": False  # Disable verbose for speed
                },
                return_source_documents=False  # Don't return source docs for speed
            )
    
    def load_and_index_documents(self, documents_directory: str) -> None:
        """Load documents from directory and add to vector store."""
        print("Processing documents...")
        chunks = self.document_processor.process_documents(documents_directory)
        
        print("Adding to vector store...")
        self.vector_store.add_documents(chunks)
        
        # Refresh QA chain with new documents
        self._setup_qa_chain()
        print("Documents indexed successfully!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system - optimized for speed."""
        if self.qa_chain is None:
            return {
                "answer": "Hello! I'm your WanderNest AI assistant. I'm ready to help with travel questions!"
            }
        
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"]
            }
        except Exception as e:
            return {
                "answer": "I'm having trouble right now. Please try rephrasing your question."
            }
    
    def get_similar_documents(self, query: str, k: int = None) -> list:
        """Get similar documents without generating answer."""
        if k is None:
            k = self.config.TOP_K_RESULTS
        
        return self.vector_store.similarity_search(query, k=k)
