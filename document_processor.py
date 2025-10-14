from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import os

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_documents(self, directory_path: str) -> List[Document]:
        """Load documents from a directory."""
        documents = []
        
        # Load text files
        try:
            txt_loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents.extend(txt_loader.load())
        except Exception as e:
            print(f"Error loading text files: {e}")
        
        # Load PDF files
        try:
            pdf_loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Error loading PDF files: {e}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def process_documents(self, directory_path: str) -> List[Document]:
        """Load and split documents from directory."""
        print(f"Loading documents from {directory_path}...")
        documents = self.load_documents(directory_path)
        print(f"Loaded {len(documents)} documents")
        
        print("Splitting documents into chunks...")
        chunks = self.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
