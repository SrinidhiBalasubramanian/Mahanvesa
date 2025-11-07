import os
# --- ADDED: Hide TensorFlow/oneDNN spam ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# --- End of Add ---
import shutil
import config
from langchain_community.vectorstores import FAISS
# --- CHANGED: Use new, non-deprecated package ---
from langchain_huggingface import HuggingFaceEmbeddings
# --- End of Change ---
from langchain_core.documents import Document
from typing import List

def create_and_save_vector_store(documents: List[Document]):
    """
    Creates, embeds, and saves a FAISS vector store from documents.
    This logic is called by data_processor.py.
    """
    print(f"Loading embedding model: {config.ENG_EMBED_MODEL}...")
    # Use 'cuda' if GPU is available, otherwise 'cpu'
    embeddings = HuggingFaceEmbeddings(  # This class is now from langchain_huggingface
        model_name=config.ENG_EMBED_MODEL,
        model_kwargs={'device': 'cpu'} 
    )

    print("Creating vector store... (This may take a moment)")
    if os.path.exists(config.DB_PATH):
        print(f"Removing existing database at {config.DB_PATH}")
        shutil.rmtree(config.DB_PATH)
        
    db = FAISS.from_documents(documents, embeddings)
    
    print("Vector store created successfully.")
    print(f"Saving to {config.DB_PATH}...")
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    db.save_local(config.DB_PATH)

def load_vector_store():
    """
    Loads an existing FAISS vector store from disk.
    This logic is called by rag_core.py.
    """
    if not os.path.exists(config.DB_PATH):
        print("="*50)
        print(f"ERROR: Vector store not found at {config.DB_PATH}")
        print("Please run 'python data_processor.py' first.")
        print("="*50)
        raise FileNotFoundError("Vector store not found.")

    print(f"Loading embedding model: {config.ENG_EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(  # This class is now from langchain_huggingface
        model_name=config.ENG_EMBED_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"Loading vector store from {config.DB_PATH}...")
    db = FAISS.load_local(
        config.DB_PATH, 
        embeddings,
        allow_dangerous_deserialization=True # Required for FAISS
    )
    return db, embeddings