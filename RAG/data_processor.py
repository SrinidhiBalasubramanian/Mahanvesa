import json
import os
import config
import vector_store  # Import our vector store logic
from langchain_core.documents import Document

def load_json(filepath):
    """Helper to load a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_entity_id(entity_str: str) -> str:
    """
    Cleans an entity string like 'e10091-person--subala'
    to just its ID 'e10091'.
    """
    return entity_str.split('-')[0]

def process_and_ingest():
    """
    Runs the complete Phase 1: Ingestion and Indexing.
    This is the main script to run ONCE.
    It reads the chapter and metadata JSONs to build and save a FAISS vector store.
    """
    print("--- Starting Phase 1: Vector Store Ingestion ---")
    
    # 1. Load all source JSON files
    print("Loading source JSON files...")
    chapters_text = load_json(config.CHAPTER_TEXT_FILE)
    chapters_meta = load_json(config.CHAPTER_METADATA_FILE)
    
    # Also check for KB files, as they are needed for Phase 2
    if not load_json(config.KB_ID_MAP_FILE) or not load_json(config.KB_NAME_MAP_FILE):
        print("Error: Knowledge Base JSON files not found.")
        print(f"Please make sure '{config.KB_ID_MAP_FILE}' and '{config.KB_NAME_MAP_FILE}' are present.")
        return
        
    if not chapters_text or not chapters_meta:
        print("Error: Could not load chapter source files. Exiting.")
        return

    print(f"Loaded {len(chapters_text)} chapters from {config.CHAPTER_TEXT_FILE}")
    print(f"Loaded {len(chapters_meta)} chapters from {config.CHAPTER_METADATA_FILE}")

    # 2. Process and Combine Data
    documents = []
    
    print(f"Processing and combining {len(chapters_text)} chapters...")
    for chapter_id, text_content in chapters_text.items():
        if not text_content:
            print(f"Skipping chapter {chapter_id}: No text content.")
            continue
            
        # Get the corresponding metadata
        meta_blob = chapters_meta.get(chapter_id)
        
        if not meta_blob:
            print(f"Warning: No metadata found for chapter {chapter_id}. Indexing with empty metadata.")
            final_metadata = {"chapter_id": chapter_id, "entities": [], "speakers": []}
        else:
            # Clean entity IDs (e.g., "e12207-person--vizRu" -> "e12207")
            clean_entities = [
                clean_entity_id(e) for e in meta_blob.get("entities", [])
            ]
            
            final_metadata = {
                "chapter_id": chapter_id,
                "entities": list(set(clean_entities)), # Deduplicate
                "speakers": meta_blob.get("speakers", [])
            }
        
        # Create a LangChain Document
        doc = Document(
            page_content=text_content,
            metadata=final_metadata
        )
        documents.append(doc)

    if not documents:
        print("No documents were processed. Exiting.")
        return

    print(f"Successfully processed {len(documents)} documents.")

    # 3. Create and Save Vector Store
    vector_store.create_and_save_vector_store(documents)
    
    print("\n--- Phase 1 complete. Vector store saved. ---")
    print(f"You can now run 'python main.py' to start the RAG pipeline.")

if __name__ == "__main__":
    process_and_ingest()

