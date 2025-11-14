import os
from dotenv import load_dotenv

# Load environment variables from a .env file (for your API key)
load_dotenv()

# --- OpenAI API Key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. Please create a .env file.")

# --- LLM Model ---
LLM_GENERATOR_MODEL = "gpt-4o"

# --- Embedding Model (from Notebook) ---
# This is the model used to create the pre-computed embeddings
EMBED_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# --- Source JSON Files (from Notebook) ---
CHAPTERS_FILE = "dataset/chapters.json"
VERSES_FILE = "dataset/verses.json"
CHAPTER_ENTITY_ID_FILE = "dataset/chapter_entity_ids.json"
KB_NAME_MAP_FILE = "dataset/entity_index.json" # This is 'entity_index.json'
KB_ID_MAP_FILE = "dataset/entities_kb.json"

# --- NEW FILE FOR UI DISPLAY ---
# This is the file you just uploaded
FULL_TEXT_FILE = "dataset/full_text.json" 

# --- Pre-computed Embedding Files (from Notebook) ---
EMBEDDINGS_NPY_FILE = "embedding_models/embeddings_multi_mp.npy"
DOC_IDS_PKL_FILE = "embedding_models/doc_ids.pkl"