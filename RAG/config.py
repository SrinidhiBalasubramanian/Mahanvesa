import os
from dotenv import load_dotenv

# Load environment variables from a .env file (for your API key)
load_dotenv()

# --- OpenAI API Key ---
# Create a file named .env in this directory and add:
# OPENAI_API_KEY="sk-..."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. Please create a .env file.")

# --- LLM Model ---
LLM_GENERATOR_MODEL = "gpt-4o"

# --- Embedding Model (for English) ---
ENG_EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Vector Database Path ---
# We use 'indexes/' to match your previous project structure
DB_PATH = "indexes/mahabharata_chapters_db"

# --- Source JSON Files ---
# These are the 4 files you provided that we will use
CHAPTER_TEXT_FILE = "chapters.json"
CHAPTER_METADATA_FILE = "chapter_entities_speakers.json"
KB_NAME_MAP_FILE = "entity_index.json"
KB_ID_MAP_FILE = "entities_kb.json"

