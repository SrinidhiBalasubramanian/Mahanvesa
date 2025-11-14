# --- ADDED: Centralized Warning Supression ---
# We put this at the very top, before any other imports
import os
import warnings

# Hide TensorFlow/oneDNN spam
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Hide basic deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# --- End of Add ---

import streamlit as st
from rag_core import RAGPipeline  # This is our newly re-written class
import config
import io
import contextlib # Used to capture terminal output

# This caches the pipeline so it only loads ONCE.
@st.cache_resource
def load_pipeline():
    """
    Initializes and returns the RAG pipeline.
    This function is cached by Streamlit.
    """
    if not config.OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set in .env file!")
        return None

    try:
        # This is the NEW init logic
        print("Initializing Mahānveşana RAG Pipeline (this runs once)...")
        # All logic is now contained within the RAGPipeline constructor
        pipeline = RAGPipeline()
        print("...Pipeline loaded successfully.")
        return pipeline

    except FileNotFoundError as e:
        st.error(f"ERROR: A required file was not found. {e}")
        st.error(f"Please check your 'dataset/' and 'embedding_models/' folders.")
        st.error(f"Make sure '{config.FULL_TEXT_FILE}' is in the correct location.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during initialization: {e}")
        return None

# --- Main Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("Mahānveṣaṇa: A Knowledge-Enhanced RAG Framework")
st.markdown("---")

# Load the RAG pipeline
pipeline = load_pipeline()

if pipeline:
    # Create the user interface
    st.header("Query the Mahabharata")
    
    query_text = st.text_area("Enter your query:", height=100)

    # --- THIS IS THE MODIFIED BLOCK ---
    if st.button("Submit Query"):
        if query_text.strip():
            with st.spinner("Processing your query... This may take a moment."):
                
                # --- 1. GET RAG ANSWER & IDs ---
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    # Get the two return values
                    rag_answer, retrieved_ids = pipeline.generate_answer(query_text)
                stats = f.getvalue()
                
                # --- 2. GET FORMATTED RAW CONTEXT ---
                # (We removed the direct answer call)
                raw_context_display = pipeline.get_formatted_context(retrieved_ids)

                # --- 3. DISPLAY EVERYTHING ---
                
                # (Removed columns)
                st.markdown("###  RAG Answer (Using Context)")
                st.success(rag_answer)
                
                st.markdown("---")

                # --- UI SECTION ---
                # Add the expander for raw context
                with st.expander("Show Retrieved Raw Context (from full_text.json)"):
                    st.markdown(raw_context_display)

                # Display the captured stats
                with st.expander("Show RAG Processing Stats"):
                    st.text(stats)
        
        else:
            st.warning("Please enter a query.")
    # --- END OF MODIFIED BLOCK ---
    
else:
    st.error("RAG Pipeline could not be loaded. Please check the terminal for errors.")