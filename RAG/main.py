import os
import warnings

# --- Warning Supression ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# --- End of Add ---

import streamlit as st
from rag_core import RAGPipeline
import config
import io
import contextlib

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
        print("Initializing Mahānveşana RAG Pipeline (this runs once)...")
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

pipeline = load_pipeline()

if pipeline:
    st.header("Query the Mahabharata")
    
    query_text = st.text_area("Enter your query:", height=100)

    if st.button("Submit Query"):
        if query_text.strip():
            with st.spinner("Processing your query... This may take a moment."):
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    rag_answer, retrieved_ids = pipeline.generate_answer(query_text)
                stats = f.getvalue()
                
                raw_context_display = pipeline.get_formatted_context(retrieved_ids)

                st.markdown("###  RAG Answer (Using Context)")
                st.success(rag_answer)
                
                st.markdown("---")

                with st.expander("Show Retrieved Raw Context (from full_text.json)"):
                    st.markdown(raw_context_display)

                with st.expander("Show RAG Processing Stats"):
                    st.text(stats)
        
        else:
            st.warning("Please enter a query.")
    
else:
    st.error("RAG Pipeline could not be loaded. Please check the terminal for errors.")
