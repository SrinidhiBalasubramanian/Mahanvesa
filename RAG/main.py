# --- ADDED: Centralized Warning Supression ---
# We put this at the very top, before any other imports
import os
import pandas
import warnings

# Hide TensorFlow/oneDNN spam
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Hide basic deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# --- End of Add ---

import streamlit as st
from rag_core import RAGPipeline
from query_enricher import QueryEnricher
from knowledge_graph import MahanamaKnowledgeGraph
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
        # This is the EXACT init logic from your main.py
        print("Initializing Mahānveşana RAG Pipeline (this runs once)...")
        kg = MahanamaKnowledgeGraph()
        enricher = QueryEnricher(kg)
        pipeline = RAGPipeline(enricher)
        print("...Pipeline loaded successfully.")
        return pipeline

    except FileNotFoundError as e:
        st.error(f"ERROR: A required file was not found. {e}")
        st.error("Please make sure all JSON files are present and you have run 'python data_processor.py'")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during initialization: {e}")
        return None

# --- Main Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("Mahānveşana: A Knowledge-Enhanced RAG Framework")
st.markdown("---")

# Load the RAG pipeline
pipeline = load_pipeline()

if pipeline:
    # Create the user interface
    st.header("Query the Mahabharata")
    
    query_text = st.text_area("Enter your query:", height=100)

    if st.button("Submit Query"):
        if query_text.strip():
            with st.spinner("Processing your query... This may take a moment."):
                
                # This captures the print() "stats" from the terminal
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    # This calls your working pipeline
                    answer = pipeline.generate_answer(query_text)
                
                # Get the captured stats
                stats = f.getvalue()

                # Display the results
                st.markdown("### Answer")
                st.success(answer)
                
                st.markdown("---")

                # Display the captured stats
                with st.expander("Show Processing Stats"):
                    st.text(stats)
        
        else:
            st.warning("Please enter a query.")
else:

    st.error("RAG Pipeline could not be loaded. Please check the terminal for errors.")
