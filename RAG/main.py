from rag_core import RAGPipeline
from query_enricher import QueryEnricher
from knowledge_graph import MahanamaKnowledgeGraph
import config
import warnings

def run():
    """
    Initializes and runs the RAG pipeline.
    This is the main script to run for Phase 2 (Querying).
    """
    # Suppress minor warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    if not config.OPENAI_API_KEY:
        print("=" * 50)
        print(" ERROR: OPENAI_API_KEY is not set.")
        print(" Please create a file named '.env' in this directory.")
        print(" Add the following line to it:")
        print(" OPENAI_API_KEY='sk-...'")
        print("=" * 50)
        return

    try:
        print("Initializing Mahānveşana RAG Pipeline...")
        
        # 1. Initialize Knowledge Graph
        kg = MahanamaKnowledgeGraph()
        
        # 2. Initialize Query Enricher
        enricher = QueryEnricher(kg)
        
        # 3. Initialize the main RAG Pipeline
        pipeline = RAGPipeline(enricher)

    except FileNotFoundError as e:
        print("="*50)
        print(f"ERROR: A required file was not found.")
        print(f"Details: {e}")
        print("Please make sure you have:")
        print(" 1. All 4 JSON files in the root directory.")
        print(" 2. Run 'python data_processor.py' successfully.")
        print("="*50)
        return
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        return

    # --- Run sample queries ---
    
    # This query is from your project proposal
    query1 = "What was the conversation between Janaka and Sulabha?"
    try:
        answer1 = pipeline.generate_answer(query1)
        print("\n" + "="*30 + " QUERY 1 " + "="*30)
        print(f"Query: {query1}")
        print(f"Answer: {answer1}")
        print("="*71)

    except Exception as e:
        print(f"An error occurred while processing Query 1: {e}")

    # This query tests the entity enrichment
    query2 = "What did Dhananjaya say about dharma?"
    try:
        answer2 = pipeline.generate_answer(query2)
        print("\n" + "="*30 + " QUERY 2 " + "="*30)
        print(f"Query: {query2}")
        print(f"Answer: {answer2}")
        print("="*71)
    
    except Exception as e:
        print(f"An error occurred while processing Query 2: {e}")


if __name__ == "__main__":
    run()

