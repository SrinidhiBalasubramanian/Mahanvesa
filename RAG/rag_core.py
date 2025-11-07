import config
import vector_store
from query_enricher import QueryEnricher
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import os
import warnings
from typing import List, Tuple

# Suppress HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

class RAGPipeline:
    """
    Implements the core Phase 2 RAG pipeline from your diagram.
    This class is initialized by main.py.
    """
    def __init__(self, enricher: QueryEnricher):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
        
        self.enricher = enricher
        
        # 1. Initialize LLM Generator (GPT)
        self.llm = ChatOpenAI(
            model_name=config.LLM_GENERATOR_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.2 # Lower temp for more factual answers
        )
        print(f"Generator LLM loaded: {config.LLM_GENERATOR_MODEL}")

        # 2. Load Vector Store (using the logic from vector_store.py)
        self.db, self.embed_model = vector_store.load_vector_store()
        
        # 3. Define the final prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are 'Mahānveşana', a specialized AI assistant for the Mahabharata. "
             "Your task is to answer the user's query based *only* on the provided context. "
             "The context consists of one or more full chapters from the Mahabharata. "
             "Read the context carefully and synthesize a coherent answer. "
             "If the context does not contain the answer, state that clearly. "
             "Do not use any outside knowledge.\n\n"
             "--- PROVIDED CONTEXT ---\n{context}"),
            ("user", "--- USER QUERY ---\n{query}")
        ])

    def _retrieve(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Retrieves k documents and their similarity scores (distances).
        Lower distance = more similar.
        """
        # --- FIXED: Changed the method name ---
        return self.db.similarity_search_with_score(query, k=k)
        # --- End of Fix ---

    def _filter_and_rerank_context(
        self, 
        retrieved_docs: List[Tuple[Document, float]],
        query_entity_ids: List[str]
    ) -> List[Document]:
        """
        This is the 'Knowledge-Enhanced' re-ranking step.
        It scores retrieved documents based on:
        1. (Primary) How many query entities they contain.
        2. (Secondary) Their original semantic similarity score.
        """
        if not query_entity_ids:
            # No entities found in query, just return top 3 semantic results
            print("No query entities. Returning top 3 semantic results.")
            # Note: score is (0=perfect, 1=bad), so we take docs with score < 1
            return [doc for doc, score in retrieved_docs[:3] if score < 1]
        
        print(f"Retrieved {len(retrieved_docs)} documents. Re-ranking based on {len(query_entity_ids)} entity IDs...")

        RankedDoc = Tuple[Document, int, float] # (doc, entity_score, semantic_score)
        ranked_list: List[RankedDoc] = []
        
        for doc, semantic_score in retrieved_docs:
            doc_entity_ids = doc.metadata.get("entities", [])
            
            # Calculate entity match score
            entity_score = len(
                set(query_entity_ids).intersection(set(doc_entity_ids))
            )
            ranked_list.append((doc, entity_score, semantic_score))

        # Re-sort the list:
        # 1. By `entity_score` (Descending: more matches is better)
        # 2. By `semantic_score` (Ascending: lower score is better)
        
        # Sort by semantic_score ascending
        sorted_list = sorted(ranked_list, key=lambda x: x[2])
        # Now, stable sort by entity_score descending
        sorted_list.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Re-ranking complete. Top doc score: {sorted_list[0][1]}")

        # Return the top 3 re-ranked documents
        final_docs = [doc for doc, e_score, s_dist in sorted_list[:3]]
        return final_docs

    def _format_context(self, docs: List[Document]) -> str:
        """Formats the retrieved documents into a string for the prompt."""
        context_str = ""
        for i, doc in enumerate(docs):
            chapter_id = doc.metadata.get('chapter_id', 'Unknown Chapter')
            context_str += f"--- CONTEXT {i+1} (Chapter {chapter_id}) ---\n"
            context_str += doc.page_content
            context_str += "\n\n"
        return context_str

    def generate_answer(self, query: str):
        """
        The main pipeline: Enrich -> Retrieve -> Re-rank -> Generate
        """
        print(f"\n--- Processing Query: {query} ---")
        
        # 1. Query Enrichment
        enriched_query, entity_ids = self.enricher.enrich_query(query)
        
        # 2. Retrieval
        retrieved_docs = self._retrieve(enriched_query, k=10) # Retrieve 10 to re-rank
        
        if not retrieved_docs:
            return "I could not find any relevant chapters for your query."

        # 3. Filter and Re-rank Context
        final_context_docs = self._filter_and_rerank_context(
            retrieved_docs, entity_ids
        )
        
        if not final_context_docs:
            # This shouldn't happen if retrieved_docs is not empty, but good to check
            return "I found some chapters, but none seemed relevant after filtering."
            
        formatted_context = self._format_context(final_context_docs)
        
        # 4. Build the final prompt
        final_prompt = self.prompt_template.format(
            context=formatted_context,
            query=query # We use the *original* query for the final answer
        )
        
        # 5. Generate Answer with LLM
        print("Generating final answer...")
        response = self.llm.invoke(final_prompt)
        
        return response.content