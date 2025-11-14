import config
import json
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

# Import the classes we re-created
from vector_store import EmbeddingSearchEvaluator
from query_enricher import ProcessQuery

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import warnings

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Helper function to load JSONs ---

def read_json(filepath):
    """Reads a JSON file and returns the data."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    # MODIFICATION: Added encoding='utf-8' for safety
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- All the new logic from the Notebook ---

def build_entity_matrix(ch_entities: Dict[str, Dict[str, int]]):
    """Create normalized entity frequency matrix (chapters × entities)."""
    print("Building entity matrix...")
    all_entities = sorted({e for ents in ch_entities.values() for e in ents})
    df = pd.DataFrame(0, index=ch_entities.keys(), columns=all_entities)

    for ch, ents in ch_entities.items():
        for ent, freq in ents.items():
            if ent in df.columns:
                df.loc[ch, ent] = freq

    # Normalize per chapter (term frequency normalization)
    df_norm = df.div(df.sum(axis=1), axis=0).fillna(0)
    print(f"Entity matrix built. Shape: {df.shape}")
    return df, df_norm, all_entities

class GraphScorer:
    def __init__(self, df_entities, df_norm, all_entities):
        self.df_entities = df_entities
        self.df_norm = df_norm
        self.all_entities = all_entities
        self.X_tfidf = self._compute_tfidf()

    def _compute_tfidf(self):
        tfidf = TfidfTransformer(norm=None)
        X_tfidf = tfidf.fit_transform(self.df_norm.values)
        return X_tfidf

    def _get_valid_query_indices(self, query_entities, query_weights):
        valid_pairs = [
            (self.all_entities.index(e), w)
            for e, w in zip(query_entities, query_weights)
            if e in self.all_entities
        ]
        if not valid_pairs:
            return None, None
        query_indices, valid_weights = zip(*valid_pairs)
        return list(query_indices), np.array(valid_weights)

    def _group_consecutive_entities(self, entities):
        if not entities: return []
        groups = []
        current_group = [entities[0]]
        for prev, curr in zip(entities, entities[1:]):
            try:
                if int(curr[1:]) == int(prev[1:]) + 1:
                    current_group.append(curr)
                else:
                    groups.append(current_group)
                    current_group = [curr]
            except ValueError: # Handle non-numeric entity IDs if they exist
                groups.append(current_group)
                current_group = [curr]
        groups.append(current_group)
        return groups

    def _compute_group_mask(self, groups):
        mask = np.zeros((len(self.df_entities), len(groups)), dtype=int)
        for g_idx, group in enumerate(groups):
            valid_entities = [e for e in group if e in self.df_entities.columns]
            if valid_entities:
                mask[:, g_idx] = (self.df_entities[valid_entities].sum(axis=1) > 0).astype(int)
        return mask

    def _compute_direct_relevance(self, query_entities, query_indices, query_weights):
        relevance = np.ravel(self.X_tfidf[:, query_indices].dot(query_weights))
        groups = self._group_consecutive_entities(query_entities)
        if groups:
            group_mask = self._compute_group_mask(groups)
            group_coverage = group_mask.sum(axis=1) / len(groups)
            relevance = relevance * group_coverage
        if not isinstance(relevance, np.ndarray):
            relevance = relevance.toarray()
        return np.ravel(relevance)

    def _propagate_relevance(self, query_entities, relevance, k=5):
        sim_matrix = cosine_similarity(self.X_tfidf)
        topk_idx = np.argsort(-relevance)[:k]
        prop_scores = np.zeros(len(self.df_entities))
        valid_cols = [e for e in query_entities if e in self.df_entities.columns]
        
        if not valid_cols: # No valid entities, no propagation
             return prop_scores

        chapter_mask = (self.df_entities[valid_cols].sum(axis=1) > 0).astype(int).values
        for idx in topk_idx:
            masked_sim = sim_matrix[:, idx] * chapter_mask
            prop_scores += relevance[idx] * masked_sim
        return prop_scores

    def _normalize_scores(self, prop_scores, alpha=2.0):
        if np.all(prop_scores == 0):
            return prop_scores
        mean_score = np.mean(prop_scores)
        return 1 / (1 + np.exp(-alpha * (prop_scores - mean_score)))

    def score(self, query_entities, query_weights, k=5, alpha=2.0) -> Dict[str, float]:
        query_indices, query_weights = self._get_valid_query_indices(
            query_entities, query_weights
        )
        if query_indices is None:
            return {idx: 0.0 for idx in self.df_entities.index}

        relevance = self._compute_direct_relevance(
            query_entities, query_indices, query_weights
        )
        prop_scores = self._propagate_relevance(
            query_entities, relevance, k=k
        )
        scores_normalized = self._normalize_scores(prop_scores, alpha=alpha)
        
        final_scores = pd.Series(scores_normalized, index=self.df_entities.index)
        return final_scores.to_dict()

def fuse(emb_scores: Dict[str, float], graph_scores: Dict[str, float], a: float = 0.9, b: float = 0.1) -> Dict[str, float]:
    fused = {}
    for chapter, v in emb_scores.items():
        gs = graph_scores.get(chapter, None)
        fused[chapter] = v if gs is None else (a * v + b * gs)
    return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True))

def cluster_best_docs(doc_scores_dict: Dict[str, float], top_n: int = 50, num_clusters: int = 3) -> List[str]:
    ranked = sorted(doc_scores_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not ranked:
        return []

    docs = [d for d, _ in ranked]
    scores = np.array([s for _, s in ranked]).reshape(-1, 1)
    k = min(num_clusters, len(scores))
    
    if k == 0:
        return []

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(scores)
    best_label = max(range(k), key=lambda c: scores[labels == c].mean())
    best_docs = [docs[i] for i in range(len(docs)) if labels[i] == best_label]
    return best_docs

def make_context(top_ids: List[str], verses: Dict[str, str]) -> str:
    """Builds the context string from the VERSES file."""
    context = ""
    for i in top_ids:
        verses_context = {}
        v_id_prefix = f"{i.strip('M.')}."
        
        # Find all verses starting with this chapter ID
        for vid in verses.keys():
            if vid.startswith(v_id_prefix):
                verses_context[vid] = verses[vid]
        
        if verses_context:
             context += f"--- Context from Chapter {i} ---\n"
             context += str(verses_context) + "\n ========== \n"
    
    if not context:
        return "No context found for the retrieved chapter IDs."
    return context

# --- The New RAG Pipeline Class ---

class RAGPipeline:
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")

        # 1. Initialize LLM Generator
        self.llm = ChatOpenAI(
            model_name=config.LLM_GENERATOR_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0, # Set to 0 for factual answers
            max_tokens=1024
        )
        print(f"Generator LLM loaded: {config.LLM_GENERATOR_MODEL}")

        # 2. Load EmbeddingSearchEvaluator
        self.embeddings = EmbeddingSearchEvaluator()

        # 3. Load data for GraphScorer
        self.chapter_entity_ids = read_json(config.CHAPTER_ENTITY_ID_FILE)
        df_entities, df_norm, all_entities = build_entity_matrix(self.chapter_entity_ids)
        
        # 4. Initialize GraphScorer
        self.graph_scorer = GraphScorer(df_entities, df_norm, all_entities)

        # 5. Load other required JSONs
        self.entity_index = read_json(config.KB_NAME_MAP_FILE)
        self.verses = read_json(config.VERSES_FILE)
        
        # --- [START] NEW CODE ADDED ---
        self.full_text_data = read_json(config.FULL_TEXT_FILE)
        
        if not self.entity_index or not self.verses or not self.full_text_data:
            raise FileNotFoundError("Could not load entity_index.json, verses.json, or full_text.json")

        # Pre-sort keys for faster lookup in get_formatted_context
        try:
            self._sorted_full_text_keys = sorted(self.full_text_data.keys())
        except AttributeError:
            print("Warning: full_text_data is not a dictionary. Context display may fail.")
            self._sorted_full_text_keys = []
        # --- [END] NEW CODE ADDED ---

        # 6. Define the final prompt template (from notebook)
        self.system_prompt = """You are an expert on Mahabharata.
Use the information from the context to give a detailed answer and analysis. Write a descriptive essay
Be authentic and use references of chapters to back up your answer. Quote verses werever required.
When asked to retirive original verses, return the relevant verses from context verbatim
The answer should be a scholarly attempt to give a research backed answer."""
        
        self.user_prompt_template = """Given the question, answer by quoting verses from the below contexts

Question:
{query}

Context:
{context}

Answer in a clear, direct manner:
"""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt_template)
        ])
        print("✅ New RAG Pipeline Initialized Successfully.")


    def generate_answer(self, query: str, a: float = 0.9, b: float = 0.1):
        """
        The main pipeline from the notebook:
        Embed-Score -> Entity-Score -> Graph-Score -> Fuse -> Cluster -> Generate
        """
        print(f"\n--- Processing Query: {query} ---")
        
        # 1. Get Embedding Scores
        print("Step 1: Calculating embedding scores...")
        emb_scores = self.embeddings.embedding_scores(query)
        
        # 2. Process Query for Entities
        print("Step 2: Extracting entities from query...")
        p = ProcessQuery(query, self.entity_index)
        p.score_query_entities()
        query_entity_ids, query_weights = p.get_scored_entities()

        # 3. Get Graph Scores
        print("Step 3: Calculating graph scores...")
        if query_entity_ids:
            graph_scores = self.graph_scorer.score(query_entity_ids, query_weights)
        else:
            print("No entities found, skipping graph score.")
            graph_scores = {idx: 0.0 for idx in self.chapter_entity_ids.keys()}

        # 4. Fuse Scores
        print(f"Step 4: Fusing scores (a={a}, b={b})...")
        fused_scores = fuse(emb_scores, graph_scores, a, b)

        # 5. Cluster Docs
        print("Step 5: Clustering top 50 documents...")
        top_ids = cluster_best_docs(fused_scores, top_n=50, num_clusters=2)
        print(f"Retrieved {len(top_ids)} document IDs after clustering: {top_ids[:5]}...")
        
        # --- MODIFIED: Return two values ---
        if not top_ids:
            return "I could not find any relevant verses for your query.", []

        # 6. Build Context from VERSES
        print("Step 6: Building context from verses.json...")
        context = make_context(top_ids, self.verses)
        
        # 7. Build the final prompt
        final_prompt = self.prompt_template.format(
            context=context,
            query=query
        )
        
        # 8. Generate Answer with LLM
        print("Step 7: Generating final answer with LLM...")
        response = self.llm.invoke(final_prompt)
        
        # --- MODIFIED: Return two values ---
        return response.content, top_ids

    def direct_answer_openai(self, query: str):
        """
        Generates a direct answer from the model without using RAG context.
        """
        print(f"\n--- Processing Query Directly: {query} ---")
        
        # We re-use the system prompt to keep the persona, but NOT the user prompt
        # that mentions context.
        direct_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{query}") # Just the query
        ])
        
        final_prompt = direct_prompt_template.format(query=query)
        
        print("Generating direct answer with LLM...")
        response = self.llm.invoke(final_prompt)
        
        return response.content

    # --- [START] NEW METHOD ADDED ---
# --- [START] NEW METHOD ADDED ---
    def get_formatted_context(self, chapter_ids: List[str]) -> str:
        """
        Extracts and formats the SANSKRIT text from full_text.json
        for display in the Streamlit UI, based on your logic.
        """
        if not chapter_ids:
            return "No chapters were retrieved to display."

        if not self.full_text_data:
            return "Error: full_text.json data is not loaded."

        final_display_text = ""
        
        # Process all retrieved IDs (e.g., ['M.3.28'])
        for ch_id in chapter_ids:
            
            # 1. Sanitize the ID (e.g., "M.3.28" -> "3.28")
            processed_id = ch_id.lstrip('M.')
            
            # 2. Check for an exact match first (handles "M.3.28.1")
            if processed_id in self.full_text_data:
                try:
                    # --- [MODIFIED] ---
                    # Changed [1] to [0] to get Sanskrit
                    sanskrit_text = self.full_text_data[processed_id][0] 
                    # --- [END MODIFIED] ---
                    
                    final_display_text += f"### 📜 Text for {ch_id}\n\n"
                    final_display_text += f"**{processed_id}:** {sanskrit_text}\n\n---\n\n"
                    continue # Go to the next ch_id in chapter_ids
                except (IndexError, TypeError, KeyError):
                    final_display_text += f"Found {ch_id} but data was malformed.\n"
                    continue

            # 3. If no exact match, handle partial ID (e.g., "M.3.28")
            # We add a '.' to match "3.28." (e.g., "3.28.1")
            prefix = processed_id + '.'
            matching_texts = []

            # Use the pre-sorted keys from __init__
            for key in self._sorted_full_text_keys:
                if key.startswith(prefix):
                    try:
                        # --- [MODIFIED] ---
                        # Changed [1] to [0] to get Sanskrit
                        sanskrit_text = self.full_text_data[key][0]
                        # --- [END MODIFIED] ---
                        
                        # Format as: **3.28.1:** ...
                        matching_texts.append(f"**{key}:** {sanskrit_text}")
                    except (IndexError, TypeError, KeyError):
                        continue # Skip malformed data
            
            if matching_texts:
                final_display_text += f"### 📜 Full Text: Chapter {ch_id}\n\n"
                # Join all verses with a double newline
                final_display_text += "\n\n".join(matching_texts)
                final_display_text += "\n\n---\n\n"

        if not final_display_text:
            # This is the error you were seeing
            return f"No raw text found for the retrieved chapter IDs {chapter_ids} in `full_text.json`."
            
        return final_display_text
    # --- [END] NEW METHOD ADDED ---