import config
import json
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

from vector_store import EmbeddingSearchEvaluator
from query_enricher import ProcessQuery

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def read_json(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_entity_matrix(ch_entities: Dict[str, Dict[str, int]]):
    print("Building entity matrix...")
    all_entities = sorted({e for ents in ch_entities.values() for e in ents})
    df = pd.DataFrame(0, index=ch_entities.keys(), columns=all_entities)

    for ch, ents in ch_entities.items():
        for ent, freq in ents.items():
            if ent in df.columns:
                df.loc[ch, ent] = freq

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
            except ValueError:
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
        
        if not valid_cols:
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
    context = ""
    for i in top_ids:
        verses_context = {}
        v_id_prefix = f"{i.strip('M.')}."
        
        for vid in verses.keys():
            if vid.startswith(v_id_prefix):
                verses_context[vid] = verses[vid]
        
        if verses_context:
             context += f"--- Context from Chapter {i} ---\n"
             context += str(verses_context) + "\n ========== \n"
    
    if not context:
        return "No context found for the retrieved chapter IDs."
    return context


class RAGPipeline:
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.llm = ChatOpenAI(
            model_name=config.LLM_GENERATOR_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0,
            max_tokens=1024
        )
        print(f"Generator LLM loaded: {config.LLM_GENERATOR_MODEL}")

        self.embeddings = EmbeddingSearchEvaluator()

        self.chapter_entity_ids = read_json(config.CHAPTER_ENTITY_ID_FILE)
        df_entities, df_norm, all_entities = build_entity_matrix(self.chapter_entity_ids)
        
        self.graph_scorer = GraphScorer(df_entities, df_norm, all_entities)

        self.entity_index = read_json(config.KB_NAME_MAP_FILE)
        self.verses = read_json(config.VERSES_FILE)
        
        self.full_text_data = read_json(config.FULL_TEXT_FILE)
        
        if not self.entity_index or not self.verses or not self.full_text_data:
            raise FileNotFoundError("Could not load entity_index.json, verses.json, or full_text.json")

        try:
            self._sorted_full_text_keys = sorted(self.full_text_data.keys())
        except AttributeError:
            print("Warning: full_text_data is not a dictionary. Context display may fail.")
            self._sorted_full_text_keys = []

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
        print(f"\n--- Processing Query: {query} ---")
        
        print("Step 1: Calculating embedding scores...")
        emb_scores = self.embeddings.embedding_scores(query)
        
        print("Step 2: Extracting entities from query...")
        p = ProcessQuery(query, self.entity_index)
        p.score_query_entities()
        query_entity_ids, query_weights = p.get_scored_entities()

        print("Step 3: Calculating graph scores...")
        if query_entity_ids:
            graph_scores = self.graph_scorer.score(query_entity_ids, query_weights)
        else:
            print("No entities found, skipping graph score.")
            graph_scores = {idx: 0.0 for idx in self.chapter_entity_ids.keys()}

        print(f"Step 4: Fusing scores (a={a}, b={b})...")
        fused_scores = fuse(emb_scores, graph_scores, a, b)

        print("Step 5: Clustering top 50 documents...")
        top_ids = cluster_best_docs(fused_scores, top_n=50, num_clusters=2)
        print(f"Retrieved {len(top_ids)} document IDs after clustering: {top_ids[:5]}...")
        
        if not top_ids:
            return "I could not find any relevant verses for your query.", []

        print("Step 6: Building context from verses.json...")
        context = make_context(top_ids, self.verses)
        
        final_prompt = self.prompt_template.format(
            context=context,
            query=query
        )
        
        print("Step 7: Generating final answer with LLM...")
        response = self.llm.invoke(final_prompt)
        
        return response.content, top_ids

    def direct_answer_openai(self, query: str):
        print(f"\n--- Processing Query Directly: {query} ---")
        
        direct_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{query}")
        ])
        
        final_prompt = direct_prompt_template.format(query=query)
        
        print("Generating direct answer with LLM...")
        response = self.llm.invoke(final_prompt)
        
        return response.content

    def get_formatted_context(self, chapter_ids: List[str]) -> str:
        if not chapter_ids:
            return "No chapters were retrieved to display."

        if not self.full_text_data:
            return "Error: full_text.json data is not loaded."

        final_display_text = ""
        
        for ch_id in chapter_ids:
            
            processed_id = ch_id.lstrip('M.')
            
            if processed_id in self.full_text_data:
                try:
                    sanskrit_text = self.full_text_data[processed_id][0] 
                    
                    final_display_text += f"### 📜 Text for {ch_id}\n\n"
                    final_display_text += f"**{processed_id}:** {sanskrit_text}\n\n---\n\n"
                    continue
                except (IndexError, TypeError, KeyError):
                    final_display_text += f"Found {ch_id} but data was malformed.\n"
                    continue

            prefix = processed_id + '.'
            matching_texts = []

            for key in self._sorted_full_text_keys:
                if key.startswith(prefix):
                    try:
                        sanskrit_text = self.full_text_data[key][0]
                        
                        matching_texts.append(f"**{key}:** {sanskrit_text}")
                    except (IndexError, TypeError, KeyError):
                        continue
            
            if matching_texts:
                final_display_text += f"### 📜 Full Text: Chapter {ch_id}\n\n"
                final_display_text += "\n\n".join(matching_texts)
                final_display_text += "\n\n---\n\n"

        if not final_display_text:
            return f"No raw text found for the retrieved chapter IDs {chapter_ids} in `full_text.json`."
            
        return final_display_text
