import numpy as np
import pickle
import faiss
import config
from sentence_transformers import SentenceTransformer

class EmbeddingSearchEvaluator:
    """
    Loads the pre-computed embeddings and model from the notebook.
    This replaces the old vector_store.py logic.
    """
    def __init__(self):
        print(f"Loading embedding model: {config.EMBED_MODEL_NAME}...")
        self.model = SentenceTransformer(config.EMBED_MODEL_NAME)
        
        print(f"Loading pre-computed embeddings from {config.EMBEDDINGS_NPY_FILE}...")
        self.embeddings = np.load(config.EMBEDDINGS_NPY_FILE)
        
        print(f"Loading doc IDs from {config.DOC_IDS_PKL_FILE}...")
        with open(config.DOC_IDS_PKL_FILE, "rb") as f:
            self.doc_ids = pickle.load(f)
            
        print(f"✅ Loaded {len(self.embeddings)} embeddings.")

        # Normalize for cosine/dot similarity
        faiss.normalize_L2(self.embeddings)

        # Build FAISS index in memory
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print("✅ FAISS index built in memory.")

    def embedding_scores(self, question: str):
        """
        Compute similarity scores of ONE question with ALL chapters.
        Returns: dict[ch_id] = similarity_score
        """
        # Encode and normalize
        query_vec = self.model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        # Search all documents
        D, I = self.index.search(query_vec, k=len(self.doc_ids))
        scores = D[0]
        ids = [self.doc_ids[i] for i in I[0]]

        # Map chapter → similarity
        return {doc_id: float(score) for doc_id, score in zip(ids, scores)}