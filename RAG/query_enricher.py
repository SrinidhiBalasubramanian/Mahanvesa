from rapidfuzz import fuzz, process
from typing import Dict, List, Tuple

class ProcessQuery:
    """
    Extracts entities from the query using RapidFuzz.
    This replaces the old QueryEnricher class.
    """
    def __init__(self, query: str, entity_index: Dict):
        self.query = query
        self.entity_index = entity_index
        self.query_entities: Dict[str, float] = {}

    def _extract_query_entities(self, query_word: str, threshold: int = 90) -> Tuple[List[str], List[float]]:
        """Finds entities matching a single query word."""
        matches = process.extract(
            query_word, 
            list(self.entity_index.keys()), 
            scorer=fuzz.token_sort_ratio, 
            limit=5
        )
        
        matched_entities = []
        scores = []
        for _match, score, _ in matches:
            if float(score) >= threshold:
                matched_entities.append(self.entity_index[_match])
                scores.append(score)
        return matched_entities, scores

    def score_query_entities(self):
        """
        Finds all entities in the query and returns them.
        Populates self.query_entities
        """
        print(f"Scoring query entities for: '{self.query}'")
        for query_word in self.query.split():
            entities, scores = self._extract_query_entities(query_word.lower())
            if entities:
                for _e in entities:
                    for e in _e:
                        # Assign the highest score if entity is found multiple times
                        if e not in self.query_entities or scores[0] > self.query_entities[e]:
                            self.query_entities[e] = scores[0]
        
        print(f"Found {len(self.query_entities)} entities: {list(self.query_entities.keys())}")

    def get_scored_entities(self) -> Tuple[List[str], List[float]]:
        """Returns the found entity IDs and their scores."""
        if not self.query_entities:
            return [], []
        
        entity_ids = list(self.query_entities.keys())
        query_weights = list(self.query_entities.values())
        return entity_ids, query_weights