from knowledge_graph import MahanamaKnowledgeGraph
import regex as re
from typing import Tuple, List

class QueryEnricher:
    """
    Implements the "Query Enrichment Module" from your diagram.
    It uses the MahanamaKnowledgeGraph to find entities in the
    user's query, expand it with aliases, and pass the
    entity IDs to the pipeline for re-ranking.
    """
    def __init__(self, kg: MahanamaKnowledgeGraph):
        self.kg = kg
        # A simple regex to "spot" potential entities (Capitalized words
        # or words longer than 4 chars).
        # A proper NER model would be an upgrade here.
        self.entity_spotter = re.compile(r'\b[A-Z][a-z]+|\b[a-z]{5,}\b')
        print("Query Enrichment Module initialized.")

    def enrich_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Enriches a user query with aliases and returns the
        list of entity IDs found.
        
        Returns:
            (enriched_query_string, found_entity_ids)
        """
        original_query = query
        
        # 1. Spot potential entities in the query
        potential_names = set(
            name.lower() for name in self.entity_spotter.findall(query)
        )
        
        found_entity_ids = set()
        enriched_terms = set()

        # 2. Look up entities in our KG
        for name in potential_names:
            ids = self.kg.get_ids_for_name(name)
            if ids:
                # Add all IDs found (e.g., "arjuna" can map to multiple IDs)
                found_entity_ids.update(ids)
                # Add aliases for all those IDs to the query
                for entity_id in ids:
                    aliases = self.kg.get_aliases_for_id(entity_id)
                    enriched_terms.update(aliases)

        # 3. Construct the new, enriched query
        if enriched_terms:
            # Remove any None or empty strings from terms
            clean_terms = [term for term in enriched_terms if term]
            enrichment_string = " ".join(list(set(clean_terms))) # Use set for unique terms
            enriched_query = f"{original_query} {enrichment_string}"
            
            print(f"Original Query: '{original_query}'")
            print(f"Found Entities: {list(found_entity_ids)}")
            print(f"Enriched Query: '{enriched_query}'")
            
            return enriched_query, list(found_entity_ids)
        else:
            print("No enrichment found. Using original query.")
            return original_query, []

# --- Example Usage ---
if __name__ == "__main__":
    try:
        kg = MahanamaKnowledgeGraph()
        enricher = QueryEnricher(kg)
        
        query1 = "What was the conversation between Janaka and Sulabha?"
        enriched_query, entity_ids = enricher.enrich_query(query1)
        print(f"IDs to re-rank with: {entity_ids}")
        
        print("-" * 20)
        
        query2 = "What did Dhananjaya say about dharma?"
        enriched_query, entity_ids = enricher.enrich_query(query2)
        print(f"IDs to re-rank with: {entity_ids}")
        
    except FileNotFoundError as e:
        print(e)
        print("Please run 'data_processor.py' first (which checks for the KB files).")

