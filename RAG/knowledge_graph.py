import json
import config
import os

class MahanamaKnowledgeGraph:
    """
    This class is the "Mahanama Knowledge Graph" interface from your diagram.
    It loads the pre-processed KB and index files into memory to provide
    fast lookups for the Query Enrichment Module.
    """
    
    def __init__(self):
        self.id_to_details = self._load_map(config.KB_ID_MAP_FILE)
        self.name_to_ids = self._load_map(config.KB_NAME_MAP_FILE)
        
        if not self.id_to_details or not self.name_to_ids:
            print("="*50)
            print("ERROR: Knowledge Base JSON files not found!")
            print(f"Please make sure '{config.KB_ID_MAP_FILE}' and")
            print(f"'{config.KB_NAME_MAP_FILE}' are in the directory.")
            print("="*50)
            raise FileNotFoundError("Could not load Knowledge Base maps.")
            
        print("Mahanama Knowledge Graph lookups loaded.")

    def _load_map(self, filepath: str):
        """Loads a JSON map from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def get_ids_for_name(self, name: str) -> list[str]:
        """
        Gets all possible entity IDs for a given name (e.g., "arjuna").
        (Uses entity_index.json)
        """
        return self.name_to_ids.get(name.lower(), [])

    def get_details_for_id(self, entity_id: str) -> dict:
        """
        Gets details (name, aliases) for a given ID (e.g., "e802").
        (Uses entities_kb.json)
        """
        return self.id_to_details.get(entity_id, {})

    def get_aliases_for_id(self, entity_id: str) -> list[str]:
        """
        Gets all TEXT aliases for a specific entity ID.
        (Uses entities_kb.json)
        
        --- BUG FIX ---
        The 'aliases' field in entities_kb.json contains
        other *entity IDs*, not text. We must only return the
        'key', which is the primary text for that ID, to avoid
        polluting the enriched query string.
        """
        details = self.get_details_for_id(entity_id)
        if details:
            # The 'key' is the primary name, e.g., "arjuna" or "sulaBA"
            key = details.get('key')
            if key:
                # We ONLY return the text key.
                return [key]
        return []

# --- Example Usage ---
if __name__ == "__main__":
    try:
        kg = MahanamaKnowledgeGraph()
        
        name = "dhananjaya"
        ids = kg.get_ids_for_name(name)
        print(f"IDs for '{name}': {ids}") # e.g., ['e802']
        
        if ids:
            entity_id = ids[0] 
            details = kg.get_details_for_id(entity_id)
            print(f"Details for {entity_id}: {details.get('key')}") # e.g., 'arjuna'
            
            aliases = kg.get_aliases_for_id(entity_id)
            print(f"TEXT aliases for {entity_id}: {aliases}") # e.g., ['arjuna']

    except FileNotFoundError as e:
        print(e)

