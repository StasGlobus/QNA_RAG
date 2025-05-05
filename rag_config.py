import configparser
import os
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_file='config.ini'):
        if not os.path.exists(config_file):
            logger.error(f"Configuration file '{config_file}' not found.")
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        logger.info(f"Loaded configuration from '{config_file}'")
        self._validate_config()

    def _validate_config(self):
        """Basic validation for essential config sections and keys."""
        required = {
            'AzureOpenAI': ['endpoint', 'api_key', 'api_version', 'chat_deployment'],
            'Embedding': ['local_model_name'],
            'ChromaDB': ['collection_name', 'persist_directory'],
            'RAGParams': ['initial_retrieval_k', 'final_retrieval_k', 'hyde_distance_threshold', 'max_context_len_for_summary', 'rerank_by_summary'],
            'DataSource': ['documents_path']
        }
        for section, keys in required.items():
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: [{section}]")
            for key in keys:
                if key not in self.config[section]:
                     raise ValueError(f"Missing required configuration key '{key}' in section [{section}]")
        # Add more specific validation if needed (e.g., check if paths exist, numbers are numeric)

    def get(self, section, key, fallback=None):
        return self.config.get(section, key, fallback=fallback)

    def getint(self, section, key, fallback=None):
        return self.config.getint(section, key, fallback=fallback)

    def getfloat(self, section, key, fallback=None):
        return self.config.getfloat(section, key, fallback=fallback)

    def getboolean(self, section, key, fallback=None):
        return self.config.getboolean(section, key, fallback=fallback)

# Example usage (optional, for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        config = ConfigLoader()
        print("Config loaded successfully.")
        print(f"Chat Deployment: {config.get('AzureOpenAI', 'chat_deployment')}")
        print(f"Embedding Model: {config.get('Embedding', 'local_model_name')}")
        print(f"Collection Name: {config.get('ChromaDB', 'collection_name')}")
        print(f"Re-rank enabled: {config.getboolean('RAGParams', 'rerank_by_summary')}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config: {e}") 