import sys
import logging
import argparse
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rag_config import ConfigLoader
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

def re_embed_collection(old_collection_name: str, old_persist_dir: str, batch_size: int = 500):
    """
    Reads data from an old collection and upserts it into a new collection 
    defined by the current config.ini, thereby re-embedding the documents.
    """
    logger.info("--- Starting Re-embedding Process ---")
    start_time = time.monotonic()

    # 1. Load NEW configuration
    try:
        config = ConfigLoader()
        new_collection_name = config.get('ChromaDB', 'collection_name')
        new_persist_dir = config.get('ChromaDB', 'persist_directory')
        new_embedding_model = config.get('Embedding', 'local_model_name')
        logger.info(f"Target NEW Collection: '{new_collection_name}' in '{new_persist_dir}'")
        logger.info(f"Target NEW Embedding Model: '{new_embedding_model}'")
    except Exception as e:
        logger.critical(f"Failed to load configuration from config.ini: {e}", exc_info=True)
        sys.exit(1)

    if old_collection_name == new_collection_name and old_persist_dir == new_persist_dir:
        logger.error("Old and new collection/directory names are the same. Update config.ini with NEW settings.")
        sys.exit(1)
        
    old_client = None
    new_client = None

    try:
        # 2. Connect to OLD database and collection
        logger.info(f"Connecting to OLD database at: {old_persist_dir}")
        old_client = chromadb.PersistentClient(path=old_persist_dir, settings=chromadb.Settings(anonymized_telemetry=False))
        
        logger.info(f"Getting OLD collection: '{old_collection_name}'")
        # We need an embedding function instance to get the collection, but it won't be used for embedding here.
        # It's safest to try and use a dummy or the *old* model name if known, but ST works okay.
        # For simplicity, we'll instantiate one - this *might* download the old model if not cached,
        # but won't use it for actual embedding lookup here.
        # A more robust solution might involve knowing the old model name or using a dummy.
        try:
             # Attempt to get collection without specifying function first (might work depending on Chroma version/metadata)
             old_collection = old_client.get_collection(name=old_collection_name)
             logger.info("Got old collection without explicitly needing embedding function.")
        except Exception:
             logger.warning("Could not get old collection without embedding function, attempting with a temporary ST function (might download model).")
             # Fallback: instantiate a default ST function just to get the collection handle
             temp_embed_fn = SentenceTransformerEmbeddingFunction() 
             old_collection = old_client.get_collection(name=old_collection_name, embedding_function=temp_embed_fn)

        old_count = old_collection.count()
        logger.info(f"Found {old_count} items in the OLD collection.")
        if old_count == 0:
            logger.warning("Old collection is empty. Nothing to re-embed.")
            return

        # 3. Retrieve ALL data from OLD collection (batching for large collections)
        logger.info(f"Retrieving data from OLD collection (in batches of {batch_size})...")
        all_ids = []
        all_documents = []
        all_metadatas = []
        
        offset = 0
        while True:
            logger.debug(f"Fetching batch from offset {offset}...")
            results = old_collection.get(
                offset=offset,
                limit=batch_size,
                include=['metadatas', 'documents']
            )
            
            batch_ids = results.get('ids', [])
            if not batch_ids: # No more items left
                 logger.debug("Reached end of old collection.")
                 break 

            all_ids.extend(batch_ids)
            all_documents.extend(results.get('documents', []))
            all_metadatas.extend(results.get('metadatas', []))
            
            logger.info(f"Retrieved {len(all_ids)} / {old_count} items...")
            
            if len(batch_ids) < batch_size:
                 logger.debug("Last batch fetched.")
                 break # Fetched the last batch
            
            offset += len(batch_ids)

        logger.info(f"Successfully retrieved all {len(all_ids)} items from '{old_collection_name}'.")

        # Basic validation
        if not all_ids or len(all_ids) != len(all_documents) or len(all_ids) != len(all_metadatas):
             logger.error("Data retrieval error: Length mismatch between IDs, documents, and metadatas.")
             sys.exit(1)

        # 4. Connect to NEW database and collection
        logger.info(f"Connecting to NEW database at: {new_persist_dir}")
        new_client = chromadb.PersistentClient(path=new_persist_dir, settings=chromadb.Settings(anonymized_telemetry=False))
        
        logger.info(f"Initializing NEW embedding function: '{new_embedding_model}'")
        new_embed_fn = SentenceTransformerEmbeddingFunction(model_name=new_embedding_model)
        
        logger.info(f"Getting or creating NEW collection: '{new_collection_name}'")
        new_collection = new_client.get_or_create_collection(
            name=new_collection_name,
            embedding_function=new_embed_fn,
            metadata={"hnsw:space": "cosine"} # Keep consistency
        )
        logger.info(f"NEW collection '{new_collection_name}' ready.")

        # 5. Upsert retrieved data into NEW collection (batching)
        logger.info(f"Upserting {len(all_ids)} items into NEW collection (in batches of {batch_size})...")
        total_upserted = 0
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i + batch_size]
            batch_documents = all_documents[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]
            
            logger.debug(f"Upserting batch {i // batch_size + 1} ({len(batch_ids)} items)...")
            new_collection.upsert(
                ids=batch_ids,
                documents=batch_documents, # NEW embeddings generated here automatically
                metadatas=batch_metadatas
            )
            total_upserted += len(batch_ids)
            logger.info(f"Upserted {total_upserted} / {len(all_ids)} items...")

        logger.info(f"Successfully upserted {total_upserted} items into '{new_collection_name}'.")
        
        # Verification
        final_count = new_collection.count()
        logger.info(f"Final item count in NEW collection: {final_count}")
        if final_count != total_upserted:
            logger.warning(f"Potential mismatch: Upserted {total_upserted} items, but final count is {final_count}.")

    except Exception as e:
        logger.critical(f"An error occurred during the re-embedding process: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up clients if they were initialized (optional but good practice)
        # Note: ChromaDB client doesn't have an explicit close method in current APIs
        logger.debug("Cleanup: Re-embedding script finished.")
        pass 

    end_time = time.monotonic()
    duration = end_time - start_time
    logger.info(f"--- Re-embedding Process Finished (Duration: {duration:.2f} seconds) ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-embed ChromaDB collection using updated config.")
    parser.add_argument("--old-collection", required=True, help="Name of the source collection to read from.")
    parser.add_argument("--old-persist-dir", required=True, help="Path to the persist directory of the source collection.")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of items to process in each batch.")
    
    args = parser.parse_args()
    
    re_embed_collection(args.old_collection, args.old_persist_dir, args.batch_size) 