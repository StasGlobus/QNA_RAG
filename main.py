import os
import sys
import asyncio
import logging

from rag_config import ConfigLoader
from rag_pipeline import RAGPipeline

# --- Setup Logging ---
# Keep basic logging setup for the script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---


# --- Main Execution Logic --- #
async def main_indexing():
    """Loads config and runs the indexing pipeline."""
    logger.info("========= Starting Document Ingestion (using RAGPipeline) =========")
    try:
        config = ConfigLoader() # Loads from config.ini by default
        pipeline = RAGPipeline(config)
        # The run_indexing method is already async, so we await it
        await pipeline.run_indexing()
        logger.info("========= Document Ingestion Pipeline Finished =========")
        # You might want to add a log message confirming where the DB is, like:
        # logger.info(f"Database updated/created at: {pipeline.persist_directory}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.critical(f"Failed to initialize or run the pipeline due to config error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during indexing: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting main.py - Indexing Process")
    # Dependency comments can be moved to a requirements.txt file
    try:
        asyncio.run(main_indexing())
    except KeyboardInterrupt:
        logger.info("Indexing process interrupted by user.")
        sys.exit(0)


