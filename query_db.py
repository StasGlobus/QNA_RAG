import sys
import asyncio
import logging
from typing import List, Optional

from rag_config import ConfigLoader
from rag_pipeline import RAGPipeline

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('query_db.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

async def main_query():
    """Loads config, initializes pipeline, and runs queries."""
    try:
        config = ConfigLoader()
        pipeline = RAGPipeline(config)
        logger.info(f"Connected to collection '{pipeline.collection_name}' ({pipeline.collection.count()} items)")

    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.critical(f"Failed to initialize the pipeline due to config error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during initialization: {e}", exc_info=True)
        sys.exit(1)

    # --- Option 1: Run predefined test questions ---
    run_test_questions = False # Default to False, or get from config if added
    # Example: run_test_questions = config.getboolean('Tests', 'run_on_startup', fallback=False)
    if run_test_questions:
        logger.info("--- Running Predefined Test Queries --- ")
        test_questions = [
    # Ambiguous or context-free queries
    "What is the model mentioned in the document?",
    "Describe the system transition mentioned in the middle of the period.",
    "How did it change over time?",
    "Was sustainability discussed consistently across sectors?",
    "Which company led the shift and how was it done?",

    # Multi-section / long-context retrieval
    "Summarize how textile manufacturing evolved from the cottage industry to digital fast fashion.",
    "Describe the key technologies that enabled automation in steel, food, and automobile production.",
    "Compare the role of Japan in the steel and textile industries across different eras.",
    "What were the different responses of Ford and GM to lean production trends?",
    "Which historical innovations impacted both the food and textile industries in the early 20th century?",

    # Missing or unverifiable information
    "What were the effects of the 2022 supply chain disruptions in steel production?",
    "Did the documents mention any AI-based food personalization platforms?",
    "Which sectors implemented blockchain for traceability first?",
    "Is there any mention of child labor regulations in textile manufacturing?",
    "Are there examples of circular economy implementation in steel production?",

    # Similar concepts or confusing distinctions
    "What is the difference between lean production and modular production in automotive manufacturing?",
    "How do electric arc furnaces and basic oxygen furnaces differ in principle and application?",
    "What distinguishes Industry 4.0 in automobile versus textile industries?",
    "Was 'digital thread' used the same way in both textile and automotive contexts?",
    "How does 'smart factory' differ from 'automated plant' in the context of food and textiles?",

    # Rare terms or unique phrasing
    "Explain the significance of 'quiet failure mode' in any of the documents.",
    "What is meant by the ‘rubber band effect’ if it is mentioned?",
    "Describe how 'platform sharing' changed vehicle manufacturing.",
    "What does the term 'war capitalism' refer to in textile production?",
    "Did the phrase 'fail fast and patch smart' appear in any manufacturing philosophy?"    
    ]
                    
        
        for q in test_questions:
            logger.info(f"--- Running Test: {q} ---")
            try:
                # Use the pipeline's query method
                context, final_answer = await pipeline.query(q, include_details=False)
                print(f"\nRetrieved Context:\n---\n{context}\n---")
                print(f"\nFinal Answer:\n{final_answer}")
                logger.info(f"CONTEXT for '{q}':\n{context}")
                logger.info(f"ANSWER for '{q}':\n{final_answer}")
            except Exception as e:
                 logger.error(f"Error processing question '{q}': {e}", exc_info=True)
            print("="*60)

    # --- Option 2: Interactive Query Loop ---
    logger.info("--- Starting Interactive Query Mode --- ")
    print("\nEnter your question (or type 'quit' to exit):")
    while True:
        try:
            user_question = input("> ")
            if user_question.lower() == 'quit': break
            if not user_question: continue

            k_val = pipeline.final_k # Default k from config
            details = False # Default to not include details
            
            # Simple command parsing for 'k=' and 'details'
            if " k=" in user_question.lower():
                parts = user_question.lower().split(" k=")
                try: 
                    k_val = int(parts[-1].strip().split()[0]) # Take only the number after k=
                    user_question = parts[0].strip()
                    logger.info(f"Using k={k_val}")
                except (ValueError, IndexError): 
                    logger.warning(f"Invalid k value, using default k={pipeline.final_k}")
                    # Attempt to recover the original question part if parsing failed
                    user_question = user_question.split(" k=")[0].strip() 
                    
            if " details" in user_question.lower():
                 details = True
                 user_question = user_question.replace(" details", "").strip()
                 logger.info("Including details in context.")


            # Use the pipeline's query method
            context, final_answer = await pipeline.query(user_question, k=k_val, include_details=details)

            print(f"\nRetrieved Context:\n---\n{context}\n---")
            print(f"\nFinal Answer:\n{final_answer}")
            # Logging is handled within the pipeline methods mostly

        except EOFError: logger.info("EOF detected, exiting."); break
        except KeyboardInterrupt: logger.info("Keyboard interrupt detected, exiting."); break
        except Exception as e:
             logger.error(f"Error during interactive loop: {e}", exc_info=True)

    logger.info("========= Exiting Query Interface =========")

if __name__ == "__main__":
    logger.info("Starting query_db.py - Query Interface")
    # Ensure dependencies are installed via requirements.txt or similar
    try:
        asyncio.run(main_query())
    except Exception as e:
        logger.critical(f"Unhandled exception in query script: {e}", exc_info=True)
        sys.exit(1)