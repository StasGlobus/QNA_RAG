import os
import sys
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple

import openai # For APIError
from openai import AsyncAzureOpenAI
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import numpy as np
from docx import Document
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from rag_config import ConfigLoader # Import config loader

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Encapsulates the RAG pipeline logic for indexing and querying."""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self._setup_logging()
        
        # Load config values
        self.azure_endpoint = config.get('AzureOpenAI', 'endpoint')
        self.azure_api_key = config.get('AzureOpenAI', 'api_key')
        self.azure_api_version = config.get('AzureOpenAI', 'api_version')
        self.azure_chat_deployment = config.get('AzureOpenAI', 'chat_deployment')
        
        self.embedding_model_name = config.get('Embedding', 'local_model_name')
        
        self.collection_name = config.get('ChromaDB', 'collection_name')
        self.persist_directory = config.get('ChromaDB', 'persist_directory')

        self.initial_k = config.getint('RAGParams', 'initial_retrieval_k', fallback=10)
        self.final_k = config.getint('RAGParams', 'final_retrieval_k', fallback=3)
        self.hyde_threshold = config.getfloat('RAGParams', 'hyde_distance_threshold', fallback=0.8)
        self.max_context_len_summary = config.getint('RAGParams', 'max_context_len_for_summary', fallback=3000)
        self.rerank_by_summary_enabled = config.getboolean('RAGParams', 'rerank_by_summary', fallback=True)
        
        # Initialize clients and embedding resources
        self.chat_client = self._initialize_chat_client()
        self.st_model, self.embed_fn = self._initialize_embedding_resources()
        self.db_client = self._initialize_chromadb_client()
        self.collection = self._get_or_create_collection()

    def _setup_logging(self):
        # Configure logging (can be more sophisticated if needed)
        log_level_str = self.config.get('Logging', 'level', fallback='INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_format = self.config.get('Logging', 'format', 
                                     fallback='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        
        # Basic config remains, consider file handler based on config later
        logging.basicConfig(level=log_level, format=log_format)
        logger.info("Logging setup complete.")

    def _initialize_chat_client(self) -> AsyncAzureOpenAI:
        logger.info("Initializing Azure OpenAI client for Chat...")
        try:
            client = AsyncAzureOpenAI(
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
                azure_endpoint=self.azure_endpoint
            )
            # Optional: Add a simple test call here if needed
            return client
        except Exception as e:
            logger.exception("Failed to initialize Azure OpenAI client.")
            sys.exit(1)

    def _initialize_embedding_resources(self) -> Tuple[SentenceTransformer, SentenceTransformerEmbeddingFunction]:
        logger.info(f"Initializing Sentence Transformer resources (Model: {self.embedding_model_name})...")
        try:
            st_model = SentenceTransformer(self.embedding_model_name)
            embed_fn = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name)
            logger.info("Sentence Transformer model and ChromaDB function loaded successfully.")
            return st_model, embed_fn
        except Exception as e:
            logger.exception(f"Failed to initialize Sentence Transformer model '{self.embedding_model_name}'. Ensure 'sentence-transformers' is installed.")
            sys.exit(1)

    def _initialize_chromadb_client(self) -> chromadb.PersistentClient:
        logger.info(f"Initializing ChromaDB client (Directory: {self.persist_directory})...")
        try:
            chroma_client = chromadb.PersistentClient(
                path=self.persist_directory, 
                settings=chromadb.Settings(anonymized_telemetry=False)
            )
            return chroma_client
        except Exception as e:
            logger.exception("Failed to initialize ChromaDB persistent client.")
            sys.exit(1)

    def _get_or_create_collection(self) -> chromadb.Collection:
        logger.info(f"Getting or creating ChromaDB collection: {self.collection_name}")
        try:
            collection = self.db_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' loaded/created. Initial item count: {collection.count()}")
            return collection
        except Exception as e:
            logger.exception("Failed to get or create ChromaDB collection.")
            sys.exit(1)

    # --- Indexing Methods --- 
    
    def _extract_word_docs(self, folder_path: str) -> Dict[str, List[Dict[str, str]]]:
        """Extracts content from .docx files, identifying headers.
        """
        content_dict = {}
        logger.info(f"--- Starting Document Extraction from: {folder_path} ---")
        if not os.path.isdir(folder_path):
            logger.error(f"Source documents folder not found at: {folder_path}")
            return {}
        try:
            all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.docx') and not f.startswith('~')]
        except OSError as e:
            logger.error(f"Error reading source documents directory {folder_path}: {e}")
            return {}

        processed_files = 0
        logger.info(f"Found {len(all_files)} .docx files to process.")
        for filename in tqdm(all_files, desc="Extracting Documents"):
            file_path = os.path.join(folder_path, filename)
            doc_name = os.path.splitext(filename)[0]
            try:
                doc = Document(file_path)
                content = []
                current_heading = None
                for para in doc.paragraphs:
                    text = para.text.strip()
                    if text:
                        style_name = para.style.name.lower()
                        is_header = 'heading' in style_name
                        if not is_header:
                           runs = para.runs
                           is_bold = all(run.bold for run in runs if run.text.strip()) and runs
                           if is_bold: is_header = True
                        if is_header:
                            content.append({"type": "header", "text": text})
                            current_heading = text
                        else:
                            content.append({"type": "paragraph", "text": text, "associated_header": current_heading})
                if content:
                    content_dict[doc_name] = content
                    processed_files += 1
                else:
                    logger.warning(f"No text content extracted from {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}", exc_info=True)
        logger.info(f"--- Extraction Complete --- Extracted content from {processed_files}/{len(all_files)} documents.")
        return content_dict

    async def _generate_summary_and_questions(self, text: str, header: Optional[str]) -> tuple[str, List[str]]:
        """(Helper) Generate summary/questions using the Azure Chat client."""
        header_context = f" about {header}" if header else ""
        summary = f"Default summary for section{header_context}."
        questions = [f"What are the key points in section{header_context}?"]
        try:
            summary_prompt = (
                f"Please provide a concise single-paragraph summary of the following text{header_context}:\n\n"
                f"{text}"
            )
            summary_response = await self.chat_client.chat.completions.create(
                model=self.azure_chat_deployment,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.1, max_tokens=150
            )
            if summary_response.choices and summary_response.choices[0].message:
                summary = summary_response.choices[0].message.content.strip()
                logger.debug(f"Generated Summary for '{header}': {summary[:100]}...")
            else: logger.warning(f"Unexpected summary response structure for header '{header}'. Using default.")

            questions_prompt = (
                f"Generate 3 specific and distinct questions that the following text{header_context} can answer:\n\n"
                f"{text}"
            )
            questions_response = await self.chat_client.chat.completions.create(
                model=self.azure_chat_deployment,
                messages=[{"role": "user", "content": questions_prompt}],
                temperature=0.6, max_tokens=150
            )
            if questions_response.choices and questions_response.choices[0].message:
                raw_questions = questions_response.choices[0].message.content.strip().split('\n')
                questions = [q.strip().lstrip('1234567890. ') for q in raw_questions if q.strip()]
                if not questions: questions = [f"What is discussed regarding {header or 'this topic'}?"]
                logger.debug(f"Generated Questions for '{header}': {questions}")
            else: logger.warning(f"Unexpected question response structure for header '{header}'. Using default.")
        except openai.APIError as e: logger.error(f"Azure OpenAI API error generating summary/questions for header '{header}': {e}")
        except Exception as e: logger.error(f"Unexpected error generating summary/questions for header '{header}': {e}", exc_info=True)
        return summary, questions

    async def _create_document_chunks(self, doc_name: str, content_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """(Helper) Combines paragraphs, generates summaries/questions concurrently."""
        logger.debug(f"Creating chunks for document: '{doc_name}'...")
        texts_to_process: List[tuple[Optional[str], str]] = []
        current_header: Optional[str] = "Introduction"
        buffer: List[str] = []
        for element in content_list:
            if element["type"] == "header":
                if buffer: texts_to_process.append((current_header, " ".join(buffer)))
                current_header = element["text"]
                buffer = []
            elif element["type"] == "paragraph": buffer.append(element["text"])
        if buffer: texts_to_process.append((current_header, " ".join(buffer)))

        tasks = [self._generate_summary_and_questions(text, header) for header, text in texts_to_process]
        logger.info(f"  Launching {len(tasks)} summary/question tasks concurrently for '{doc_name}'...")
        results = await async_tqdm.gather(*tasks, desc=f"AI Processing '{doc_name}'")

        chunks = []
        for i, (header, text) in enumerate(texts_to_process):
            summary, questions = results[i]
            chunks.append({"header": header, "text": text, "doc_title": doc_name, 
                           "summary": summary, "questions": questions})
        logger.debug(f"  Created {len(chunks)} chunks with AI data for '{doc_name}'.")
        return chunks

    def _index_chunks(self, doc_name: str, chunks: List[Dict[str, Any]]):
        """(Helper) Indexes document chunks into ChromaDB.
           Creates separate entries for summary and each generated question,
           with original text stored in metadata.
        """
        if not chunks: 
            logger.warning(f"Skipping indexing for '{doc_name}': No chunks.")
            return 0
        
        documents_to_add, ids_to_add, metadatas_to_add = [], [], []
        items_prepared = 0

        logger.debug(f"Preparing summary/question embeddings for {len(chunks)} original chunks ('{doc_name}')...")
        for i, chunk in enumerate(chunks):
            original_text = chunk['text']
            header = chunk.get('header', 'N/A')
            summary = chunk.get('summary', '') # Ensure summary exists
            questions = chunk.get('questions', [])
            doc_title = chunk.get('doc_title', 'Unknown')

            # Base metadata containing the original text and context
            base_metadata = {
                "original_text": original_text,
                "doc_title": doc_title,
                "header": header,
                "original_summary": summary, # Keep original summary for potential re-ranking or context
                "original_questions": " | ".join(questions),
                "chunk_char_length": len(original_text)
            }

            # 1. Add entry for the Summary
            if summary: # Only index if summary is not empty
                summary_id = f"{doc_name}_{i}_summary"
                ids_to_add.append(summary_id)
                documents_to_add.append(summary) # Embed the summary text
                summary_metadata = base_metadata.copy()
                summary_metadata["source_type"] = "summary" # Add type identifier
                metadatas_to_add.append(summary_metadata)
                items_prepared += 1
            else:
                 logger.warning(f"Chunk {i} for '{doc_name}' has empty summary, skipping summary embedding.")

            # 2. Add entry for each Question
            if questions:
                for q_idx, question_text in enumerate(questions):
                    if question_text: # Only index if question text is not empty
                        question_id = f"{doc_name}_{i}_q{q_idx}"
                        ids_to_add.append(question_id)
                        documents_to_add.append(question_text) # Embed the question text
                        question_metadata = base_metadata.copy()
                        question_metadata["source_type"] = "question" # Add type identifier
                        metadatas_to_add.append(question_metadata)
                        items_prepared += 1
                    else:
                         logger.warning(f"Chunk {i}, Question {q_idx} for '{doc_name}' is empty, skipping question embedding.")
            else:
                 logger.warning(f"Chunk {i} for '{doc_name}' has no questions, skipping question embeddings.")

        if not ids_to_add:
            logger.warning(f"No valid summary/question items to index for '{doc_name}'.")
            return 0

        try:
            # Upsert all prepared items (summaries/questions) for this original document chunk batch
            logger.info(f"Upserting {len(ids_to_add)} summary/question items for '{doc_name}' into '{self.collection.name}'...")
            self.collection.upsert(
                ids=ids_to_add, 
                documents=documents_to_add,  # These are the summaries/questions to be embedded
                metadatas=metadatas_to_add   # These contain the original_text
            )
            logger.debug(f"Successfully upserted {len(ids_to_add)} items derived from '{doc_name}'.")
            # Return total items successfully prepared and attempted to upsert for this original doc
            return len(ids_to_add) 
        except Exception as e: 
            logger.error(f"Error upserting summary/question items for {doc_name}: {e}", exc_info=True)
            return 0

    async def run_indexing(self):
        """Runs the full document ingestion and indexing pipeline."""
        logger.info("========= Starting Document Ingestion Pipeline =========")
        source_path = self.config.get('DataSource', 'documents_path')
        extracted_content = self._extract_word_docs(source_path)
        if not extracted_content: logger.error("Extraction failed. Exiting."); sys.exit(1)

        logger.info("--- Starting Chunking and Indexing --- ")
        total_chunks_processed, total_chunks_indexed, processed_doc_count = 0, 0, 0
        doc_items = list(extracted_content.items())
        for doc_name, content_list in tqdm(doc_items, desc="Processing Documents"):
            processed_doc_count += 1
            # logger.info(f"Processing Document {processed_doc_count}/{len(doc_items)}: '{doc_name}'")
            try:
                chunks = await self._create_document_chunks(doc_name, content_list)
                total_chunks_processed += len(chunks)
            except Exception as e: logger.error(f"Failed chunk creation for '{doc_name}': {e}", exc_info=True); continue
            indexed_count = self._index_chunks(doc_name, chunks)
            total_chunks_indexed += indexed_count

        logger.info("--- Chunking and Indexing Complete ---")
        logger.info(f"Docs processed: {processed_doc_count}, Chunks created: {total_chunks_processed}, Chunks indexed: {total_chunks_indexed}")
        try:
            final_count = self.collection.count()
            logger.info(f"Final item count in collection '{self.collection_name}': {final_count}")
            if total_chunks_indexed != final_count: logger.warning(f"Count mismatch: Indexed={total_chunks_indexed}, Final={final_count}. Check DB.")
        except Exception as e: logger.error(f"Failed to get final collection count: {e}")
        logger.info("========= Document Ingestion Pipeline Finished =========")

    # --- Querying Methods --- 

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """(Helper) Generates embedding using the local Sentence Transformer model."""
        if not text: logger.warning("Embedding empty text skipped."); return None
        try:
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(None, self.st_model.encode, text)
            return embedding.tolist()
        except Exception as e: logger.error(f"Error getting local embedding: {e}", exc_info=True); return None

    async def _generate_hypothetical_answer(self, question: str) -> str:
        """(Helper) Generates HyDE answer using Azure Chat client."""
        logger.info("Generating hypothetical answer for HyDE...")
        prompt = (
            "Generate a concise, hypothetical answer to the following question. "
            "Focus on capturing the key concepts the question is asking about:\n\n"
            f"Question: {question}\n\n" + "Hypothetical Answer:"
        )
        try:
            response = await self.chat_client.chat.completions.create(
                model=self.azure_chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=150
            )
            if response.choices and response.choices[0].message:
                hyde_answer = response.choices[0].message.content.strip()
                logger.debug(f"Generated HyDE answer: {hyde_answer[:100]}...")
                return hyde_answer
            logger.warning("HyDE generation returned invalid structure.")
            return f"Potential answer for {question}" # Fallback
        except Exception as e: logger.error(f"Error generating HyDE answer: {e}", exc_info=True); return f"Error generating answer for {question}"

    def _cosine_similarity(self, v1: Optional[List[float]], v2: Optional[List[float]]) -> float:
        if v1 is None or v2 is None: return -1.0 # Indicate failure
        vec1 = np.array(v1); vec2 = np.array(v2)
        norm_prod = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_prod == 0: return 0.0 # Avoid division by zero
        return np.dot(vec1, vec2) / norm_prod

    async def _rerank_results_by_summary(self, question: str, results: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """(Helper) Re-ranks results based on MINIMUM of (query-summary similarity) and (query-original_text similarity)."""
        if not self.rerank_by_summary_enabled:
             logger.debug("Summary/Text re-ranking is disabled by config."); return results
        logger.info("Re-ranking results by min(query-summary, query-text) similarity...")
        if not results or not results.get('ids') or not results['ids'][0]: return results
        
        question_embedding = await self._get_embedding(question)
        if not question_embedding: logger.warning("No question embedding, skipping re-ranking."); return results

        ids = results.get('ids', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        # Get both summaries and original texts
        summaries = [meta.get('original_summary', '') for meta in metadatas]
        original_texts = [meta.get('original_text', '') for meta in metadatas]
        
        # Get embeddings concurrently for summaries and texts
        summary_tasks = [self._get_embedding(summary) for summary in summaries]
        text_tasks = [self._get_embedding(text) for text in original_texts]
        
        logger.debug(f"Gathering {len(summary_tasks)} summary and {len(text_tasks)} text embeddings for re-ranking...")
        summary_embeddings, text_embeddings = await asyncio.gather(
            asyncio.gather(*summary_tasks),
            asyncio.gather(*text_tasks)
        )
        logger.debug("Embeddings gathered.")
        
        scored_indices = []
        for i in range(len(ids)):
            # Calculate Q vs Summary similarity
            sim_q_summary = self._cosine_similarity(question_embedding, summary_embeddings[i])
            if summary_embeddings[i] is None: logger.warning(f"No embedding for summary of {ids[i]}, Q-Summ score: {sim_q_summary:.4f}")
                
            # Calculate Q vs Original Text similarity
            sim_q_text = self._cosine_similarity(question_embedding, text_embeddings[i])
            if text_embeddings[i] is None: logger.warning(f"No embedding for original_text of {ids[i]}, Q-Text score: {sim_q_text:.4f}")

            # Use the minimum of the two similarities as the score
            # If either embedding failed, similarity is -1.0, so min works correctly
            min_similarity = min(sim_q_summary, sim_q_text)
            scored_indices.append((i, min_similarity))
            logger.debug(f"Item {ids[i]}: Q-Summ Sim={sim_q_summary:.4f}, Q-Text Sim={sim_q_text:.4f} -> Min Sim={min_similarity:.4f}")
            
        # Sort by the minimum similarity score, descending (higher min score is better)
        scored_indices.sort(key=lambda item: item[1], reverse=True)

        # Reorder results based on sorted indices
        new_results = {k: [[]] for k in ['ids', 'distances', 'metadatas', 'documents']}
        new_results['embeddings'] = None # Chroma expects this structure
        original_indices_sorted = [idx for idx, score in scored_indices]

        for key in new_results: # Iterate through keys we want to reorder
             if key == 'embeddings': continue # Skip embeddings
             original_list = results.get(key, [[]])[0]
             if original_list: # Check if list exists and is not empty
                new_results[key][0] = [original_list[i] for i in original_indices_sorted if i < len(original_list)]
        
        logger.info("Re-ranking complete.")
        return new_results

    def _format_context(self, results: Dict[str, List[Any]], include_details: bool = False) -> str:
        """(Helper) Formats results into context string.
           Uses 'original_text' from metadata as the primary content source.
        """
        context_parts = []
        if not results or not results.get('ids') or not results['ids'][0]: 
            logger.warning("No results found to format for context.")
            return "No relevant context found."

        # Ensure all necessary lists exist and get the first list from the results structure
        ids = results.get('ids', [[]])[0]
        distances = results.get('distances', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        # The 'documents' field now contains the summary/question text that matched
        matched_texts = results.get('documents', [[]])[0] 
        num_results = len(ids)

        processed_originals = set() # Keep track of original texts already included

        logger.debug(f"Formatting {num_results} retrieved items for LLM context...")
        for i in range(num_results):
            # Safely access elements using the index `i`
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else float('nan')
            matched_text = matched_texts[i] if i < len(matched_texts) else "[Matched text not available]"
            chunk_id = ids[i] if i < len(ids) else "[ID not available]"

            # --- Core Change: Use original_text from metadata --- 
            original_text = meta.get('original_text', None)
            doc_title = meta.get('doc_title', 'Unknown Document')
            header = meta.get('header', 'N/A')
            source_type = meta.get('source_type', 'unknown') # summary or question

            if not original_text:
                logger.warning(f"Result item {chunk_id} missing 'original_text' in metadata. Skipping.")
                continue

            # Avoid duplicating the exact same original text chunk if multiple views (summary, question) retrieve it
            # Create a unique identifier for the original chunk
            original_chunk_identifier = (doc_title, header, original_text)
            if original_chunk_identifier in processed_originals:
                 logger.debug(f"Skipping duplicate original text from {doc_title} - {header} (matched via {chunk_id})")
                 continue
            processed_originals.add(original_chunk_identifier)
            # --- End Core Change ---
            
            part = f"Source Document: {doc_title}\nSection: {header}\nContent:\n{original_text}"
            
            if include_details:
                 # Add details about *what* specific item matched (summary/question)
                 part += f"\n(Matched via {source_type}: \"{matched_text[:100]}...\" - ID: {chunk_id}, Distance: {dist:.4f})"
                 
            context_parts.append(part)

        if not context_parts:
             logger.warning("Could not format any context, possibly due to missing original_text or duplicates.")
             return "No relevant context could be formatted."

        full_context = "\n\n---\n\n".join(context_parts)
        logger.debug(f"Formatted Context Length: {len(full_context)}")
        return full_context

    async def _get_llm_answer(self, question: str, context: str) -> str:
        """(Helper) Gets final answer from LLM based on context."""
        logger.info(f"Sending request to LLM ({self.azure_chat_deployment}) for final answer...")
        prompt = (
            f"You are an assistant tasked with answering questions based *exclusively* on the provided text excerpts (Context). "
            f"Your answer must be derived solely from the information presented in the context below. "
            f"Do not use any external knowledge or make assumptions beyond what is written.\n\n"
            f"If the context contains the necessary information, synthesize a clear and concise answer. "
            f"If the context does not provide enough information to answer the question, state clearly: \"Based on the provided context, I cannot answer this question.\"\n\n"
            f"Context:\n---\n{context}\n---\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        logger.debug(f"Final LLM Prompt:\n{prompt[:500]}...")
        try:
            response = await self.chat_client.chat.completions.create(
                model=self.azure_chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            if response.choices and response.choices[0].message:
                answer = response.choices[0].message.content.strip()
                logger.info("Received final answer from LLM.")
                logger.debug(f"LLM Answer: {answer}")
                return answer
            logger.error("LLM response structure invalid/empty for final answer.")
            return "Error: Invalid response from LLM."
        except openai.APIError as e: logger.error(f"Azure OpenAI API error for final answer: {e}"); return f"Error: API error: {e}"
        except Exception as e: logger.error(f"Unexpected error getting final answer: {e}", exc_info=True); return "Error: Unexpected error generating answer."

    async def query(self, question: str, k: Optional[int] = None, include_details: bool = False) -> Tuple[str, str]:
        """Runs the full query pipeline: retrieve, optionally HyDE, re-rank, format, generate answer."""
        final_k = k if k is not None else self.final_k
        logger.info(f"--- Starting RAG Query for: '{question}' (Final k={final_k}) ---")
        results = None
        context_str = "Error: Initial retrieval failed."
        try:
            # Initial Retrieval
            initial_results = self.collection.query(
                query_texts=[question],
                n_results=self.initial_k,
                include=['metadatas', 'documents', 'distances']
            )
            retrieved_count = len(initial_results.get('ids', [[]])[0])
            logger.info(f"Initial retrieval got {retrieved_count} results.")

            use_hyde = False
            if retrieved_count > 0: top_distance = initial_results['distances'][0][0]; logger.debug(f"Top distance: {top_distance:.4f}"); use_hyde = top_distance > self.hyde_threshold
            elif retrieved_count == 0: use_hyde = True; logger.warning("0 results from initial retrieval, triggering HyDE.")
            if use_hyde: logger.warning(f"Triggering HyDE (Threshold: {self.hyde_threshold}).")

            if use_hyde:
                hyde_answer = await self._generate_hypothetical_answer(question)
                hyde_embedding = await self._get_embedding(hyde_answer)
                if hyde_embedding:
                    logger.info("Re-querying ChromaDB using HyDE embedding...")
                    hyde_results = self.collection.query(
                        query_embeddings=[hyde_embedding],
                        n_results=self.initial_k,
                        include=['metadatas', 'documents', 'distances']
                    )
                    hyde_retrieved_count = len(hyde_results.get('ids', [[]])[0])
                    logger.info(f"HyDE retrieval got {hyde_retrieved_count} results.")
                    results = hyde_results
                else: logger.error("Failed to get HyDE embedding. Falling back."); results = initial_results
            else: results = initial_results

            # Re-rank & Format
            if results and len(results.get('ids', [[]])[0]) > 0:
                 ranked_results = await self._rerank_results_by_summary(question, results)
                 final_results = {}
                 for key in ['ids', 'distances', 'metadatas', 'documents']:
                     if ranked_results.get(key) and ranked_results[key][0]: final_results[key] = [ranked_results[key][0][:final_k]]
                     else: final_results[key] = [[]]
                 context_str = self._format_context(final_results, include_details)
            else: logger.warning("No results after initial/HyDE retrieval."); context_str = "No relevant context found."

        except Exception as e:
            logger.error(f"Error during query/rerank/HyDE process: {e}", exc_info=True)
            context_str = "Error during context retrieval process."

        # Generate Final Answer
        final_answer = "Error: Could not generate final answer."
        if "Error" not in context_str and "No relevant context" not in context_str:
            final_answer = await self._get_llm_answer(question, context_str)
        else:
             logger.warning("Skipping final LLM answer generation due to context issues.")
             final_answer = context_str # Pass through the error/no context message
             
        return context_str, final_answer 