# RAG Pipeline for Document Q&A

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on a collection of `.docx` documents. It uses local sentence transformers for embedding, ChromaDB as a vector store, and Azure OpenAI for Large Language Model (LLM) tasks (summarization, question generation, final answer synthesis).

## Features

*   Ingestion pipeline (`main.py`) to process `.docx` files, chunk text, generate summaries/questions via LLM, and index embeddings into ChromaDB.
*   Indexing strategy that embeds summaries and generated questions separately, linking them back to the original text chunk via metadata.
*   Query pipeline (`query_db.py`) that takes a user question and performs the following:
    *   Embeds the query using a local sentence transformer.
    *   Retrieves relevant text chunks from ChromaDB based on semantic similarity to summaries/questions.
    *   (Optional) Employs HyDE (Hypothetical Document Embeddings) for query expansion if initial results are weak.
    *   (Optional) Re-ranks results based on the minimum similarity between the query and both the chunk's summary *and* its original text.
    *   Constructs a context prompt using the retrieved original text.
    *   Calls an Azure OpenAI chat model to generate a final answer based *only* on the provided context.
*   Utility script (`re_embed.py`) to re-index data with a different embedding model without re-running expensive LLM calls.
*   Configuration managed via `config.ini`.
*   Object-oriented design centered around `RAGPipeline` and `ConfigLoader` classes.

## Setup

**1. Clone the Repository:**

```bash
git clone <your-repository-url>
cd <repository-directory>
```

**2. Create a Virtual Environment (Recommended):**

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**3. Install Dependencies:**

```bash
pip install -r requirements.txt
```
This will install `openai`, `chromadb`, `sentence-transformers`, `python-docx`, `tqdm`, and `numpy`.

**4. Configure Credentials and Parameters (`config.ini`):**

*   Open the `config.ini` file.
*   **[AzureOpenAI]:** Update `endpoint`, `api_key`, `api_version`, and `chat_deployment` with your Azure OpenAI service details.
*   **[Embedding]:** Set `local_model_name` to the desired Sentence Transformer model from Hugging Face (e.g., `all-mpnet-base-v2`, `all-MiniLM-L6-v2`). The first time you run the scripts, this model will be downloaded if not cached.
*   **[ChromaDB]:** Configure `collection_name` and `persist_directory`. Ensure the directory exists or has appropriate permissions.
*   **[RAGParams]:** Adjust parameters like `initial_retrieval_k`, `final_retrieval_k`, `hyde_distance_threshold`, `rerank_by_summary` (True/False) to tune the RAG process.
*   **[DataSource]:** Set `documents_path` to the *relative or absolute path* containing your source `.docx` files.

## Running the System

**1. Indexing Documents:**

*   Place your source `.docx` files in the directory specified by `documents_path` in `config.ini`.
*   Run the indexing script:

    ```bash
    python main.py
    ```
*   This will process the documents, call the Azure LLM for summaries/questions, embed them using the configured local model, and store everything in the ChromaDB specified in `config.ini`. The time taken will be logged.

**2. Querying the Documents:**

*   Once indexing is complete, run the query script:

    ```bash
    python query_db.py
    ```
*   The script will connect to the ChromaDB collection specified in `config.ini`.
*   It will enter an interactive loop. Type your question and press Enter.
*   The pipeline will retrieve relevant context, generate an answer using Azure OpenAI, and print both.
*   **Options:**
    *   Append `k=<number>` to your query to override the `final_retrieval_k` for that specific query (e.g., `What is AI? k=5`).
    *   Append `details` to your query to include debug information (matched source type, chunk ID, distance) in the context output (e.g., `What is AI? details`).
*   Type `quit` to exit.

**3. (Optional) Re-embedding with a New Model:**

*   If you want to switch embedding models without re-generating summaries/questions:
    *   Update `config.ini` with the **new** `local_model_name`, **new** `collection_name`, and **new** `persist_directory`.
    *   Run the `re_embed.py` script, pointing it to the *old* collection:

        ```bash
        python re_embed.py --old-collection <previous_collection_name> --old-persist-dir <previous_persist_directory>
        ```
    *   This reads data from the old collection and upserts it into the new one, generating embeddings with the new model.

## Code Flow and Input Handling

**Core Components:**

*   `config.ini`: Stores all configuration (API keys, paths, model names, parameters).
*   `rag_config.py`: Contains the `ConfigLoader` class for reading `config.ini`.
*   `rag_pipeline.py`: Contains the `RAGPipeline` class, encapsulating the main logic for indexing and querying (initialization, document processing, embedding, vector search, LLM calls, re-ranking, context formatting).
*   `main.py`: Entry point for the indexing process. Initializes `ConfigLoader` and `RAGPipeline`, then calls `pipeline.run_indexing()`.
*   `query_db.py`: Entry point for querying. Initializes `ConfigLoader` and `RAGPipeline`, then enters an interactive loop calling `pipeline.query()`.
*   `re_embed.py`: Utility script to migrate data between collections using different embedding models.

**Input Handling:**

*   **Indexing:** Takes the path to a directory of `.docx` files as input (via `config.ini`).
*   **Querying:** Takes user questions interactively via the command line.
*   **Configuration:** Reads API keys, model names, paths, and parameters from `config.ini`.

**General Flow:**

1.  **Configuration:** Settings are loaded from `config.ini` by `ConfigLoader`.
2.  **Initialization:** `RAGPipeline` uses the configuration to set up clients (Azure OpenAI, ChromaDB) and resources (Sentence Transformer model).
3.  **Indexing (`main.py` -> `RAGPipeline.run_indexing`):**
    *   Documents are read and parsed.
    *   Summaries/Questions are generated via Azure OpenAI.
    *   Summaries/Questions are embedded using the local Sentence Transformer.
    *   Embeddings and metadata (including original text) are stored in ChromaDB.
4.  **Querying (`query_db.py` -> `RAGPipeline.query`):**
    *   User query is embedded.
    *   ChromaDB is searched for relevant summary/question embeddings.
    *   Results are potentially refined using HyDE and re-ranking (comparing query vs. original summary/text embeddings).
    *   Context is built using the `original_text` from the metadata of the final results.
    *   Azure OpenAI generates the final answer based on the context.

This separation allows for a configurable and extensible RAG system. 