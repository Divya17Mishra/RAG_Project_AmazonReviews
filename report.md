# RAG Assignment: Vector Store and Embedding Pipeline

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Azure OpenAI embeddings and FAISS vector store for efficient document retrieval. The workflow covers chunking documents, generating embeddings, and building a vector store for semantic search.

---

## Project Structure

- **embedder.py**: Loads chunked text data, generates embeddings using Azure OpenAI, and saves them.
- **vector_store.py**: Loads embeddings, reconstructs LangChain documents, and builds a FAISS vector store for retrieval.
- **chunked_dataset.csv**: Input CSV containing document chunks.
- **embedded_chunks.pkl**: Output pickle file with embeddings.
- **faiss_index/**: Directory containing the FAISS index and metadata for retrieval.

---

## Pipeline Steps

### 1. Chunking Documents

Documents are preprocessed and split into manageable text chunks, each with associated metadata (e.g., `doc_id`, `metadata`). These chunks are stored in `chunked_dataset.csv`.

### 2. Embedding Generation (`embedder.py`)

- Loads environment variables for Azure OpenAI credentials.
- Reads the chunked dataset.
- Converts each row into a LangChain `Document` object.
- Batches text chunks and generates embeddings using Azure OpenAI.
- Handles batching and rate limits.
- Saves the resulting embeddings alongside the original data in `embedded_chunks.pkl`.

### 3. Vector Store Creation (`vector_store.py`)

- Loads the embeddings from `embedded_chunks.pkl`.
- Reconstructs LangChain `Document` objects for each chunk.
- Initializes a FAISS vector store using LangChain's FAISS wrapper.
- Saves the FAISS index and metadata to the `faiss_index/` directory for later retrieval.

---

## Usage

1. **Set up environment variables** for Azure OpenAI (see `.env.example`).
2. **Run the embedder** to generate embeddings:
   ```bash
   python embedder.py
   ```
3. **Build the FAISS vector store**:
   ```bash
   python vector_store.py
   ```

---

## Key Technologies

- **Azure OpenAI**: For generating high-quality text embeddings.
- **LangChain**: For document abstraction and vector store integration.
- **FAISS**: For efficient similarity search over large embedding sets.
- **Pandas**: For data manipulation and storage.

---

## Notes

- Ensure all required environment variables are set before running the scripts.
- The pipeline is designed to handle batching and rate limits for large datasets.
- The FAISS index and metadata can be used for fast semantic search in downstream RAG applications.

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)

---

*This report summarizes the implementation and workflow as described in the provided assignment PDF.*