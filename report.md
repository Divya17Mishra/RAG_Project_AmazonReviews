# Amazon Review RAG-Based QA System

## 📚 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to answer user questions about Amazon products using customer reviews. The system uses Azure OpenAI for embeddings and language modeling, FAISS for vector search, and Streamlit for an interactive web interface.

---

## 🗂️ Project Structure

- **data_ingest.py**
  - Loads and cleans raw Amazon review data.
  - Removes HTML tags, handles missing values, and creates a metadata field.
  - Outputs: `cleaned_dataset.csv`
- **chunker.py**
  - Splits each review into chunks (default: 500 words).
  - Associates each chunk with its document ID and metadata.
  - Outputs: `chunked_dataset.csv`
- **embedder.py**
  - Loads chunked data and generates embeddings using Azure OpenAI.
  - Handles batching and rate limits.
  - Outputs: `embedded_chunks.pkl`
- **vector_store.py**
  - Loads embeddings and metadata.
  - Builds a FAISS vector store using LangChain.
  - Outputs: `faiss_index/` (contains FAISS index and metadata)
- **app.py**
  - Streamlit web app for interactive Q&A.
  - Loads the FAISS vector store and Azure OpenAI model.
  - Uses a custom prompt to ensure answers are based only on review context.
  - Displays answers and the top retrieved review chunks.

---

## 🔄 Data Pipeline Steps

### 1. Data Ingestion & Cleaning (`data_ingest.py`)
- Reads raw Amazon review CSV.
- Drops rows with missing review text.
- Cleans text by removing HTML tags.
- Combines summary and product ID into a metadata field.
- Keeps only necessary columns: `id`, `text`, `metadata`.
- Saves cleaned data to `cleaned_dataset.csv`.

### 2. Text Chunking (`chunker.py`)
- Splits each review into chunks of up to 500 words.
- Each chunk is labeled with `doc_id`, `chunk_id`, `text`, and `metadata`.
- Saves chunked data to `chunked_dataset.csv`.

### 3. Embedding Generation (`embedder.py`)
- Loads environment variables for Azure OpenAI.
- Reads the chunked dataset.
- Converts each chunk into a LangChain `Document`.
- Batches text and generates embeddings using Azure OpenAI.
- Handles batching and rate limits for large datasets.
- Saves embeddings and chunk data to `embedded_chunks.pkl`.

### 4. Vector Store Creation (`vector_store.py`)
- Loads embeddings and chunk metadata.
- Reconstructs LangChain `Document` objects.
- Builds a FAISS vector store using LangChain's FAISS wrapper.
- Saves the FAISS index and metadata to the `faiss_index/` directory.

### 5. Interactive Q&A Application (`app.py`)
- Loads the FAISS vector store and Azure OpenAI model.
- Uses a custom prompt to ensure:
  - Answers are based only on the provided review context.
  - No made-up information is given.
  - If context is insufficient, responds with "I don't know based on the reviews."
  - Summarizes sentiments and uses bullet points for clarity.
- Streamlit UI allows users to input questions and view answers with supporting review chunks.

---

## 🛠️ Technologies Used

- **Azure OpenAI**: For generating embeddings and language model completions.
- **LangChain**: For document abstraction, prompt management, and vector store integration.
- **FAISS**: For fast similarity search over large sets of embeddings.
- **Pandas**: For data manipulation and storage.
- **Streamlit**: For building the interactive web application.

---

## ⚙️ How to Run

1. **Set up environment variables** for Azure OpenAI (API key, deployment name, endpoint, etc.).
2. **Prepare the data**:
    - Run `python data_ingest.py` to clean the raw data.
    - Run `python chunker.py` to split reviews into chunks.
3. **Generate embeddings**:
    - Run `python embedder.py` to create and save embeddings.
4. **Build the vector store**:
    - Run `python vector_store.py` to create and save the FAISS index.
5. **Launch the app**:
    - Run `streamlit run app.py` to start the web interface.

---

## 📝 Key Features

- **Contextual Q&A**: Answers are strictly based on customer review context.
- **No Hallucination**: The system avoids making up information not present in the reviews.
- **Sentiment Summarization**: Answers summarize common sentiments and cite review phrases.
- **Bullet Points**: Answers are formatted with bullet points for clarity and interactivity.
- **Transparency**: Shows which review chunks were used to generate the answer.
- **Scalable**: Handles large datasets with batching and efficient vector search.

---
