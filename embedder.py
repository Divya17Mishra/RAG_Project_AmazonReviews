# # import os
# # import pandas as pd
# # from dotenv import load_dotenv
# # from langchain_openai import AzureOpenAIEmbeddings
# # from langchain_core.documents import Document

# # # Load environment variables
# # load_dotenv()

# # # Initialize Azure OpenAI Embeddings
# # embedding_model = AzureOpenAIEmbeddings(
# #     deployment=os.getenv("EMBEDDINGS_MODEL"),
# #     azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT"),
# #     api_key=os.getenv("OPENAI_API_KEY"),
# #     api_version=os.getenv("OPENAI_API_VERSION")
# # )
# # # ...existing code...

# # # Check required environment variables
# # required_vars = [
# #     "EMBEDDINGS_MODEL",
# #     "OPENAI_DEPLOYMENT_ENDPOINT",
# #     "OPENAI_API_KEY",
# #     "OPENAI_API_VERSION"
# # ]
# # for var in required_vars:
# #     if not os.getenv(var):
# #         raise EnvironmentError(f"Missing required environment variable: {var}")

# # # ...existing code...

# # if __name__ == "__main__":
# #     # Load chunked data
# #     df = pd.read_csv("chunked_dataset.csv")

# #     # Convert rows to LangChain Document format
# #     documents = df.apply(lambda row: Document(
# #         page_content=row["text"],
# #         metadata={"doc_id": row["doc_id"], "info": row["metadata"]}
# #     ), axis=1).tolist()

# #     # Generate embeddings
# #     texts = [doc.page_content for doc in documents]
# #     vectors = embedding_model.embed_documents(texts)

# #     # Add embeddings to dataframe
# #     df["embedding"] = vectors

# #     # Save to file
# #     df.to_pickle("embedded_chunks.pkl")
# #     print("✅ Embeddings saved to embedded_chunks.pkl")

# import os
# import pandas as pd
# import time
# from dotenv import load_dotenv
# from langchain_openai import AzureOpenAIEmbeddings
# from langchain_core.documents import Document

# # Load environment variables
# load_dotenv()

# # Initialize Azure OpenAI Embedding client
# embedding_model = AzureOpenAIEmbeddings(
#     deployment=os.getenv("EMBEDDINGS_MODEL"),
#     azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT"),
#     api_key=os.getenv("OPENAI_API_KEY"),
#     api_version=os.getenv("OPENAI_API_VERSION")
# )

# # Batch utility
# def batch_texts(texts, batch_size):
#     for i in range(0, len(texts), batch_size):
#         yield texts[i:i + batch_size]

# if __name__ == "__main__":
#     # Load chunked dataset
#     df = pd.read_csv("chunked_dataset.csv")
    
#     # Optional: limit size for quick testing
#     # df = df.head(20)

#     # Convert to LangChain documents
#     docs = [
#         Document(page_content=row["text"], metadata={"doc_id": row["doc_id"], "info": row["metadata"]})
#         for _, row in df.iterrows()
#     ]
    
#     texts = [doc.page_content for doc in docs]
#     vectors = []
    
#     batch_size = 5
#     delay = 5  # seconds

#     print(f"Total texts: {len(texts)}")
    
#     for i, batch in enumerate(batch_texts(texts, batch_size)):
#         try:
#             batch_vectors = embedding_model.embed_documents(batch)
#             vectors.extend(batch_vectors)
#             print(f"✅ Batch {i+1} embedded successfully.")
#             time.sleep(delay)  # avoid rate limit
#         except Exception as e:
#             print(f"⚠️ Batch {i+1} failed. Waiting 60 seconds before retry...")
#             print(f"Error: {e}")
#             time.sleep(60)
    
#     # Save output
#     df["embedding"] = vectors
#     df.to_pickle("embedded_chunks.pkl")
#     print("✅ All embeddings saved to embedded_chunks.pkl")
import os
import pandas as pd
import time
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI Embedding client
embedding_model = AzureOpenAIEmbeddings(
    deployment=os.getenv("EMBEDDINGS_MODEL"),
    azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION")
)

# Batch utility
def batch_texts(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]

if __name__ == "__main__":
    # Load chunked dataset
    df = pd.read_csv("chunked_dataset.csv")
    
    # Optional: limit size for quick testing
    # df = df.head(20)

    # Convert to LangChain documents
    docs = [
        Document(page_content=row["text"], metadata={"doc_id": row["doc_id"], "info": row["metadata"]})
        for _, row in df.iterrows()
    ]
    
    texts = [doc.page_content for doc in docs]
    vectors = []
    
    batch_size = 5
    delay = 5  # seconds
    max_batches = 100  # Limit to 100 batches only

    print(f"Total texts: {len(texts)}")
    
    for i, batch in enumerate(batch_texts(texts, batch_size)):
        if i >= max_batches:
            print(f"🛑 Reached the maximum limit of {max_batches} batches. Stopping...")
            break
        try:
            batch_vectors = embedding_model.embed_documents(batch)
            vectors.extend(batch_vectors)
            print(f"✅ Batch {i+1} embedded successfully.")
            time.sleep(delay)  # avoid rate limit
        except Exception as e:
            print(f"⚠️ Batch {i+1} failed. Waiting 60 seconds before retry...")
            print(f"Error: {e}")
            time.sleep(60)
    
    # Save output
    df = df.iloc[:len(vectors)]  # ensure df length matches vector count
    df["embedding"] = vectors
    df.to_pickle("embedded_chunks.pkl")
    print("✅ All embeddings saved to embedded_chunks.pkl")



