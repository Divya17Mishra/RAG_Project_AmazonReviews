# import pandas as pd
# import faiss
# import numpy as np

# def build_faiss_index(embeddings):
#     dim = len(embeddings[0])
#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(embeddings).astype("float32"))
#     return index

# if __name__ == "__main__":
#     df = pd.read_pickle("embedded_chunks.pkl")
#     vectors = np.vstack(df["embedding"])
#     index = build_faiss_index(vectors)
#     faiss.write_index(index, "faiss_index.index")
#     df.to_pickle("vector_metadata.pkl")
# import os
# import pandas as pd
# import faiss
# import numpy as np

# def build_faiss_index(embeddings):
#     dim = len(embeddings[0])
#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(embeddings).astype("float32"))
#     return index

# if __name__ == "__main__":
#     # Load embeddings
#     df = pd.read_pickle("embedded_chunks.pkl")
#     vectors = np.vstack(df["embedding"])

#     # Build FAISS index
#     index = build_faiss_index(vectors)

#     # ✅ Save the index in the correct folder structure for LangChain
#     os.makedirs("faiss_index", exist_ok=True)  # ensure folder exists
#     faiss.write_index(index, "faiss_index/index.faiss")  # save inside folder with correct name

#     # Save metadata separately if needed
#     df.to_pickle("vector_metadata.pkl")

import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS as LangFAISS

# Load env vars for Azure keys
load_dotenv()

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

if __name__ == "__main__":
    # Load embeddings
    df = pd.read_pickle("embedded_chunks.pkl")
    vectors = np.vstack(df["embedding"])

    # Recreate LangChain documents
    docs = [
        Document(page_content=row["text"], metadata={"doc_id": row["doc_id"], "info": row["metadata"]})
        for _, row in df.iterrows()
    ]

    # Create LangChain Embeddings object
    embedding_model = AzureOpenAIEmbeddings(
        deployment=os.getenv("EMBEDDINGS_MODEL"),
        azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION")
    )

    # Create LangChain FAISS vector store object
    langchain_faiss = LangFAISS.from_documents(docs, embedding_model)

    # ✅ Save full index (both index.faiss and index.pkl)
    langchain_faiss.save_local("faiss_index")

    print("✅ FAISS vector store saved to 'faiss_index/' folder.")




