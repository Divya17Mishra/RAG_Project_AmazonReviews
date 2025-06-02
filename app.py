# import os
# import streamlit as st
# from dotenv import load_dotenv
# import pandas as pd
# import faiss
# import numpy as np
# from langchain.vectorstores import FAISS as LangFAISS
# from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS as LangFAISS


# # Load secrets
# load_dotenv()
# OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
# OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT")

# # Streamlit UI
# st.set_page_config(page_title="Azure RAG", layout="wide")
# st.title("🔍 RAG-Based Q&A using Azure OpenAI")

# # Load embeddings & metadata
# @st.cache_resource
# def load_faiss():
#     df = pd.read_pickle("vector_metadata.pkl")
#     index = faiss.read_index("faiss_index.index")
#     embeddings = AzureOpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT)
#     docs = [
#         Document(page_content=row["text"], metadata={"doc_id": row["doc_id"], "info": row["metadata"]})
#         for _, row in df.iterrows()
#     ]
#     db = LangFAISS(embedding_function=embeddings, index=index, documents=docs)
#     return db

# # Initialize RAG Chain
# def build_rag_chain(vector_store):
#     retriever = vector_store.as_retriever(search_type="similarity", k=3)
#     llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME, temperature=0.3)

#     prompt_template = PromptTemplate.from_template("""
#     Answer the question based only on the context below. Be concise and factual.

#     Context:
#     {context}

#     Question: {question}
#     """)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt_template}
#     )
#     return qa_chain

# # Main RAG logic
# vector_store = load_faiss()
# qa = build_rag_chain(vector_store)

# question = st.text_input("Ask a question based on the review data:")
# if question:
#     with st.spinner("Thinking..."):
#         result = qa(question)
#         st.subheader("💬 Answer")
#         st.write(result["result"])

#         st.subheader("📄 Retrieved Chunks")
#         for doc in result["source_documents"]:
#             st.info(doc.page_content[:300])

# import os
# import streamlit as st
# from dotenv import load_dotenv
# import pandas as pd
# import faiss
# import numpy as np
# from langchain_community.vectorstores import FAISS as LangFAISS
# from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_core.documents import Document

# # Load environment variables
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
# OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDINGS_MODEL")
# OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
# AZURE_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")

# # Streamlit UI setup
# st.set_page_config(page_title="Azure RAG", layout="wide")
# st.title("🔍 RAG-Based Q&A using Azure OpenAI")

# # Load FAISS vector store with embeddings
# @st.cache_resource
# def load_faiss():
#     df = pd.read_pickle("vector_metadata.pkl")
#     index = faiss.read_index("faiss_index.index")

#     embeddings = AzureOpenAIEmbeddings(
#         api_key=OPENAI_API_KEY,
#         azure_endpoint=AZURE_ENDPOINT,
#         api_version=OPENAI_API_VERSION,
#         deployment=OPENAI_EMBEDDING_DEPLOYMENT
#     )

#     # Prepare documents
#     docs = [
#         Document(page_content=row["text"], metadata={"doc_id": row["doc_id"], "info": row["metadata"]})
#         for _, row in df.iterrows()
#     ]

#     # Use from_documents to create a FAISS index if building from scratch:
#     # db = LangFAISS.from_documents(docs, embedding=embeddings)

#     # Since you're loading an existing index, use this:
#     db = LangFAISS(embedding_function=embeddings, index=index)
#     db.docstore.add_documents(docs)  # Attach documents to the index

#     return db

# # Build RAG pipeline
# def build_rag_chain(vector_store):
#     retriever = vector_store.as_retriever(search_type="similarity", k=3)
#     llm = AzureChatOpenAI(
#         api_key=OPENAI_API_KEY,
#         azure_endpoint=AZURE_ENDPOINT,
#         api_version=OPENAI_API_VERSION,
#         deployment_name=OPENAI_DEPLOYMENT_NAME,
#         temperature=0.3
#     )

#     prompt_template = PromptTemplate.from_template("""
#     Answer the question based only on the context below. Be concise and factual.

#     Context:
#     {context}

#     Question: {question}
#     """)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt_template}
#     )
#     return qa_chain

# # Main logic
# vector_store = load_faiss()
# qa = build_rag_chain(vector_store)

# question = st.text_input("Ask a question based on the review data:")
# if question:
#     with st.spinner("Thinking..."):
#         result = qa(question)
#         st.subheader("💬 Answer")
#         st.write(result["result"])

#         st.subheader("📄 Retrieved Chunks")
#         for doc in result["source_documents"]:
#             st.info(doc.page_content[:300])
import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDINGS_MODEL")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")

# Streamlit UI setup
st.set_page_config(page_title="Azure RAG", layout="wide")
st.title("🔍 RAG-Based Q&A using Azure OpenAI")

# Load FAISS vector store with embeddings
@st.cache_resource
def load_faiss():
    embeddings = AzureOpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=OPENAI_API_VERSION,
        deployment=OPENAI_EMBEDDING_DEPLOYMENT
    )

    # Load FAISS vector store from saved folder
    db = LangFAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db

# Build RAG pipeline
def build_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", k=3)
    # llm = AzureChatOpenAI(
    #     api_key=OPENAI_API_KEY,
    #     azure_endpoint=AZURE_ENDPOINT,
    #     api_version=OPENAI_API_VERSION,
    #     deployment_name=OPENAI_DEPLOYMENT_NAME,
    #     temperature=0.3
    # )
    llm = AzureChatOpenAI(
    deployment_name=os.getenv("OPENAI_MODEL_NAME"),   # 👈 deployment name in Azure
    azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION")
)

    # prompt_template = PromptTemplate.from_template("""
    # Answer the question based only on the context below. Be concise and factual.

    # Context:
    # {context}

    # Question: {question}
    # """)
    prompt_template = PromptTemplate.from_template("""
    You are a helpful assistant answering user queries about a product using its customer reviews.

    Instructions:
    - Use ONLY the context provided below.
    - DO NOT make up answers.
    - If the context lacks enough information, respond with "I don't know based on the reviews."
    - Prefer summarizing sentiments (e.g., "Most users found...", "Several reviewers mention...")
    - Paraphrase or cite short review phrases when appropriate.
    - use points in detail and add bullets for interactive look
    Context (Customer Reviews):
    {context}

    Question:
    {question}

    Answer:
    """)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

# Main logic
vector_store = load_faiss()
qa = build_rag_chain(vector_store)

question = st.text_input("Ask a question based on the review data:")
if question:
    with st.spinner("Thinking..."):
        result = qa(question)
        st.subheader("💬 Answer")
        st.write(result["result"])

        st.subheader("📄 Retrieved Chunks")
        for doc in result["source_documents"]:
            st.info(doc.page_content[:300])

