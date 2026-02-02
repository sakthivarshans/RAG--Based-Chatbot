import os
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore 

from pinecone import Pinecone as PineconeClient, ServerlessSpec
from llm import get_embeddings

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-rag")
FAISS_INDEX_PATH = "faiss_index"

def get_retriever():
    embeddings = get_embeddings()
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if pinecone_api_key:
        try:
            print("Attempting to connect to Pinecone...")
            pc = PineconeClient(api_key=pinecone_api_key)
            
            existing_indexes = [i.name for i in pc.list_indexes()]
            if PINECONE_INDEX_NAME in existing_indexes:
                print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
                vectorstore = PineconeVectorStore.from_existing_index(
                    index_name=PINECONE_INDEX_NAME,
                    embedding=embeddings
                )
                return vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 3, "fetch_k": 10}
                )
            else:
                print(f"Pinecone index '{PINECONE_INDEX_NAME}' not found.")
        except Exception as e:
            print(f"Pinecone connection failed: {e}")
            print("Falling back to FAISS...")

    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading local FAISS index...")
        try:
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings,
                allow_dangerous_deserialization=True 
            )
            return vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10}
            )
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
    
    raise ValueError("Could not initialize Pinecone or load local FAISS index. Run ingest.py first.")
