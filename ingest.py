import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from llm import get_embeddings

load_dotenv()

DATA_PATH = "data/Ebook-Agentic-AI.pdf"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-rag")
FAISS_INDEX_PATH = "faiss_index"

def ingest_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        print("Please place 'Ebook-Agentic-AI.pdf' in the 'data/' folder.")
        return

    print("Loading PDF...")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    embeddings = get_embeddings()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    use_pinecone = False

    if pinecone_api_key:
        try:
            print("Initializing Pinecone...")
            pc = PineconeClient(api_key=pinecone_api_key)
            
            existing_indexes = [i.name for i in pc.list_indexes()]
            if PINECONE_INDEX_NAME not in existing_indexes:
                print(f"Creating index {PINECONE_INDEX_NAME}...")
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=768, 
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=os.getenv("PINECONE_ENV", "us-east-1")
                    )
                )
                time.sleep(2) 
            
            print(f"Upserting to Pinecone index '{PINECONE_INDEX_NAME}'...")
            PineconeVectorStore.from_documents(
                chunks, 
                embeddings, 
                index_name=PINECONE_INDEX_NAME
            )
            print("Pinecone ingestion complete.")
            use_pinecone = True
            
        except Exception as e:
            print(f"Pinecone ingestion error: {e}")
            print("Proceeding to generate local FAISS index as fallback/primary.")

    print("Creating local FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}'.")
    
    print("Ingestion pipeline finished.")

if __name__ == "__main__":
    ingest_data()
