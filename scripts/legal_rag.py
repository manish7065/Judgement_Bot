import os
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
import uuid

from dotenv import load_dotenv
load_dotenv()

# ===================== CONFIG =====================
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "orissa_judgments_basic-1"
# PERSIST_DIR = "./chroma_db"
host: str = "manishs-mac-mini.tailc96719.ts.net"
port: int = 8030


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"device": "cpu"}   # change to "cuda" if available
)

# Create Chroma HTTP Client
chroma_client = chromadb.HttpClient(
    host=host,
    port=port,
    # ssl=True,           # uncomment if using HTTPS
    # headers={"X-Api-Key": "your-api-key"}  # if you have authentication
)

# LangChain VectorStore using remote client
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    client=chroma_client,                    # ← This is the key
    # persist_directory is NOT used with HttpClient
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)  # or gpt-4o
# ================================================

def load_and_chunk_judgment(pdf_path: str):
    """Load PDF + Semantic Chunking"""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    full_text = "\n\n".join([doc.page_content for doc in docs])
    
    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=92,
    )
    
    chunks = semantic_chunker.split_text(full_text)
    
    return chunks, pdf_path

def ingest_judgments(pdf_folder: str):
    """Ingest all PDFs in a folder"""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            print(f"Processing: {filename}")
            
            chunks, source = load_and_chunk_judgment(path)
            
            metadatas = [{
                "source": filename,
                "case_name": filename.replace(".pdf", "").replace("display_judgement", ""),
                "chunk_id": i,
                "type": "judgment"
            } for i in range(len(chunks))]
            
            ids = [str(uuid.uuid4()) for _ in chunks]
            
            vectorstore.add_texts(
                texts=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"  → Added {len(chunks)} chunks")

# ===================== RETRIEVAL & GENERATION =====================

def retrieve_context(query: str, k: int = 6):
    docs = vectorstore.similarity_search(query, k=k)
    return docs

def generate_answer(query: str, context_docs) -> Dict:
    context_text = ""
    for i, doc in enumerate(context_docs):
        meta = doc.metadata
        context_text += f"--- Document {i+1} | {meta.get('source', 'Unknown')} ---\n"
        context_text += f"Content:\n{doc.page_content}\n\n"

        # source = doc.metadata.get('source', 'Unknown')
        # context_text += f"--- Document {i+1} | Source: {source} ---\n"
        # context_text += doc.page_content + "\n\n"

    system_prompt = """You are an expert Indian Legal AI Assistant.
Answer the question using the provided context documents.
If the relevant information is present, summarize it clearly.
If you cannot find enough information, say so honestly."""
    
#     system_prompt = """You are a precise Indian Legal AI Assistant.
# Answer ONLY using the provided context.
# If the information is not present or insufficient, reply: 
# "I cannot find sufficient information in the provided documents to answer this question." 
# Do not speculate or make up facts."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]
    
    response = llm.invoke(messages)
    
    return {
        "answer": response.content,
        "sources": [doc.metadata for doc in context_docs]
    }

# ===================== USAGE =====================

if __name__ == "__main__":
    # Ingest once
    # ingest_judgments("/Users/m2sm/Desktop/projects/Agentic-AI/Judgement_Bot/Data/odisha_judgement_files")
    
    query = "Summarize the Bandhna Toppo case"
    context = retrieve_context(query, k=5)
    result = generate_answer(query, context)
    
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(s)