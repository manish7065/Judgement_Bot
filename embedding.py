import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

def store_chunks_in_chroma(
    processed_chunks: List[Dict], 
    db_path: str = "./legal_chroma_db", 
    collection_name: str = "judgments",
    batch_size: int = 250
):
    """
    Takes processed context-aware chunks, embeds them, and stores them in ChromaDB in safe batches.
    """
    
    print(f"Initializing embedding model...")
    # You can easily swap this to "BAAI/bge-m3" if you want to upgrade later!
    encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    print(f"Connecting to ChromaDB at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    
    # get_or_create ensures we don't overwrite the database if we run this script multiple times
    collection = client.get_or_create_collection(name=collection_name)
    
    total_chunks = len(processed_chunks)
    print(f"Starting ingestion of {total_chunks} chunks...")

    # Process in batches to prevent Out-Of-Memory (OOM) errors
    for i in range(0, total_chunks, batch_size):
        batch = processed_chunks[i : i + batch_size]
        
        # 1. Extract the individual lists required by ChromaDB
        batch_ids = [chunk["id"] for chunk in batch]
        batch_texts = [chunk["text_to_embed"] for chunk in batch]
        
        # ChromaDB requires metadata values to be strings, ints, or floats. 
        # Lists (like acts_and_sections) need to be converted to comma-separated strings.
        batch_metadatas = []
        for chunk in batch:
            clean_metadata = {}
            for key, value in chunk["metadata_for_db"].items():
                if isinstance(value, list):
                    clean_metadata[key] = ", ".join(value)
                else:
                    clean_metadata[key] = value
            batch_metadatas.append(clean_metadata)

        # 2. Generate Embeddings for this batch
        print(f"Embedding batch {i} to {i + len(batch)}...")
        batch_embeddings = encoder.encode(batch_texts).tolist()

        # 3. Insert into ChromaDB
        print(f"Saving batch to database...")
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas
        )
        
    print(f"✅ Successfully saved {total_chunks} chunks to ChromaDB!")
    
    # Return the collection so we can run a quick test
    return collection

# --- Execution Example ---
if __name__ == "__main__":
    # Simulating the output from your chunking function
    sample_processed_chunks = [
        {
            "id": "WP(C)_1234_chunk_0",
            "text_to_embed": "[Context - Court: Delhi HC, Case No: W.P.(C) 1234/2023]\n---\nThe writ petition under Article 226...",
            "metadata_for_db": {"court_name": "Delhi HC", "case_number": "W.P.(C) 1234/2023", "acts_and_sections": ["Article 226"]}
        },
        {
            "id": "WP(C)_1234_chunk_1",
            "text_to_embed": "[Context - Court: Delhi HC, Case No: W.P.(C) 1234/2023]\n---\nThe counsel argued about natural justice...",
            "metadata_for_db": {"court_name": "Delhi HC", "case_number": "W.P.(C) 1234/2023", "acts_and_sections": ["Article 226"]}
        }
    ]
    
    # Run the storage function
    collection = store_chunks_in_chroma(sample_processed_chunks)
    
    # Let's do a quick sanity check to prove it worked!
    print("\n--- Sanity Check ---")
    count = collection.count()
    print(f"Total documents now in ChromaDB collection: {count}")



    