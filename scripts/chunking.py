from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_context_aware_chunks(raw_text: str, metadata: dict) -> list:
    """
    Chunks legal text intelligently and injects global context into every chunk.
    """
    
    # 1. Initialize the Recursive Splitter
    # This acts like a smart scalpel. It tries to split by paragraphs (\n\n) first. 
    # If a paragraph is too long, it tries single newlines (\n), then periods (. ), etc.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # About 150-200 words, ideal for BGE-Large
        chunk_overlap=120,    # ~15% overlap ensures clauses spanning chunks aren't lost
        length_function=len,
        separators=[
            "\n\n",           # Split by major paragraphs first
            "\n",             # Then split by line breaks
            ". ",             # Then split by sentences
            ", ",             # Then split by clauses
            " "               # Last resort: split by words
        ]
    )

    # 2. Split the raw text into base chunks
    raw_chunks = text_splitter.split_text(raw_text)
    
    context_aware_chunks = []
    
    # 3. The Amnesia Fix: Metadata Injection
    # We format a 'context header' using the extracted metadata
    context_header = (
        f"[Context - Court: {metadata.get('court_name', 'Unknown')}, "
        f"Case No: {metadata.get('case_number', 'Unknown')}, "
        f"Date: {metadata.get('judgment_date', 'Unknown')}]\n"
        f"---\n"
    )

    for i, chunk_text in enumerate(raw_chunks):
        # We physically prepend the context to the text that will be embedded
        enriched_text = context_header + chunk_text
        
        # We package it into a dictionary to easily insert into ChromaDB later
        context_aware_chunks.append({
            "id": f"{metadata.get('case_number')}_chunk_{i}",
            "text_to_embed": enriched_text,
            "metadata_for_db": metadata # We also keep the raw metadata for exact filtering
        })
        
    return context_aware_chunks

# --- Execution Example ---
if __name__ == "__main__":
    # Simulating the outputs from Phase 1, Step 1
    sample_metadata = {
        "court_name": "High Court of Delhi",
        "case_number": "W.P.(C) 1234/2023",
        "judgment_date": "2023-08-15"
    }
    
    sample_legal_text = (
        "The writ petition under Article 226 of the Constitution of India is filed "
        "challenging the impugned order dated 12.01.2023. \n\n"
        "The counsel for the petitioner argues that the principles of natural justice "
        "were completely violated during the tribunal proceedings. The respondent failed "
        "to provide the necessary documents." * 10 # Multiplying to simulate length
    )
    
    # Run the chunker
    processed_chunks = create_context_aware_chunks(sample_legal_text, sample_metadata)
    
    print(f"Total chunks created: {len(processed_chunks)}\n")
    print("Example of Chunk #2 (Notice how the context is preserved!):\n")
    print(processed_chunks[1]['text_to_embed'])