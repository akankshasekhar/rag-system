import os
import re
from pypdf import PdfReader
import chromadb

DATA_DIR = "data"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "my_documents"

# ~1800 chars ≈ 450 tokens; ~350 chars ≈ 85 tokens overlap
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 350


def extract_pdfs(data_dir: str) -> list[dict]:
    """Extract text page-by-page from all PDFs in data_dir."""
    pages = []
    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(data_dir, filename)
        reader = PdfReader(filepath)
        print(f"Loading {filename} ({len(reader.pages)} pages)...")
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Clean excessive whitespace/newlines
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append({
                    "source": filename,
                    "page": i + 1,
                    "text": text,
                })
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into sentence-aware chunks with overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Build overlap from the end of the current chunk
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_len += len(s) + 1

            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += len(sentence) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def main():
    pages = extract_pdfs(DATA_DIR)
    if not pages:
        raise ValueError(f"No PDF files found in '{DATA_DIR}/'. Add a PDF and re-run.")

    documents, metadatas, ids = [], [], []
    chunk_counter = 0

    for page_data in pages:
        chunks = chunk_text(page_data["text"])
        for chunk in chunks:
            chunk_counter += 1
            documents.append(chunk)
            metadatas.append({
                "source": page_data["source"],
                "page": page_data["page"],
            })
            ids.append(f"chunk_{chunk_counter}")

    print(f"\nCreated {len(documents)} chunks from {len(pages)} pages.")
    print("\n--- SAMPLE CHUNK INSPECTION (Chunk #1) ---")
    print(documents[0][:500], "...\n")

    # Store in local persistent ChromaDB
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )

    print(f"Successfully stored {collection.count()} chunks in ChromaDB at '{DB_DIR}'.")


if __name__ == "__main__":
    main()