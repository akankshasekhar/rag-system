import os
import sys
from dotenv import load_dotenv
import chromadb
from google import genai

load_dotenv()

DB_DIR = "./chroma_db"
COLLECTION_NAME = "my_documents"
MODEL_NAME = "gemini-3.8-flash"

# Initialize clients
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
llm_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def retrieve(question: str, top_k: int = 4) -> list[dict]:
    """Fetch top-k most relevant chunks from ChromaDB."""
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "distance": round(dist, 4),
        })
    return retrieved


def ask_without_rag(question: str) -> str:
    """Plain LLM call with no retrieved context."""
    response = llm_client.models.generate_content(
        model=MODEL_NAME,
        contents=question,
    )
    return response.text


def ask_with_rag(question: str, chunks: list[dict]) -> str:
    """LLM call grounded strictly in retrieved document chunks."""
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context below.
If the answer is not in the context, state that the provided documents do not contain this information.
Cite the source and page number in your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    response = llm_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text


def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your question: ")

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    # 1. Verify Retrieval
    chunks = retrieve(question, top_k=4)
    print("\n[1] RETRIEVED CHUNKS:")
    for idx, c in enumerate(chunks, 1):
        print(f"  {idx}. {c['source']} (Page {c['page']}) | Distance: {c['distance']}")
        print(f"     Preview: {c['text'][:160]}...\n")

    # 2. Plain LLM (No RAG)
    print("\n[2] WITHOUT RAG (Plain LLM):")
    print("-" * 60)
    print(ask_without_rag(question))

    # 3. Grounded LLM (With RAG)
    print("\n[3] WITH RAG (Grounded in Documents):")
    print("-" * 60)
    print(ask_with_rag(question, chunks))
    print("=" * 60)


if __name__ == "__main__":
    main()