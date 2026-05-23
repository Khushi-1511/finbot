import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── 1. Load & Chunk PDF ──────────────────────────────────────
def load_and_chunk(pdf_path: str, chunk_size: int = 50):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    words = full_text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - 10  # 10-word overlap
    print(f"✅ Loaded PDF → {len(chunks)} chunks")
    return chunks

# ── 2. Embed using Google API (no local model!) ───────────────
def embed_texts(texts: list[str]) -> np.ndarray:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    vectors = [e.values for e in result.embeddings]
    return np.array(vectors, dtype="float32")

# ── 3. Build FAISS Vector Store ──────────────────────────────
def build_vectorstore(chunks):
    print("⏳ Building embeddings via Google API...")
    # Embed in small batches (API limit)
    all_vectors = []
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vecs = embed_texts(batch)
        all_vectors.append(vecs)
        print(f"  Embedded {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")

    embeddings = np.vstack(all_vectors)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "faiss.index")
    with open("chunks.json", "w") as f:
        json.dump(chunks, f)
    print("✅ Vector store saved!")
    return index, chunks

def load_vectorstore():
    index = faiss.read_index("faiss.index")
    with open("chunks.json", "r") as f:
        chunks = json.load(f)
    print("✅ Vector store loaded!")
    return index, chunks

# ── 4. Retrieve Relevant Chunks ──────────────────────────────
def retrieve(query: str, index, chunks, top_k: int = 4):
    query_vec = embed_texts([query])
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

# ── 5. Ask Gemini ────────────────────────────────────────────
def ask(query: str, index, chunks):
    context_chunks = retrieve(query, index, chunks)
    context = "\n\n".join(context_chunks)

    prompt = f"""You are FinBot, an AI assistant for financial document analysis.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the document to answer this."
Be concise, factual, and precise.

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
)
    answer = response.choices[0].message.content
    print(f"\n🤖 Answer: {answer}")
    print(f"📄 Retrieved {len(context_chunks)} chunks")
    return answer

# ── 6. Quick Test ────────────────────────────────────────────
if __name__ == "__main__":
    # Delete old index files if they exist
    import os
    for f in ["faiss.index", "chunks.json"]:
        if os.path.exists(f):
            os.remove(f)

    chunks = load_and_chunk("data/rbi_report.pdf")
    index, chunks = build_vectorstore(chunks)

    ask("What is the minimum credit score required for a personal loan?", index, chunks)
    ask("What is the waiting period for pre-existing diseases in health insurance?", index, chunks)
    ask("What is the maximum tenure for a home loan?", index, chunks)