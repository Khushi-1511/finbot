import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_pipeline import load_and_chunk, build_vectorstore, load_vectorstore, ask

load_dotenv(override=True)

app = FastAPI(
    title="FinBot API",
    description="RAG-powered financial document Q&A assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load vector store on startup ─────────────────────────────
if os.path.exists("faiss.index"):
    index, chunks = load_vectorstore()
else:
    chunks = load_and_chunk("data/rbi_report.pdf")
    index, chunks = build_vectorstore(chunks)

# ── Request/Response Models ───────────────────────────────────
class QuestionRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    chunks_retrieved: int
    latency_seconds: float

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "FinBot is running!",
        "usage": "POST /ask with JSON body: {query: 'your question'}"
    }

@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(chunks)}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start = time.time()
    answer = ask(request.query, index, chunks)
    latency = round(time.time() - start, 2)

    return AnswerResponse(
        question=request.query,
        answer=answer,
        chunks_retrieved=4,
        latency_seconds=latency
    )

@app.get("/metrics")
def metrics():
    return {
        "total_chunks_indexed": len(chunks),
        "embedding_model": "gemini-embedding-001",
        "llm_model": "llama-3.3-70b-versatile (via Groq)",
        "vector_store": "FAISS IndexFlatL2",
        "chunk_size": 50,
        "top_k_retrieval": 4
    }