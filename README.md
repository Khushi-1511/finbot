# 🤖 FinBot — RAG-Powered Financial Q&A Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers natural language questions from financial documents — built from scratch without LangChain.

---

## 🎯 What It Does

Ask questions in plain English about any financial document — loan policies, insurance terms, eligibility criteria. FinBot retrieves the most relevant context and generates accurate, grounded answers with no hallucination.

**Example queries:**
- *"What is the minimum credit score for a personal loan?"* → **720**
- *"What is the maximum tenure for a home loan?"* → **30 years**
- *"What is the waiting period for pre-existing diseases?"* → **2 years**

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[Query Embedding]          ← Google Gemini Embedding API
    │
    ▼
[FAISS Vector Search]      ← Top-4 semantically similar chunks
    │
    ▼
[Prompt Engineering]       ← Context + instruction template
    │
    ▼
[Llama 3.3 70B via Groq]  ← Fast LLM inference (~1s latency)
    │
    ▼
Grounded Answer
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Llama 3.3 70B (via Groq) |
| Embeddings | Google Gemini Embedding 001 |
| Vector Store | FAISS (IndexFlatL2) |
| PDF Parsing | PyPDF |
| API Layer | FastAPI |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Khushi-1511/finbot.git
cd finbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your API keys
echo "GEMINI_API_KEY=your_key" > .env
echo "GROQ_API_KEY=your_key" >> .env

# Run Streamlit app
python -m streamlit run app.py

# Or run FastAPI server
uvicorn main:app --reload
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Server status + chunks loaded |
| POST | `/ask` | Ask a question |
| GET | `/metrics` | System metrics |

**Example request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the minimum credit score for a personal loan?"}'
```

**Example response:**
```json
{
  "question": "What is the minimum credit score for a personal loan?",
  "answer": "The minimum credit score required for a personal loan is 720.",
  "chunks_retrieved": 4,
  "latency_seconds": 1.03
}
```

---

## 🧠 Key Concepts Implemented

- **Chunking with overlap** — 50-word chunks, 10-word overlap to preserve context boundaries
- **Semantic search** — Dense vector similarity via FAISS IndexFlatL2
- **Prompt engineering** — Structured template with grounding instructions to prevent hallucination
- **Latency tracking** — Response time monitoring per query
- **Quality metrics** — Relevance, factual grounding, chunk retrieval accuracy

---

## 📁 Project Structure

```
finbot/
├── app.py              # Streamlit UI
├── main.py             # FastAPI server
├── rag_pipeline.py     # Core RAG logic
├── data/
│   └── rbi_report.pdf  # Sample financial document
├── requirements.txt
└── .env                # API keys (not committed)
```

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google AI Studio API key |
| `GROQ_API_KEY` | Groq Console API key |

---

Built by **Khushi Wadhwa** · [LinkedIn](https://www.linkedin.com/in/khushi-wadhwa-50a8a6214/) · [GitHub](https://github.com/Khushi-1511)
