import streamlit as st
import time
from rag_pipeline import load_and_chunk, build_vectorstore, load_vectorstore, ask
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FinBot",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTextInput input { background-color: #1e2130; color: white; }
    .metric-card {
        background: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3148;
        margin: 5px 0;
    }
    .answer-box {
        background: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #0e9f6e;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Vector Store ─────────────────────────────────────────
@st.cache_resource
def load_index():
    if os.path.exists("faiss.index"):
        return load_vectorstore()
    else:
        chunks = load_and_chunk("data/rbi_report.pdf")
        return build_vectorstore(chunks)

# ── Header ───────────────────────────────────────────────────
st.markdown("# 🤖 FinBot")
st.markdown("#### RAG-Powered Financial Document Assistant")
st.divider()

# ── Layout ───────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📋 Sample Questions")
    samples = [
        "What is the minimum credit score for a personal loan?",
        "What is the maximum tenure for a home loan?",
        "What is the waiting period for pre-existing diseases?",
        "What are the eligibility criteria for self-employed?",
        "How long does it take to resolve a grievance?",
    ]
    for s in samples:
        if st.button(s, use_container_width=True):
            st.session_state.query = s

    st.divider()
    st.markdown("### ⚙️ System Info")
    st.markdown("""
    <div class='metric-card'>
        <b>LLM:</b> Llama 3.3 70B (Groq)<br>
        <b>Embeddings:</b> Gemini-001<br>
        <b>Vector DB:</b> FAISS<br>
        <b>Framework:</b> RAG from scratch
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 💬 Ask a Question")

    # Load index
    with st.spinner("Loading knowledge base..."):
        index, chunks = load_index()
    st.success(f"✅ Knowledge base loaded — {len(chunks)} chunks indexed")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "meta" in msg:
                st.caption(f"⏱ {msg['meta']['latency']}s · 📄 {msg['meta']['chunks']} chunks retrieved")

    # Input
    query = st.chat_input("Ask about loans, insurance, eligibility...")

    # Handle sample button clicks
    if "query" in st.session_state and st.session_state.query:
        query = st.session_state.query
        st.session_state.query = None

    if query:
        # Show user message
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.time()
                answer = ask(query, index, chunks)
                latency = round(time.time() - start, 2)
            st.write(answer)
            st.caption(f"⏱ {latency}s · 📄 4 chunks retrieved")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "meta": {"latency": latency, "chunks": 4}
        })