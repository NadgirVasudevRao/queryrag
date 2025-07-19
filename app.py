import os
import pickle
import streamlit as st
import requests
import pdfplumber
import docx2txt
import numpy as np
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# ─── Persistence paths ─────────────────────────────────────────────────
INDEX_FILE = "faiss.index"
DOCS_FILE  = "docs.pkl"

# ─── 1) CACHED RESOURCES ────────────────────────────────────────────────

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_generator():
    return pipeline(
        "text-generation",
        model="tiiuae/falcon-rw-1b",
        device="cpu",           # set to 0 for GPU
        max_new_tokens=150,
        do_sample=False
    )

# ─── 2) EXTRACTION ───────────────────────────────────────────────────────

def extract_text(source, mode):
    if mode == "Website URL":
        r = requests.get(source, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return "\n".join(p.get_text() for p in soup.find_all("p"))
    else:
        if source.type == "application/pdf":
            text = ""
            with pdfplumber.open(source) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        elif source.type == (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ):
            return docx2txt.process(source)
        else:
            return source.read().decode("utf-8")

# ─── 3) CHUNKING, EMBEDDING & INDEXING ──────────────────────────────────

def chunk_and_index(text):
    # 3.1) Chunk with larger size
    splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    docs = splitter.split_text(text)

    # 3.2) Batch‐embed
    model = get_embedder()
    embs = model.encode(docs, batch_size=32, show_progress_bar=False)
    embs = np.array(embs, dtype="float32")

    # 3.3) Handle single‐vector edge‐case
    if embs.ndim == 1:
        dim = embs.shape[0]
        matrix = embs.reshape(1, -1)
    else:
        dim = embs.shape[1]
        matrix = embs

    # 3.4) Build & persist index
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

    return docs, index

def load_persisted_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        idx  = faiss.read_index(INDEX_FILE)
        docs = pickle.load(open(DOCS_FILE, "rb"))
        return docs, idx
    return None, None

# ─── 4) QUERY & GENERATION ───────────────────────────────────────────────

def chat_with_content(query, docs, index):
    if not docs or index.ntotal == 0:
        return "No content indexed. Please process first.", ""

    q_emb = get_embedder().encode(query).reshape(1, -1)
    k = min(3, len(docs))
    _, inds = index.search(q_emb, k)
    context = "\n\n".join(docs[i] for i in inds[0] if 0 <= i < len(docs))

    prompt = f"""
You are a helpful assistant. Only output the direct answer—do NOT repeat headers.

Context:
{context}

Question:
{query}

Answer:
"""
    raw = get_generator()(prompt)[0]["generated_text"]

    # strip echoes
    if "Answer:" in raw:
        raw = raw.split("Answer:", 1)[1].strip()
    for marker in ("Question:", "Context:"):
        if marker in raw:
            raw = raw.split(marker, 1)[0].strip()

    return raw, context

# ─── 5) STREAMLIT UI ─────────────────────────────────────────────────────

st.set_page_config(page_title="QueryRAG", layout="wide")
st.sidebar.title("🔎 Input Source")

# Load persisted index if available
docs, idx = load_persisted_index()
if docs:
    st.session_state.docs  = docs
    st.session_state.index = idx

# Input mode
mode = st.sidebar.radio(
    "Input source",
    ["Website URL", "Upload File"],
    label_visibility="visible"
)
if mode == "Website URL":
    source = st.sidebar.text_input("Enter URL")
else:
    source = st.sidebar.file_uploader("Choose a file", type=["pdf","txt","docx"])

if st.sidebar.button("🔄 Process & Index"):
    if not source:
        st.sidebar.error("Provide a URL or file.")
    else:
        progress = st.sidebar.progress(0)
        st.sidebar.write("Processing…")
        raw = extract_text(source, mode)
        docs, idx = chunk_and_index(raw)
        st.session_state.docs  = docs
        st.session_state.index = idx
        progress.progress(100)
        st.sidebar.success("✅ Done and saved!")

        # Rerun so chat UI updates
        st.experimental_rerun()

# Chat UI
st.title("🗣️ QueryRAG Chatbot")
if "index" in st.session_state and st.session_state.index.ntotal > 0:
    q = st.text_input("Your question:")
    if q:
        answer, ctx = chat_with_content(
            q, st.session_state.docs, st.session_state.index
        )
        st.markdown(f"**Answer:**  {answer}")
        with st.expander("Show context"):
            st.write(ctx)
else:
    st.info("▶ Please Process & Index first.")
