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

# ─── Persistence filenames ───────────────────────────────────────────────
INDEX_FILE = "faiss.index"
DOCS_FILE  = "docs.pkl"

# ─── 1) CACHED RESOURCES ─────────────────────────────────────────────────
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_generator():
    return pipeline(
        "text-generation",
        model="tiiuae/falcon-rw-1b",
        device="cpu",           # set to 0 for GPU if available
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
        elif source.type.startswith("application/vnd.openxmlformats"):
            return docx2txt.process(source)
        else:
            return source.read().decode("utf-8")

# ─── 3) CHUNKING, EMBEDDING & INDEXING ──────────────────────────────────
def chunk_and_index(text):
    # split into ~3000‑char chunks
    splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    docs = splitter.split_text(text)

    # batch‑embed all chunks
    embedder = get_embedder()
    embs = embedder.encode(docs, batch_size=32, show_progress_bar=False)
    embs = np.array(embs, dtype="float32")

    # handle single‑chunk edge case
    if embs.ndim == 1:
        dim = embs.shape[0]
        matrix = embs.reshape(1, -1)
    else:
        dim = embs.shape[1]
        matrix = embs

    # build & persist FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

    return docs, index

# ─── 4) LOAD OR REBUILD INDEX ────────────────────────────────────────────
def load_or_clear_index():
    """
    Tries to load an existing index/docs; if any error or dimension mismatch,
    deletes the files and returns (None, None).
    """
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        try:
            idx  = faiss.read_index(INDEX_FILE)
            docs = pickle.load(open(DOCS_FILE, "rb"))
            expected_dim = get_embedder().get_sentence_embedding_dimension()
            # faiss index has attribute d for dimension
            if getattr(idx, "d", idx.ntotal) != expected_dim:
                raise ValueError("Dimension mismatch")
            return docs, idx
        except Exception:
            # stale or corrupted → delete and start fresh
            for fn in (INDEX_FILE, DOCS_FILE):
                try: os.remove(fn)
                except: pass
    return None, None

# ─── 5) QUERY & GENERATION ───────────────────────────────────────────────
def chat_with_content(query, docs, index):
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

    # strip any echoed headers
    if "Answer:" in raw:
        raw = raw.split("Answer:", 1)[1].strip()
    for marker in ("Question:", "Context:"):
        if marker in raw:
            raw = raw.split(marker, 1)[0].strip()

    return raw, context

# ─── 6) STREAMLIT UI ─────────────────────────────────────────────────────
st.set_page_config(page_title="QueryRAG", layout="wide")
st.sidebar.title("🔎 Input Source")

# load or clear any on-disk index at startup
docs, idx = load_or_clear_index()
if docs:
    st.session_state.docs  = docs
    st.session_state.index = idx

mode = st.sidebar.radio(
    "Input source",
    ["Website URL", "Upload File"],
    label_visibility="visible"
)
source = (
    st.sidebar.text_input("Enter URL")
    if mode=="Website URL"
    else st.sidebar.file_uploader("Choose a file", type=["pdf","txt","docx"])
)

if st.sidebar.button("🔄 Process & Index"):
    if not source:
        st.sidebar.error("Please provide a URL or upload a file.")
    else:
        progress = st.sidebar.progress(0)
        st.sidebar.info("Indexing content…")
        raw = extract_text(source, mode)
        docs, idx = chunk_and_index(raw)
        st.session_state.docs  = docs
        st.session_state.index = idx
        progress.progress(100)
        st.sidebar.success("✅ Indexed!")

st.title("🗣️ QueryRAG Chatbot")
if "index" in st.session_state and st.session_state.index.ntotal > 0:
    st.caption(f"▸ Indexed {len(st.session_state.docs)} chunks")
    q = st.text_input("Your question:")
    if q:
        answer, ctx = chat_with_content(q, st.session_state.docs, st.session_state.index)
        st.markdown(f"**Answer:** {answer}")
        with st.expander("Show context"):
            st.write(ctx)
else:
    st.info("▶ Please click ‘Process & Index’ first in the sidebar.")
