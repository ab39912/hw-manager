
# ============================================================
# HW5: Short-Term Memory Chatbot with Vector Search (RAG)
# ============================================================

# --- SQLite shim (fixes Chroma sqlite3 version issue) ---
try:
    import pysqlite3  # ships modern SQLite
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
import glob
import textwrap
from typing import List, Tuple, Dict

import streamlit as st
from openai import OpenAI, BadRequestError

# ChromaDB
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# PDF reading
from pypdf import PdfReader


# ======================
# Config
# ======================
CHAT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a helpful assistant for course information.\n"
    "Always use retrieved course context when possible.\n"
    "If no relevant context is found, clearly say so, then answer generally.\n"
    "Keep answers short, clear, and in plain English."
)

PDF_DIR = "lab4_pdfs"          # put your 7 PDFs here
CHROMA_PATH = ".chroma_lab5"   # persistent vector DB folder
COLLECTION_NAME = "HW5Collection"

TOP_K = 3
PAGE_CHUNK_SIZE = 2000
CHARS_PER_DOC = 1200
BATCH_SIZE = 64


# ======================
# Chroma Helpers
# ======================
def _embedding_fn(api_key: str):
    return OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")

def _client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def _get_or_create_collection(api_key: str):
    client = _client()
    emb = _embedding_fn(api_key)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb,
        metadata={"hnsw:space": "cosine"},
    )


# ======================
# PDF → text chunks
# ======================
def _pdf_to_page_texts(path: str) -> List[str]:
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]

def _split_long_text(txt: str, chunk_size: int = PAGE_CHUNK_SIZE) -> List[str]:
    txt = txt.strip()
    if not txt:
        return []
    if len(txt) <= chunk_size:
        return [txt]

    chunks = []
    start, n = 0, len(txt)
    while start < n:
        end = min(start + chunk_size, n)
        # cut on space if possible
        if end < n:
            space = txt.rfind(" ", start, end)
            if space != -1 and space > start + int(0.6 * chunk_size):
                end = space
        chunks.append(txt[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _pdf_to_chunks_with_meta(path: str) -> List[Tuple[str, Dict]]:
    """Return list of (chunk_text, metadata) for one PDF."""
    filename = os.path.basename(path)
    results = []
    page_texts = _pdf_to_page_texts(path)
    for p_idx, page_txt in enumerate(page_texts, start=1):
        for c_idx, chunk in enumerate(_split_long_text(page_txt), start=1):
            if not chunk:
                continue
            md = {"filename": filename, "page": p_idx, "chunk": c_idx}
            results.append((chunk, md))
    return results


# ======================
# Build Chroma collection
# ======================
def initialize_vector_db(pdf_dir: str, api_key: str):
    st.session_state.setdefault("HW5_chroma_path", CHROMA_PATH)
    st.session_state.setdefault("HW5_collection_name", COLLECTION_NAME)

    collection = _get_or_create_collection(api_key)

    # Skip if already built
    if (collection.count() or 0) > 0:
        st.session_state["HW5_vectorDB_ready"] = True
        return

    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        st.error(f"No PDFs found in '{pdf_dir}'. Place your 7 course PDFs there.")
        st.stop()

    texts, metas, ids = [], [], []
    for path in pdf_paths:
        chunks = _pdf_to_chunks_with_meta(path)
        for (chunk_text, md) in chunks:
            cid = f"{md['filename']}|p{md['page']}|c{md['chunk']}"
            ids.append(cid)
            texts.append(chunk_text)
            metas.append(md)

    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        j = min(i + BATCH_SIZE, total)
        collection.add(ids=ids[i:j], documents=texts[i:j], metadatas=metas[i:j])

    st.session_state["HW5_vectorDB_ready"] = True


# ======================
# Retrieval
# ======================
def retrieve_context(query: str, api_key: str, n_results: int = TOP_K):
    if not st.session_state.get("HW5_vectorDB_ready", False):
        initialize_vector_db(PDF_DIR, api_key)

    collection = _get_or_create_collection(api_key)
    res = collection.query(query_texts=[query], n_results=n_results)

    metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    documents = res.get("documents", [[]])[0] if res.get("documents") else []

    used_docs, parts = [], []
    for md, doc in zip(metadatas, documents):
        snippet = (doc or "")[:CHARS_PER_DOC]
        used_docs.append(md)
        parts.append(f"[{md.get('filename')} p.{md.get('page')}] {snippet}")

    return "\n\n".join(parts)


# ======================
# Build Messages
# ======================
def build_messages(query: str, context: str, history: list):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Add history
    for role, text in history:
        messages.append({"role": role, "content": text})
    # Add current turn
    if context:
        user_msg = (
            f"User question: {query}\n\n"
            f"Relevant course information:\n{context}\n\n"
            "Answer clearly, citing doc filenames + page numbers."
        )
    else:
        user_msg = f"User question: {query}\n\nNo course info retrieved. Answer generally."
    messages.append({"role": "user", "content": user_msg})
    return messages


# ======================
# Streamlit UI
# ======================
def run():
    st.title("📝 HW5: Course Chatbot with Short-Term Memory")

    # API key
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Missing OPENAI_API_KEY in `.streamlit/secrets.toml`.")
        st.stop()

    client = OpenAI(api_key=api_key)

    if "hw5_history" not in st.session_state:
        st.session_state.hw5_history = []

    # Render history
    for role, text in st.session_state.hw5_history:
        with st.chat_message(role):
            st.markdown(text)

    query = st.chat_input("Ask about courses")
    if not query:
        return

    # User message
    st.session_state.hw5_history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve context
    context = retrieve_context(query, api_key, n_results=TOP_K)

    # Build messages
    messages = build_messages(query, context, st.session_state.hw5_history)

    # LLM answer
    with st.chat_message("assistant"):
        try:
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
            answer = resp.choices[0].message.content
            st.markdown(answer)
        except BadRequestError:
            st.error("OpenAI BadRequestError — check model & API key.")
            return

    # Save assistant reply
    st.session_state.hw5_history.append(("assistant", answer))


if __name__ == "__main__":
    run()
