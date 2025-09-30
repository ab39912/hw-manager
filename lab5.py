# HW5.py — Intelligent Chatbot with Short-Term Memory (RAG + Diagnostics)

# ── SQLite shim (fixes Chroma's sqlite>=3.35 requirement)
try:
    import pysqlite3
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import os, glob, textwrap
from typing import List, Tuple, Dict

import streamlit as st
from openai import OpenAI, BadRequestError

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pypdf import PdfReader

# ── Config (editable in sidebar too)
DEFAULT_PDF_DIR   = "lab4_pdfs"
DEFAULT_DB_PATH   = ".chroma_hw5"
DEFAULT_COLL_NAME = "HW5Collection"

CHAT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a helpful Course Information Assistant.\n"
    "Use retrieved course context when possible; if none, say so clearly, then answer generally.\n"
    "Be concise; cite doc filenames and page numbers if you used them."
)

TOP_K = 3
PAGE_CHUNK_SIZE = 2000
CHARS_PER_DOC = 1200
BATCH_SIZE = 64


# ─────────────────────────────────────────────────────────────
# Vector DB helpers
# ─────────────────────────────────────────────────────────────
def _embedding_fn(api_key: str):
    return OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")

def _client(db_path: str):
    return chromadb.PersistentClient(path=db_path)

def _get_or_create_collection(db_path: str, coll_name: str, api_key: str):
    client = _client(db_path)
    emb = _embedding_fn(api_key)
    return client.get_or_create_collection(
        name=coll_name,
        embedding_function=emb,
        metadata={"hnsw:space": "cosine"},
    )

# ── PDF → chunks
def _pdf_to_page_texts(path: str) -> List[str]:
    reader = PdfReader(path)
    return [(page.extract_text() or "").strip() for page in reader.pages]

def _split_long_text(txt: str, chunk_size: int = PAGE_CHUNK_SIZE) -> List[str]:
    txt = txt.strip()
    if not txt:
        return []
    if len(txt) <= chunk_size:
        return [txt]
    chunks, start, n = [], 0, len(txt)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = txt.rfind(" ", start, end)
            if cut != -1 and cut > start + int(0.6 * chunk_size):
                end = cut
        chunks.append(txt[start:end].strip())
        start = end
    return [c for c in chunks if c]

def _pdf_to_chunks_with_meta(path: str) -> List[Tuple[str, Dict]]:
    filename = os.path.basename(path)
    results = []
    for p_idx, page_txt in enumerate(_pdf_to_page_texts(path), start=1):
        for c_idx, chunk in enumerate(_split_long_text(page_txt), start=1):
            if not chunk: 
                continue
            results.append((chunk, {"filename": filename, "page": p_idx, "chunk": c_idx}))
    return results

# ── Build / Rebuild collection
def build_vector_db(pdf_dir: str, db_path: str, coll_name: str, api_key: str) -> int:
    collection = _get_or_create_collection(db_path, coll_name, api_key)

    # if already has vectors, skip
    try:
        if (collection.count() or 0) > 0:
            return collection.count()
    except Exception:
        pass

    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdfs:
        st.error(f"No PDFs found in '{pdf_dir}'. Put your 7 course PDFs there or change the path in sidebar.")
        st.stop()

    texts, metas, ids = [], [], []
    for path in pdfs:
        chunks = _pdf_to_chunks_with_meta(path)
        for (chunk_text, md) in chunks:
            ids.append(f"{md['filename']}|p{md['page']}|c{md['chunk']}")
            texts.append(chunk_text)
            metas.append(md)

    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        j = min(i + BATCH_SIZE, total)
        _get_or_create_collection(db_path, coll_name, api_key).add(
            ids=ids[i:j], documents=texts[i:j], metadatas=metas[i:j]
        )
    return _get_or_create_collection(db_path, coll_name, api_key).count()

# ── Retrieval
def retrieve_context(query: str, db_path: str, coll_name: str, api_key: str, n_results: int = TOP_K):
    collection = _get_or_create_collection(db_path, coll_name, api_key)
    res = collection.query(query_texts=[query], n_results=n_results)

    metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    documents = res.get("documents", [[]])[0] if res.get("documents") else []

    used_labels, parts = [], []
    for md, doc in zip(metadatas, documents):
        if not doc:
            continue
        label = f"{md.get('filename','?')} (p.{md.get('page','?')})"
        used_labels.append(label)
        parts.append(f"[{label}]\n{textwrap.dedent(doc[:CHARS_PER_DOC]).strip()}")
    return used_labels, "\n\n---\n\n".join(parts)

# ── Prompt
def build_messages(user_query: str, context_block: str, history: list):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, text in history:
        msgs.append({"role": role, "content": text})
    if context_block:
        user = (
            f"User question:\n{user_query}\n\n"
            "Use ONLY this retrieved course material to answer. Cite filenames and pages inline.\n"
            "===== CONTEXT START =====\n"
            f"{context_block}\n"
            "===== CONTEXT END =====\n"
        )
    else:
        user = (
            f"User question:\n{user_query}\n\n"
            "No course context retrieved. Answer generally and say you did not use course PDFs."
        )
    msgs.append({"role": "user", "content": user})
    return msgs

# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────
def run():
    st.title("🗒️ HW5: Intelligent Chatbot with Short-Term Memory")

    # API key
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Missing OPENAI_API_KEY in `.streamlit/secrets.toml`.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Sidebar controls + diagnostics
    with st.sidebar:
        st.header("RAG Settings")
        pdf_dir = st.text_input("PDF folder", value=DEFAULT_PDF_DIR)
        db_path = st.text_input("Chroma path", value=DEFAULT_DB_PATH)
        coll_name = st.text_input("Collection name", value=DEFAULT_COLL_NAME)
        force_rebuild = st.checkbox("Force rebuild from PDFs", value=False)
        st.caption("If you change any setting above, toggle ‘Force rebuild’ once.")
        st.divider()
        st.subheader("Diagnostics")
        if st.button("Run diagnostics probe"):
            try:
                coll = _get_or_create_collection(db_path, coll_name, api_key)
                count = coll.count() or 0
                st.info(f"Collection `{coll_name}` has {count} vectors.")
                if count == 0:
                    st.warning("No vectors: you must build from PDFs.")
            except Exception as e:
                st.error(f"Collection error: {e}")

    # Build / Rebuild vector DB
    # Rebuild if first run, or if user requested force rebuild, or if collection is empty
    coll = _get_or_create_collection(db_path, coll_name, api_key)
    need_build = force_rebuild or (coll.count() or 0) == 0
    if need_build:
        with st.status("Building vector DB from PDFs…", expanded=True) as status:
            count = build_vector_db(pdf_dir, db_path, coll_name, api_key)
            status.update(label=f"Vector DB ready with {count} chunks.", state="complete")
    else:
        st.caption(f"Vector DB OK — `{coll_name}` has {coll.count()} chunks.")

    # Chat memory
    if "hw5_history" not in st.session_state:
        st.session_state.hw5_history = []  # list[(role, text)]

    # Render past turns
    for role, text in st.session_state.hw5_history:
        with st.chat_message(role):
            st.markdown(text)

    # Input
    user_query = st.chat_input("Ask a course question (e.g., Which courses focus Python?)")
    if not user_query:
        return

    # Add user turn
    st.session_state.hw5_history.append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve
    used_labels, context_block = retrieve_context(user_query, db_path, coll_name, api_key, n_results=TOP_K)

    # Answer
    with st.chat_message("assistant"):
        if used_labels:
            st.markdown("**Using course docs (RAG):** " + ", ".join(used_labels))
        else:
            st.markdown("**No relevant course docs found** — answering generally.")

        try:
            messages = build_messages(user_query, context_block, st.session_state.hw5_history)
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
            answer = resp.choices[0].message.content
            st.markdown(answer)

            # Show snippet previews when we used RAG
            if used_labels:
                st.markdown("**Retrieved snippets:**")
                for lbl, part in zip(used_labels, context_block.split("\n\n---\n\n")):
                    preview = part.split("\n", 1)[-1].replace("\n", " ")
                    preview = (preview[:400] + "…") if len(preview) > 400 else preview
                    st.markdown(f"- **{lbl}** — {preview}")
        except BadRequestError:
            st.error("OpenAI BadRequestError — check model name & API key.")
            return

    # Save assistant reply
    st.session_state.hw5_history.append(("assistant", answer))


if __name__ == "__main__":
    run()
