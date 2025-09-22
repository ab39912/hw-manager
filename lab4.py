# lab4.py

# ============================================================
# SQLite shim: ensure sqlite3 >= 3.35 for Chroma without OS upgrades
# Requires 'pysqlite3-binary' in requirements.txt.
# ============================================================
try:
    import pysqlite3  # modern SQLite
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
import glob
import math
import textwrap
from typing import List, Tuple, Dict

import streamlit as st
from openai import OpenAI, BadRequestError

# Chroma / PDF
try:
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
except Exception:
    chromadb = None
    OpenAIEmbeddingFunction = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# ======================
# App / RAG configuration
# ======================
CHAT_MODEL = "gpt-4o-mini"  # use a valid model you have access to
SYSTEM_PROMPT = (
    "You are a helpful Course Information Assistant.\n"
    "When context from course PDFs is provided, rely on it first and be concise and clear.\n"
    "If the context seems unrelated or empty, say that you're answering generally.\n"
    "Always write in short, clear sentences."
)

DEFAULT_PDF_DIR = "hw4_pdfs"
CHROMA_PATH = ".chroma_lab4"       # persistent directory
COLLECTION_NAME = "Lab4Collection"

TOP_K = 3
CHARS_PER_DOC = 1200               # LLM context snippet cap per retrieved chunk
PAGE_CHUNK_SIZE = 2000             # characters per chunk (kept well under embed limits)
BATCH_SIZE = 64                    # add() batch size


# ======================
# OpenAI chat streaming
# ======================
def stream_openai_response(client: OpenAI, messages, model: str = CHAT_MODEL):
    response_text = ""
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            response_text += delta
            yield delta
    stream.full_text = response_text  # type: ignore
    return


# ======================
# Chroma helpers
# ======================
def _embedding_fn(openai_api_key: str):
    if OpenAIEmbeddingFunction is None:
        raise RuntimeError("Missing dependency: chromadb. Please install it.")
    return OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")


def _client():
    if chromadb is None:
        raise RuntimeError("Missing dependency: chromadb. Please install it.")
    return chromadb.PersistentClient(path=st.session_state.get("Lab4_chroma_path", CHROMA_PATH))


def _get_or_create_collection(openai_api_key: str):
    client = _client()
    emb = _embedding_fn(openai_api_key)
    # Prefer get_or_create if available
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(
            name=st.session_state.get("Lab4_collection_name", COLLECTION_NAME),
            embedding_function=emb,
            metadata={"hnsw:space": "cosine"},
        )
    # Fallback
    try:
        return client.get_collection(
            name=st.session_state.get("Lab4_collection_name", COLLECTION_NAME),
            embedding_function=emb,
        )
    except Exception:
        return client.create_collection(
            name=st.session_state.get("Lab4_collection_name", COLLECTION_NAME),
            embedding_function=emb,
            metadata={"hnsw:space": "cosine"},
        )


# ======================
# PDF → chunks
# ======================
def _pdf_to_page_texts(path: str) -> List[str]:
    if PdfReader is None:
        raise RuntimeError("Missing dependency: pypdf. Please install it.")
    try:
        reader = PdfReader(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF '{path}': {e}")
    pages = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(txt.strip())
    return pages


def _split_long_text(txt: str, chunk_size: int = PAGE_CHUNK_SIZE) -> List[str]:
    txt = txt.strip()
    if not txt:
        return []
    if len(txt) <= chunk_size:
        return [txt]
    # Simple char-based splitting with whitespace respect
    chunks = []
    start = 0
    n = len(txt)
    while start < n:
        end = min(start + chunk_size, n)
        # try to cut on a space near the end
        if end < n:
            space = txt.rfind(" ", start, end)
            if space != -1 and space > start + int(0.6 * chunk_size):
                end = space
        chunks.append(txt[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _pdf_to_chunks_with_meta(path: str) -> List[Tuple[str, Dict]]:
    """Return list of (chunk_text, metadata) for one PDF, per page and sub-chunk."""
    filename = os.path.basename(path)
    results = []
    page_texts = _pdf_to_page_texts(path)
    for p_idx, page_txt in enumerate(page_texts, start=1):
        for c_idx, chunk in enumerate(_split_long_text(page_txt), start=1):
            if not chunk:
                continue
            md = {
                "filename": filename,
                "source_path": path,
                "page": p_idx,
                "chunk": c_idx,
            }
            results.append((chunk, md))
    return results


# ======================
# Build collection once (idempotent, chunked, batched)
# ======================
def initialize_lab4_vector_db(pdf_dir: str, openai_api_key: str):
    """Ensure collection exists and is populated (once)."""
    if chromadb is None or OpenAIEmbeddingFunction is None:
        st.error("Missing dependency: chromadb.")
        st.stop()
    if PdfReader is None:
        st.error("Missing dependency: pypdf.")
        st.stop()

    st.session_state.setdefault("Lab4_chroma_path", CHROMA_PATH)
    st.session_state.setdefault("Lab4_collection_name", COLLECTION_NAME)

    collection = _get_or_create_collection(openai_api_key)

    # If it already has vectors, we are done
    try:
        if (collection.count() or 0) > 0:
            st.session_state["Lab4_vectorDB_ready"] = True
            return
    except Exception:
        # Try a probe
        try:
            probe = collection.query(query_texts=["probe"], n_results=1)
            if bool(probe.get("ids")):
                st.session_state["Lab4_vectorDB_ready"] = True
                return
        except Exception:
            pass

    # Build from PDFs (first time)
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        st.error(f"No PDFs found in '{pdf_dir}'. Place your 7 PDF files there.")
        st.stop()
    if len(pdf_paths) != 7:
        st.warning(f"Expected 7 PDFs, found {len(pdf_paths)}. Continuing anyway...")

    # Prepare chunks & metadata
    texts: List[str] = []
    metas: List[Dict] = []
    ids:   List[str] = []

    for f_idx, path in enumerate(pdf_paths, start=1):
        chunks = _pdf_to_chunks_with_meta(path)
        if not chunks:
            st.warning(f"PDF '{os.path.basename(path)}' produced no text; skipping.")
            continue
        for (chunk_text, md) in chunks:
            # Unique, stable id: filename-page-chunk
            cid = f"{md['filename']}|p{md['page']}|c{md['chunk']}"
            ids.append(cid)
            texts.append(chunk_text)
            metas.append(md)

    if not texts:
        st.error("All PDFs were empty or failed to parse; aborting vector DB creation.")
        st.stop()

    # Add in batches to avoid InternalError & API limits
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        j = min(i + BATCH_SIZE, total)
        try:
            collection.add(ids=ids[i:j], documents=texts[i:j], metadatas=metas[i:j])
        except Exception as e:
            st.error(f"Failed adding batch {i//BATCH_SIZE+1}: {e}")
            st.stop()

    st.session_state["Lab4_vectorDB_ready"] = True


# ======================
# Retrieval for RAG
# ======================
def retrieve_context(query: str, openai_api_key: str, pdf_dir: str, n_results: int = TOP_K):
    """Return (used_docs, context_block) for the LLM."""
    if not st.session_state.get("Lab4_vectorDB_ready", False):
        initialize_lab4_vector_db(pdf_dir, openai_api_key)

    collection = _get_or_create_collection(openai_api_key)

    try:
        res = collection.query(query_texts=[query], n_results=n_results)
    except Exception as e:
        st.error(f"Vector DB query error: {e}")
        return [], ""

    metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    documents = res.get("documents", [[]])[0] if res.get("documents") else []

    used_docs = []
    for md, doc in zip(metadatas, documents):
        fname = md.get("filename", "unknown.pdf") if md else "unknown.pdf"
        page = md.get("page")
        snippet = (doc or "")[:CHARS_PER_DOC]
        used_docs.append({"filename": fname, "page": page, "snippet": snippet})

    if not used_docs:
        return [], ""

    parts = []
    for i, d in enumerate(used_docs, start=1):
        label = f"{d['filename']}" + (f" (p.{d['page']})" if d.get("page") else "")
        parts.append(f"[Doc {i}: {label}]\n" + textwrap.dedent(d["snippet"]).strip())
    context_block = "\n\n---\n\n".join(parts)
    return used_docs, context_block


def build_messages(user_query: str, context_block: str):
    if context_block:
        system = SYSTEM_PROMPT
        user = (
            "User question:\n"
            f"{user_query}\n\n"
            "Use the following course materials context to answer. If the context does not contain the answer, "
            "say that clearly and then answer generally if you can.\n"
            "==== COURSE CONTEXT START ====\n"
            f"{context_block}\n"
            "==== COURSE CONTEXT END ====\n"
            "Cite the document names (and page numbers) inline in plain text where relevant."
        )
    else:
        system = SYSTEM_PROMPT
        user = (
            "No course context is available for this question. "
            "Answer generally and say that you are not using course PDFs.\n\n"
            f"User question:\n{user_query}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ======================
# Main UI (RAG chat)
# ======================
def run():
    st.title("📚 Lab 4b: Course Information Chatbot (RAG)")
    st.caption("Builds a persistent ChromaDB from 7 PDFs (chunked) and answers using retrieved context.")

    # API key
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Missing OPENAI_API_KEY in `.streamlit/secrets.toml`.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Vector DB Setup")
        pdf_dir = st.text_input("PDF folder", value=DEFAULT_PDF_DIR, help="Folder containing your 7 PDFs")
        build_now = st.checkbox("Build on load if missing", value=True)
        st.caption("Chunked + persistent collection avoids embedding every rerun.")
        st.divider()
        st.header("Chat Settings")
        use_rag = st.checkbox("Use course PDFs (RAG)", value=True)
        st.markdown(f"**LLM:** `{CHAT_MODEL}`")

    # Build if needed
    if build_now and not st.session_state.get("Lab4_vectorDB_ready", False):
        initialize_lab4_vector_db(pdf_dir, openai_api_key)

    # Chat UI
    client = OpenAI(api_key=openai_api_key)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render transcript
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a course-related question")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG retrieve
    used_docs = []
    context_block = ""
    if use_rag:
        used_docs, context_block = retrieve_context(user_input, openai_api_key, pdf_dir, n_results=TOP_K)

    messages = build_messages(user_input, context_block)

    with st.chat_message("assistant"):
        if use_rag and used_docs:
            labels = []
            for d in used_docs:
                label = d["filename"] + (f" (p.{d['page']})" if d.get("page") else "")
                labels.append(label)
            st.markdown("**Using course docs (RAG):** " + ", ".join(labels))
        elif use_rag and not used_docs:
            st.markdown("**No relevant course docs found** — answering generally.")
        else:
            st.markdown("**RAG disabled** — answering generally.")

        try:
            text_stream = stream_openai_response(client, messages, model=CHAT_MODEL)
            full_answer = st.write_stream(text_stream)
        except BadRequestError:
            st.error("OpenAI BadRequestError — check your model name and API key. Try `gpt-4o-mini`.")
            return

        # Show short previews inside the chat bubble
        if use_rag and used_docs:
            previews = []
            for i, d in enumerate(used_docs, start=1):
                snippet = (d["snippet"] or "").strip().replace("\n", " ")
                if len(snippet) > 400:
                    snippet = snippet[:400].rstrip() + "…"
                label = d["filename"] + (f" (p.{d['page']})" if d.get("page") else "")
                previews.append(f"**Doc {i}: {label}** — {snippet}")
            st.markdown("\n\n**Retrieved snippets:**\n\n" + "\n\n".join(previews))

    st.session_state.messages.append({"role": "assistant", "content": full_answer})


if __name__ == "__main__":
    run()
