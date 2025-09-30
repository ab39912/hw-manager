# HW5.py — RAG chatbot with short-term memory, multi-LLM, and robust Chroma handling

# ── SQLite shim (Chroma requires sqlite>=3.35; this swaps in pysqlite3 if available)
try:
    import pysqlite3
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass  # If not available, Chroma may error later on old sqlite

import os, glob, shutil, textwrap
from typing import List, Tuple, Dict

import streamlit as st

# Optional provider SDKs (we check presence at runtime)
try:
    from openai import OpenAI, BadRequestError
except Exception:
    OpenAI, BadRequestError = None, Exception

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import anthropic
except Exception:
    anthropic = None

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.config import Settings
from chromadb.errors import InternalError
from pypdf import PdfReader


# ─────────────────────────────────────────
# App Defaults
# ─────────────────────────────────────────
DEFAULT_PDF_DIR   = "lab4_pdfs"
DEFAULT_DB_PATH   = ".chroma_hw5"
DEFAULT_COLL_NAME = "HW5Collection"

TOP_K = 3
PAGE_CHUNK_SIZE = 2000
CHARS_PER_DOC = 1200
BATCH_SIZE = 64
MEMORY_TURNS = 8  # keep last N turns total

SYSTEM_PROMPT = (
    "You are a helpful Course Information Assistant.\n"
    "Use retrieved course context when possible; if none, say so clearly, then answer generally.\n"
    "Be concise; cite doc filenames and page numbers if you used them."
)

GPT_MODELS = ["gpt-5-nano", "gpt-5-chat-latest"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
CLAUDE_MODELS = [
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-20250514",
    "claude-3-5-haiku-20241022",
]


# ─────────────────────────────────────────
# Chroma helpers (with auto-heal for tenant migration)
# ─────────────────────────────────────────
def _embedding_fn(openai_api_key: str):
    # We use OpenAI embeddings for Chroma regardless of chat provider
    return OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")

def _client(db_path: str) -> chromadb.Client:
    os.makedirs(db_path, exist_ok=True)
    try:
        return chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
    except InternalError as e:
        # Auto-heal common migration issue: "no such table: tenants"
        if "no such table: tenants" in str(e).lower():
            shutil.rmtree(db_path, ignore_errors=True)
            os.makedirs(db_path, exist_ok=True)
            return chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
        raise

def _get_or_create_collection(db_path: str, coll_name: str, openai_api_key: str):
    client = _client(db_path)
    emb = _embedding_fn(openai_api_key)
    return client.get_or_create_collection(
        name=coll_name,
        embedding_function=emb,
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────
# PDF → chunks
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# Build collection
# ─────────────────────────────────────────
def build_vector_db(pdf_dir: str, db_path: str, coll_name: str, openai_api_key: str) -> int:
    collection = _get_or_create_collection(db_path, coll_name, openai_api_key)
    try:
        existing = collection.count() or 0
        if existing > 0:
            return existing
    except Exception:
        pass

    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdfs:
        st.error(f"No PDFs found in '{pdf_dir}'. Put your course PDFs there or change the path in sidebar.")
        st.stop()

    texts, metas, ids = [], [], []
    for path in pdfs:
        for (chunk_text, md) in _pdf_to_chunks_with_meta(path):
            ids.append(f"{md['filename']}|p{md['page']}|c{md['chunk']}")
            texts.append(chunk_text)
            metas.append(md)

    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        j = min(i + BATCH_SIZE, total)
        _get_or_create_collection(db_path, coll_name, openai_api_key).add(
            ids=ids[i:j], documents=texts[i:j], metadatas=metas[i:j]
        )
    return _get_or_create_collection(db_path, coll_name, openai_api_key).count()


# ─────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────
def retrieve_context(query: str, db_path: str, coll_name: str, openai_api_key: str, n_results: int = TOP_K):
    collection = _get_or_create_collection(db_path, coll_name, openai_api_key)
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


# ─────────────────────────────────────────
# Multi-LLM unified chat wrapper
# ─────────────────────────────────────────
def chat_with_model(provider: str, model: str, messages: list) -> str:
    if provider == "OpenAI":
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed.")
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content

    elif provider == "Gemini":
        if genai is None:
            raise RuntimeError("google-generativeai SDK not installed.")
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")
        genai.configure(api_key=api_key)

        # Convert OpenAI-style messages to Gemini request
        sys_prompt = ""
        convo = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                sys_prompt += content.strip() + "\n\n"
            elif role == "user":
                convo.append({"role": "user", "parts": [content]})
            else:
                convo.append({"role": "model", "parts": [content]})
        if sys_prompt:
            if convo and convo[0]["role"] == "user":
                convo[0]["parts"] = [sys_prompt + convo[0]["parts"][0]]
            else:
                convo.insert(0, {"role": "user", "parts": [sys_prompt]})

        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(convo, safety_settings=None)
        return getattr(resp, "text", "").strip()

    elif provider == "Anthropic":
        if anthropic is None:
            raise RuntimeError("anthropic SDK not installed.")
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is missing.")
        client = anthropic.Anthropic(api_key=api_key)

        # Split out system
        system_text = ""
        converted = []
        for m in messages:
            if m["role"] == "system":
                system_text += (m["content"] or "") + "\n"
            elif m["role"] in ("user", "assistant"):
                converted.append({"role": m["role"], "content": m["content"]})
        resp = client.messages.create(model=model, max_tokens=1024, system=system_text.strip(), messages=converted)
        parts = resp.content or []
        for p in parts:
            if hasattr(p, "text"):
                return p.text
            if isinstance(p, dict) and p.get("type") == "text":
                return p.get("text", "")
        return ""

    else:
        raise RuntimeError(f"Unknown provider: {provider}")


# ─────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────
def build_messages(user_query: str, context_block: str, history: list):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, text in history:
        msgs.append({"role": role, "content": text})
    if context_block:
        user = (
            f"User question:\n{user_query}\n\n"
            "Use ONLY the following retrieved course material to answer. "
            "Cite filenames and page numbers inline.\n"
            "===== CONTEXT START =====\n"
            f"{context_block}\n"
            "===== CONTEXT END =====\n"
        )
    else:
        user = (
            f"User question:\n{user_query}\n\n"
            "No course context retrieved. Answer generally and state that no PDFs were used."
        )
    msgs.append({"role": "user", "content": user})
    return msgs


# ─────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────
def run():
    st.title("🗒️ HW5: Intelligent Chatbot with Short-Term Memory (Multi-LLM)")

    with st.sidebar:
        st.header("RAG Settings")
        pdf_dir = st.text_input("PDF folder", value=DEFAULT_PDF_DIR)
        db_path = st.text_input("Chroma path", value=DEFAULT_DB_PATH)
        coll_name = st.text_input("Collection name", value=DEFAULT_COLL_NAME)
        force_rebuild = st.checkbox("Force rebuild from PDFs", value=False)

        # Quick DB reset button (helps recover from corruption / version mismatch)
        if st.button("🧹 Reset Chroma DB (delete & rebuild)"):
            shutil.rmtree(db_path, ignore_errors=True)
            st.success("Deleted DB folder. Toggle 'Force rebuild from PDFs' and ask again.")

        st.header("Model Provider")
        provider = st.radio("Provider", ["OpenAI", "Gemini", "Anthropic"], index=0)
        if provider == "OpenAI":
            model = st.selectbox("Model", GPT_MODELS, index=0)
            if not st.secrets.get("OPENAI_API_KEY"):
                st.warning("Needs OPENAI_API_KEY")
        elif provider == "Gemini":
            model = st.selectbox("Model", GEMINI_MODELS, index=0)
            if not st.secrets.get("GEMINI_API_KEY"):
                st.warning("Needs GEMINI_API_KEY")
        else:
            model = st.selectbox("Model", CLAUDE_MODELS, index=1)
            if not st.secrets.get("ANTHROPIC_API_KEY"):
                st.warning("Needs ANTHROPIC_API_KEY")

        st.divider()
        if st.button("Diagnostics"):
            try:
                coll = _get_or_create_collection(db_path, coll_name, st.secrets.get("OPENAI_API_KEY", ""))
                count = coll.count() or 0
                st.info(f"Collection `{coll_name}` has {count} vectors.")
                if count == 0:
                    st.warning("No vectors: build from PDFs.")
            except Exception as e:
                st.error(f"Collection error: {e}")

        # Show embeddings key warning only if missing (used for retrieval)
        if not st.secrets.get("OPENAI_API_KEY"):
            st.info("OpenAI embeddings are required for retrieval (set OPENAI_API_KEY).")

        # Optional: Secrets status expander (remove if you don’t want it)
        with st.expander("🔐 Secrets status", expanded=False):
            st.write("OPENAI_API_KEY:", bool(st.secrets.get("OPENAI_API_KEY")))
            st.write("GEMINI_API_KEY:", bool(st.secrets.get("GEMINI_API_KEY")))
            st.write("ANTHROPIC_API_KEY:", bool(st.secrets.get("ANTHROPIC_API_KEY")))

    # Ensure embeddings key exists for Chroma (stop early if missing)
    openai_embed_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_embed_key:
        st.stop()

    # Build / rebuild vector DB if needed
    coll = _get_or_create_collection(db_path, coll_name, openai_embed_key)
    need_build = force_rebuild or (coll.count() or 0) == 0
    if need_build:
        with st.status("Building vector DB from PDFs…", expanded=True) as status:
            count = build_vector_db(pdf_dir, db_path, coll_name, openai_embed_key)
            status.update(label=f"Vector DB ready with {count} chunks.", state="complete")
    else:
        st.caption(f"Vector DB OK — `{coll_name}` has {coll.count()} chunks.")

    # Short-term memory
    if "hw5_history" not in st.session_state:
        st.session_state.hw5_history = []  # list[(role, content)]

    # Render prior turns
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
    used_labels, context_block = retrieve_context(user_query, db_path, coll_name, openai_embed_key, n_results=TOP_K)

    # Build messages & call selected model
    messages = build_messages(user_query, context_block, st.session_state.hw5_history)

    with st.chat_message("assistant"):
        if used_labels:
            st.markdown("**Using course docs (RAG):** " + ", ".join(used_labels))
        else:
            st.markdown("**No relevant course docs found** — answering generally.")

        try:
            answer = chat_with_model(provider, model, messages)
            st.markdown(answer or "_(Empty response)_")
        except Exception as e:
            st.error(f"{provider} error: {e}")
            return

    # Save assistant reply & trim memory window
    st.session_state.hw5_history.append(("assistant", answer))
    st.session_state.hw5_history = st.session_state.hw5_history[-MEMORY_TURNS:]

if __name__ == "__main__":
    run()
