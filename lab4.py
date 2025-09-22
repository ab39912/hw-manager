# lab4.py

# ============================================================
# SQLite shim: make sure sqlite3 >= 3.35 for Chroma without OS upgrades
# Requires 'pysqlite3-binary' in requirements.txt.
# ============================================================
try:
    import pysqlite3  # modern SQLite packaged wheel
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
import glob
import re
import textwrap
from typing import List, Tuple, Dict

import streamlit as st

# ===== LLM SDKs =====
from openai import OpenAI, BadRequestError
try:
    import anthropic  # Claude
except Exception:
    anthropic = None
try:
    import google.generativeai as genai  # Gemini
except Exception:
    genai = None

# ===== Vector DB =====
try:
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
except Exception:
    chromadb = None
    OpenAIEmbeddingFunction = None

# ===== HTML parsing =====
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


# ======================
# Configuration
# ======================
# Provider + model choices
OPENAI_MODELS = [
    ("gpt-5-nano",  "gpt-5-nano"),
    ("chat-latest", "gpt-5-chat-latest"),
    ("gpt-4o",      "gpt-4o"),
]
CLAUDE_MODELS = [
    ("claude-4.1-opus",  "Claude-Opus 4.1"),
    ("claude-4-sonnet",  "Claude-Sonnet 4"),
    ("claude-3.5-haiku", "Claude-Haiku 3.5"),
]
GEMINI_MODELS = [
    ("gemini-2.5-pro",        "Gemini-2.5 Pro"),
    ("gemini-2.5-flash",      "Gemini-2.5 Flash"),
    ("gemini-2.5-flash-lite", "Gemini-2.5 Flash Lite"),
]

DEFAULT_PROVIDER = "OpenAI"
DEFAULT_MODEL_OPENAI = "gpt-5-nano"
DEFAULT_MODEL_CLAUDE = "claude-4.1-opus"
DEFAULT_MODEL_GEMINI = "gemini-2.5-pro"

# Optional compatibility map in case your Anthropic project doesn't have the new aliases yet
ANTHROPIC_COMPAT_MAP = {
    "claude-4.1-opus":  "claude-3-opus-latest",
    "claude-4-sonnet":  "claude-3-5-sonnet-latest",
    "claude-3.5-haiku": "claude-3-5-haiku-latest",
}

SYSTEM_PROMPT = (
    "You are a helpful Course Information Assistant.\n"
    "When context from course HTML pages is provided, rely on it first and be concise and clear.\n"
    "If the context seems unrelated or empty, say that you're answering generally.\n"
    "Always write in short, clear sentences."
)

# HTML folder (your repo shows 'hw4_htmls' at root)
DEFAULT_HTML_DIR = "hw4_htmls"

# Persistent Chroma on disk (so we only embed once)
CHROMA_PATH = ".chroma_hw4"
COLLECTION_NAME = "HW4Collection"

# Retrieval & memory
TOP_K = 3
SNIPPET_CHARS = 1000
MAX_QA_PAIRS = 5  # memory buffer: last 5 Q&A pairs


# ======================
# Utilities
# ======================
def _get_secret(name: str) -> str | None:
    val = None
    try:
        val = st.secrets[name]
    except Exception:
        pass
    if not val:
        val = os.getenv(name)
    return val

def _pick_bs_parser() -> str:
    """Prefer lxml, then html5lib, then built-in html.parser."""
    try:
        import lxml  # noqa
        return "lxml"
    except Exception:
        pass
    try:
        import html5lib  # noqa
        return "html5lib"
    except Exception:
        pass
    return "html.parser"


# ======================
# Streaming helpers (ONE write_stream per response)
# ======================
def stream_openai_once(client: OpenAI, prompt: str, model: str):
    """
    Generator that yields chunks from OpenAI.
    Call st.write_stream(stream_openai_once(...)) ONCE.
    """
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    stream = client.chat.completions.create(model=model, messages=msgs, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            yield delta


def stream_claude_once(anthropic_client, prompt: str, model: str):
    """Generator for Claude streaming. Call write_stream once."""
    try:
        with anthropic_client.messages.stream(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and getattr(event.delta, "text", None):
                    yield event.delta.text
    except Exception as e:
        yield f"\n\n[Claude error: {e}]"


def stream_gemini_once(prompt: str, model: str, google_api_key: str):
    """Generator for Gemini streaming. Call write_stream once."""
    if genai is None:
        yield "[Gemini SDK not installed]"
        return
    genai.configure(api_key=google_api_key)
    gmodel = genai.GenerativeModel(model_name=model)
    try:
        responses = gmodel.generate_content(prompt, stream=True)
        for r in responses:
            if hasattr(r, "text") and r.text:
                yield r.text
    except Exception as e:
        yield f"\n\n[Gemini error: {e}]"


# ======================
# Vector DB (Chroma)
# ======================
def _embedding_fn(openai_api_key: str):
    if OpenAIEmbeddingFunction is None:
        raise RuntimeError("Missing dependency: chromadb. Please install it.")
    return OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")

def _client():
    if chromadb is None:
        raise RuntimeError("Missing dependency: chromadb. Please install it.")
    return chromadb.PersistentClient(path=st.session_state.get("HW4_chroma_path", CHROMA_PATH))

def _get_or_create_collection(openai_api_key: str):
    client = _client()
    emb = _embedding_fn(openai_api_key)
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(
            name=st.session_state.get("HW4_collection_name", COLLECTION_NAME),
            embedding_function=emb,
            metadata={"hnsw:space": "cosine"},
        )
    try:
        return client.get_collection(
            name=st.session_state.get("HW4_collection_name", COLLECTION_NAME),
            embedding_function=emb,
        )
    except Exception:
        return client.create_collection(
            name=st.session_state.get("HW4_collection_name", COLLECTION_NAME),
            embedding_function=emb,
            metadata={"hnsw:space": "cosine"},
        )


# ======================
# HTML → EXACTLY TWO CHUNKS each
# ======================
def _html_file_to_text(path: str) -> str:
    """
    Extract readable text from one HTML file (drop scripts/styles/nav/etc).
    Uses a parser fallback: lxml → html5lib → html.parser.
    """
    if BeautifulSoup is None:
        raise RuntimeError("Missing dependency: beautifulsoup4. Please install it.")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read HTML '{path}': {e}")

    parser = _pick_bs_parser()
    soup = BeautifulSoup(html, parser)
    for t in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        t.extract()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _two_balanced_chunks(text: str) -> List[str]:
    """
    >>> CHUNKING STRATEGY (exactly TWO mini-docs per HTML file) <<<
    Split near the midpoint by character count, but cut at a *paragraph boundary*
    (double newline) closest to the middle so we avoid splitting sentences.

    WHY this method?
      - Deterministic & simple: consistently two chunks per doc.
      - Balanced: similar sizes help retrieval.
      - Paragraph-aware: cleaner snippets and better semantic coherence.

    Fallback: if no clear paragraphs or very short text, do a naive 50/50 split.
    """
    if not text:
        return []
    if text.count("\n\n") == 0 or len(text) < 400:
        mid = len(text) // 2
        return [text[:mid].strip(), text[mid:].strip()]

    paragraphs = text.split("\n\n")
    cum = 0
    mids = []
    for p in paragraphs:
        cum += len(p) + 2  # +2 accounts for the removed '\n\n'
        mids.append(cum)
    target = len(text) // 2
    cut_idx = min(range(len(mids)), key=lambda i: abs(mids[i] - target))
    cut_pos = mids[cut_idx]
    left, right = text[:cut_pos].strip(), text[cut_pos:].strip()
    if not left or not right:
        mid = len(text) // 2
        left, right = text[:mid].strip(), text[mid:].strip()
    return [left, right]


def _html_to_two_chunks_with_meta(path: str) -> List[Tuple[str, Dict]]:
    """Return exactly two (chunk_text, metadata) tuples for one HTML file."""
    filename = os.path.basename(path)
    text = _html_file_to_text(path)
    chunks = _two_balanced_chunks(text)
    out = []
    for i, chunk in enumerate(chunks, start=1):
        if not chunk:
            continue
        md = {
            "filename": filename,
            "source_path": path,
            "part": i,     # 1 or 2
            "chunking": "two_balanced_paragraph_aware",
        }
        out.append((chunk, md))
    return out


# ======================
# Build vector DB once (idempotent)
# ======================
def initialize_hw4_vector_db(html_dir: str, openai_api_key: str):
    """Create/populate the Chroma collection only if missing/empty."""
    if chromadb is None or OpenAIEmbeddingFunction is None:
        st.error("Missing dependency: chromadb.")
        st.stop()
    if BeautifulSoup is None:
        st.error("Missing dependency: beautifulsoup4.")
        st.stop()

    st.session_state.setdefault("HW4_chroma_path", CHROMA_PATH)
    st.session_state.setdefault("HW4_collection_name", COLLECTION_NAME)

    collection = _get_or_create_collection(openai_api_key)

    # Already populated?
    try:
        if (collection.count() or 0) > 0:
            st.session_state["HW4_vectorDB_ready"] = True
            return
    except Exception:
        try:
            probe = collection.query(query_texts=["probe"], n_results=1)
            if bool(probe.get("ids")):
                st.session_state["HW4_vectorDB_ready"] = True
                return
        except Exception:
            pass

    # First-time build
    html_paths = sorted(glob.glob(os.path.join(html_dir, "*.html")) + glob.glob(os.path.join(html_dir, "*.htm")))
    if not html_paths:
        st.error(f"No HTML files found in '{html_dir}'. Copy all supplied HTML pages there.")
        st.stop()

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []

    for path in html_paths:
        pairs = _html_to_two_chunks_with_meta(path)  # EXACTLY TWO per file
        if not pairs:
            st.warning(f"HTML '{os.path.basename(path)}' produced no text; skipping.")
            continue
        for (chunk_text, md) in pairs:
            cid = f"{md['filename']}|part{md['part']}"
            ids.append(cid)
            docs.append(chunk_text)
            metas.append(md)

    if not docs:
        st.error("All HTML files were empty or failed to parse; aborting vector DB creation.")
        st.stop()

    try:
        collection.add(ids=ids, documents=docs, metadatas=metas)
    except Exception as e:
        st.error(f"Failed creating the vector DB: {e}")
        st.stop()

    st.session_state["HW4_vectorDB_ready"] = True


# ======================
# Retrieval for RAG
# ======================
def retrieve_context(query: str, openai_api_key: str, html_dir: str, n_results: int = TOP_K):
    """Return (used_docs, context_block) for the LLM."""
    if not st.session_state.get("HW4_vectorDB_ready", False):
        initialize_hw4_vector_db(html_dir, openai_api_key)

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
        fname = md.get("filename", "unknown.html") if md else "unknown.html"
        part = md.get("part")
        snippet = (doc or "")[:SNIPPET_CHARS]
        used_docs.append({"filename": fname, "part": part, "snippet": snippet})

    if not used_docs:
        return [], ""

    blocks = []
    for i, d in enumerate(used_docs, start=1):
        label = f"{d['filename']}" + (f" (part {d['part']})" if d.get("part") else "")
        blocks.append(f"[Doc {i}: {label}]\n" + textwrap.dedent(d["snippet"]).strip())
    context_block = "\n\n---\n\n".join(blocks)
    return used_docs, context_block


# ======================
# Memory buffer (last 5 Q&A pairs)
# ======================
def get_buffered_history() -> List[Dict]:
    """Return up to the last 5 Q&A pairs (10 messages)."""
    full = st.session_state.get("messages", [])
    user_idxs = [i for i, m in enumerate(full) if m["role"] == "user"]
    keep_user = user_idxs[-MAX_QA_PAIRS:]
    kept = []
    for ui in keep_user:
        kept.append(full[ui])
        for j in range(ui + 1, len(full)):
            if full[j]["role"] == "assistant":
                kept.append(full[j])
                break
    return kept


def build_unified_prompt(user_query: str, context_block: str) -> str:
    """
    Build one prompt usable across providers. Contains:
      - SYSTEM intent
      - Short memory transcript (last 5 Q&A)
      - User question
      - Retrieved context (if any)
    """
    mem = get_buffered_history()
    mem_lines = []
    for m in mem:
        role = "User" if m["role"] == "user" else "Assistant"
        mem_lines.append(f"{role}: {m['content']}")
    memory_text = "\n".join(mem_lines).strip()

    parts = [f"SYSTEM:\n{SYSTEM_PROMPT}\n"]
    if memory_text:
        parts.append("SHORT MEMORY (last 5 Q&A):\n" + memory_text + "\n")
    parts.append("USER QUESTION:\n" + user_query + "\n")
    if context_block:
        parts.append(
            "COURSE CONTEXT (from HTML retrieval):\n"
            "Use this context first. If it doesn't contain the answer, say so and then answer generally.\n"
            "==== CONTEXT START ====\n"
            f"{context_block}\n"
            "==== CONTEXT END ====\n"
            "Cite the HTML file names (and part numbers) inline where relevant.\n"
        )
    return "\n".join(parts)


# ======================
# Main UI
# ======================
def run():
    st.title("🧠 Lab 4: HTML RAG Chatbot with Memory")
    st.caption("Persistent ChromaDB from HTML (two chunks per doc), short conversation memory, and provider/model switch.")

    # API keys
    openai_key = _get_secret("OPENAI_API_KEY")
    anthropic_key = _get_secret("ANTHROPIC_API_KEY")
    google_key = _get_secret("GOOGLE_API_KEY")

    # Sidebar controls
    with st.sidebar:
        st.header("Vector DB Setup")
        html_dir = st.text_input(
            "HTML folder",
            value=DEFAULT_HTML_DIR,
            help="Path containing your HTML pages (e.g., 'hw4_htmls').",
        )
        build_now = st.checkbox("Build on load if collection empty", value=True)

        st.divider()
        st.header("Provider & Model")

        provider = st.selectbox("Provider", ["OpenAI", "Claude", "Gemini"], index=["OpenAI","Claude","Gemini"].index(DEFAULT_PROVIDER))

        if provider == "OpenAI":
            model = st.selectbox(
                "OpenAI model",
                options=[m[0] for m in OPENAI_MODELS],
                format_func=lambda x: dict(OPENAI_MODELS)[x],
                index=[m[0] for m in OPENAI_MODELS].index(DEFAULT_MODEL_OPENAI),
            )
        elif provider == "Claude":
            model = st.selectbox(
                "Claude model",
                options=[m[0] for m in CLAUDE_MODELS],
                format_func=lambda x: dict(CLAUDE_MODELS)[x],
                index=[m[0] for m in CLAUDE_MODELS].index(DEFAULT_MODEL_CLAUDE),
            )
        else:  # Gemini
            model = st.selectbox(
                "Gemini model",
                options=[m[0] for m in GEMINI_MODELS],
                format_func=lambda x: dict(GEMINI_MODELS)[x],
                index=[m[0] for m in GEMINI_MODELS].index(DEFAULT_MODEL_GEMINI),
            )

        use_rag = st.checkbox("Use HTML vector DB (RAG)", value=True)
        st.caption("Memory buffer keeps last 5 Q&A pairs.")

    # Build vector DB once (needs OpenAI key for embeddings)
    if build_now and not st.session_state.get("HW4_vectorDB_ready", False):
        if not openai_key:
            st.warning("OpenAI key required to build embeddings for the vector DB. Add OPENAI_API_KEY.")
        else:
            initialize_hw4_vector_db(html_dir, openai_key)

    # Chat transcript state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render transcript
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a course-related question")
    if not user_input:
        return

    # Append & render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieval
    used_docs = []
    context_block = ""
    if use_rag and openai_key:
        used_docs, context_block = retrieve_context(user_input, openai_key, html_dir, n_results=TOP_K)
    elif use_rag and not openai_key:
        st.info("RAG enabled, but no OPENAI_API_KEY set for embeddings. Proceeding without RAG.")

    # Build unified prompt with memory + context
    prompt = build_unified_prompt(user_input, context_block)

    # Assistant reply (ONE write_stream call per provider)
    with st.chat_message("assistant"):
        header = f"**Provider:** `{provider}`  •  **Model:** `{model}` — "
        if use_rag and used_docs:
            labels = [d["filename"] + (f" (part {d['part']})" if d.get("part") else "") for d in used_docs]
            st.markdown(header + "**Using HTML docs (RAG):** " + ", ".join(labels))
        elif use_rag and not used_docs:
            st.markdown(header + "**No relevant HTML context found** — answering generally.")
        else:
            st.markdown(header + "**RAG disabled** — answering generally.")

        try:
            if provider == "OpenAI":
                if not openai_key:
                    st.error("Missing OPENAI_API_KEY.")
                    return
                oclient = OpenAI(api_key=openai_key)
                full_answer = st.write_stream(stream_openai_once(oclient, prompt, model))

            elif provider == "Claude":
                if anthropic is None:
                    st.error("anthropic package not installed.")
                    return
                if not anthropic_key:
                    st.error("Missing ANTHROPIC_API_KEY.")
                    return
                aclient = anthropic.Anthropic(api_key=anthropic_key)
                # Optional compatibility mapping:
                model_to_use = ANTHROPIC_COMPAT_MAP.get(model, model)
                full_answer = st.write_stream(stream_claude_once(aclient, prompt, model_to_use))

            else:  # Gemini
                if genai is None:
                    st.error("google-generativeai package not installed.")
                    return
                if not google_key:
                    st.error("Missing GOOGLE_API_KEY.")
                    return
                full_answer = st.write_stream(stream_gemini_once(prompt, model, google_key))

        except BadRequestError:
            st.error("Bad request to the provider API. Check model name, quota, or prompt size.")
            return

        # Show short previews inside the chat bubble (transparency)
        if use_rag and used_docs:
            previews = []
            for i, d in enumerate(used_docs, start=1):
                snippet = (d["snippet"] or "").strip().replace("\n", " ")
                if len(snippet) > 400:
                    snippet = snippet[:400].rstrip() + "…"
                label = d["filename"] + (f" (part {d['part']})" if d.get("part") else "")
                previews.append(f"**Doc {i}: {label}** — {snippet}")
            st.markdown("\n\n**Retrieved snippets:**\n\n" + "\n\n".join(previews))

    # Save assistant reply and trim memory to last 5 Q&A pairs
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    msgs = st.session_state.messages
    user_idxs = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    if len(user_idxs) > MAX_QA_PAIRS:
        cutoff_user_idx = user_idxs[-MAX_QA_PAIRS]
        st.session_state.messages = msgs[cutoff_user_idx:]


if __name__ == "__main__":
    run()
