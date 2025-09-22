# HWs/HW4.py
# iSchool Student Orgs RAG Chatbot (HTML-based, vector DB built once)
# Providers: OpenAI, Anthropic (Claude), Google (Gemini)
# Storage: simple persisted pickle vector DB (embeddings + metadata)
# Exactly TWO chunks per HTML document; chunking method explained in comments.

import os
import pickle
from collections import deque
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import streamlit as st
from bs4 import BeautifulSoup

# =========================
# Page / App Config
# =========================
st.set_page_config(page_title="HW4 iSchool Orgs RAG", page_icon="🎓", layout="wide")

# ---- Your corpus folder ----
HTML_DIR = Path("hw4_htmls")                # <-- your folder with the HTML files
DB_PATH  = Path("data/ischool_vecdb.pkl")   # persisted vector db file
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Embeddings: OpenAI
EMBED_MODEL = "text-embedding-3-small"      # 1536-dim; cost-effective & solid
TOP_K = 4                                   # retrieved chunks per query
MEMORY_TURNS = 5                            # keep last 5 Q&A turns

# =========================
# Model menus (exactly as requested; no silent fallbacks)
# =========================
OPENAI_MODELS = [
    "gpt-5-nano",
    "gpt-5-chat-latest",
    # keep older ones around if you want them available
    "gpt-4o"
]

ANTHROPIC_MODELS = [
    "Opus 4-1-2025-08-25",
    "Opus 4-2025-05-14",
    "Sonnet 4-2025-05-14",
    "3-Sonnet-2025",
    "3-5-haiku",
    "3-haiku-2024-03-07",
]

GOOGLE_MODELS = [
    "Gemini-2.5-pro",
    "Gemini-2.5-flash",
    "Gemini-2.5-flash-lite",   
]

SYSTEM_PROMPT = (
    "You are a helpful iSchool assistant. Answer ONLY using the retrieved context from the HTML corpus "
    "about student organizations. If the answer is not in the context, say you don't know. "
    "Prefer specific club names, meeting times, locations, eligibility rules, and links if present."
)

# =========================
# Lazy API clients
# =========================
_openai_client = None
_anthropic_client = None
_gemini_model_cache: Dict[str, Any] = {}

def ensure_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
    return _openai_client

def ensure_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY", ""))
    return _anthropic_client

def ensure_gemini(model_name: str):
    if model_name not in _gemini_model_cache:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY", ""))
        _gemini_model_cache[model_name] = genai.GenerativeModel(model_name)
    return _gemini_model_cache[model_name]

# =========================
# HTML → text → two chunks
# =========================
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

def two_chunk_split(text: str) -> List[str]:
    """
    EXACTLY TWO chunks per document (assignment requirement).

    Method: balanced sentence-halving.
    - Split text into rough sentences by '.', '!', '?', and line breaks.
    - Concatenate sentences into two halves with ~equal character count (preserve order).

    Why this method?
    - Keeps local cohesion without over-fragmentation.
    - Simple, deterministic, and effective for short/medium HTML pages typical of club info.
    """
    text = text.strip()
    if not text:
        return ["", ""]
    rough = []
    for line in text.split("\n"):
        start = 0
        for i, ch in enumerate(line):
            if ch in ".!?":
                seg = line[start:i+1].strip()
                if seg:
                    rough.append(seg)
                start = i+1
        tail = line[start:].strip()
        if tail:
            rough.append(tail)
    if not rough:
        rough = [text]

    total = sum(len(s) for s in rough)
    target = total // 2
    left, right, acc = [], [], 0
    for s in rough:
        if acc < target:
            left.append(s); acc += len(s)
        else:
            right.append(s)
    return [" ".join(left).strip(), " ".join(right).strip()]

# =========================
# Embeddings + simple vector DB
# =========================
def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    client = ensure_openai()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32); b = b.astype(np.float32)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T

def build_db_once(html_dir: Path, db_path: Path):
    if db_path.exists():
        return
    files = sorted([p for p in html_dir.glob("**/*.html") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No HTML files found in: {html_dir.resolve()}")

    chunks, metas = [], []
    for p in files:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        txt = html_to_text(raw)
        c1, c2 = two_chunk_split(txt)
        for idx, c in enumerate([c1, c2], start=1):
            chunks.append(c)
            metas.append({"source": str(p), "part": idx})

    embs = embed_texts(chunks)
    payload = {"embeddings": embs, "chunks": chunks, "metas": metas}
    with open(db_path, "wb") as f:
        pickle.dump(payload, f)

def retrieve(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if not DB_PATH.exists():
        raise RuntimeError("Vector DB not found. Click 'Build Vector DB' in the sidebar after placing HTMLs.")
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
    q = embed_texts([query])
    sims = cosine_sim(q, db["embeddings"])[0]
    idxs = np.argsort(-sims)[:k]
    out = []
    for i in idxs:
        out.append({
            "score": float(sims[i]),
            "chunk": db["chunks"][i],
            "meta": db["metas"][i],
        })
    return out

def format_context(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for r in hits:
        src = Path(r["meta"]["source"]).name
        parts.append(f"[{src} • part {r['meta']['part']} • score {r['score']:.3f}]\n{r['chunk']}")
    return "\n\n---\n\n".join(parts)

# =========================
# LLM calls (no hidden fallbacks)
# =========================
def call_openai(model: str, sys_prompt: str, history: List[Dict[str, str]]) -> str:
    client = ensure_openai()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys_prompt}] + history,
        temperature=0.2,
    )
    return resp.choices[0].message.content

def call_anthropic(model: str, sys_prompt: str, history: List[Dict[str, str]]) -> str:
    client = ensure_anthropic()
    # Flatten to Anthropic format; pass system via 'system'
    msgs = []
    for m in history:
        role = m["role"]
        if role == "user":
            msgs.append({"role":"user", "content": m["content"]})
        else:
            msgs.append({"role":"assistant", "content": m["content"]})
    resp = client.messages.create(
        model=model, system=sys_prompt, temperature=0.2, max_tokens=1200, messages=msgs
    )
    # Concatenate text blocks
    return "".join([b.text for b in resp.content if hasattr(b, "text")])

def call_gemini(model: str, sys_prompt: str, history: List[Dict[str, str]]) -> str:
    mdl = ensure_gemini(model)
    # Compose a plain prompt with roles + system header (Gemini style)
    lines = [f"System:\n{sys_prompt}\n"]
    for m in history:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}:\n{m['content']}\n")
    resp = mdl.generate_content("\n".join(lines), generation_config={"temperature": 0.2})
    return resp.text or ""

# =========================
# Session state
# =========================
def init_state():
    if "chat" not in st.session_state:
        st.session_state.chat = deque(maxlen=MEMORY_TURNS)  # (user, assistant, model_tag)
    if "provider" not in st.session_state:
        st.session_state.provider = "OpenAI"
init_state()

# =========================
# UI
# =========================
with st.sidebar:
    st.header("⚙️ Setup")
    st.write("1) Put your iSchool org **.html** files in `hw4_htmls/`.\n2) Build the vector DB once.")
    if st.button("Build Vector DB"):
        try:
            build_db_once(HTML_DIR, DB_PATH)
            st.success(f"Vector DB ready: {DB_PATH}")
        except Exception as e:
            st.error(f"Build failed: {e}")

    st.divider()
    st.header("🤖 Provider & Model")
    provider = st.selectbox("Provider", ["OpenAI", "Anthropic", "Google"], index=["OpenAI","Anthropic","Google"].index(st.session_state.provider))
    st.session_state.provider = provider

    if provider == "OpenAI":
        model = st.selectbox("Model", OPENAI_MODELS, index=0)
        if not st.secrets.get("OPENAI_API_KEY"):
            st.warning("Add OPENAI_API_KEY to secrets.")
    elif provider == "Anthropic":
        model = st.selectbox("Model", ANTHROPIC_MODELS, index=0)
        if not st.secrets.get("ANTHROPIC_API_KEY"):
            st.warning("Add ANTHROPIC_API_KEY to secrets.")
    else:
        model = st.selectbox("Model", GOOGLE_MODELS, index=0)
        if not st.secrets.get("GOOGLE_API_KEY"):
            st.warning("Add GOOGLE_API_KEY to secrets.")

    st.divider()
    top_k = st.slider("Retrieved chunks (k)", 2, 6, TOP_K, 1)
    st.caption("Answers will cite only from retrieved context.")

st.title("🎓 HW4: iSchool Student Orgs RAG Chatbot")
st.write("Ask any question about iSchool student organizations. The bot retrieves from your HTML corpus and answers using that context only.")

# Show history (last 5 turns)
for u, a, tag in list(st.session_state.chat):
    with st.chat_message("user"):
        st.markdown(u)
    with st.chat_message("assistant"):
        st.markdown(f"_{tag}_\n\n{a}")

user_q = st.chat_input("Type your question…")
if user_q:
    # Retrieve context
    try:
        hits = retrieve(user_q, k=top_k)
        context = format_context(hits)
    except Exception as e:
        hits, context = [], ""
        st.error(f"Retrieval error: {e}")

    # Build conversation messages (current-turn context + user question)
    history: List[Dict[str, str]] = []
    if context:
        history.append({"role":"assistant", "content": "Relevant context:\n\n" + context})
    history.append({"role":"user", "content": user_q})

    # Call chosen model (no hidden fallback)
    with st.chat_message("assistant"):
        try:
            if provider == "OpenAI":
                ans = call_openai(model, SYSTEM_PROMPT, history)
            elif provider == "Anthropic":
                ans = call_anthropic(model, SYSTEM_PROMPT, history)
            else:
                ans = call_gemini(model, SYSTEM_PROMPT, history)
        except Exception as e:
            ans = f"⚠️ Model error: {e}"

        # Show quick source previews
        if hits:
            previews = []
            for i, h in enumerate(hits, start=1):
                src = Path(h["meta"]["source"]).name
                part = h["meta"]["part"]
                snippet = (h["chunk"] or "").strip().replace("\n", " ")
                if len(snippet) > 340:
                    snippet = snippet[:340].rstrip() + "…"
                previews.append(f"**Doc {i}: {src} (part {part})** — {snippet}")
            st.markdown("\n\n**Retrieved snippets:**\n\n" + "\n\n".join(previews))

        st.markdown(f"_{provider} · {model}_\n\n{ans}")

    # Append to memory buffer (cap to last 5 turns)
    st.session_state.chat.append((user_q, ans, f"{provider} · {model}"))
