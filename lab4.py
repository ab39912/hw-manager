# ─────────────────────────────────────────────────────────────────────────────
# Polyfill for Streamlit programmatic pages (so we don't have to edit streamlit_app.py)
# If your Streamlit already supports st.Page/st.navigation, this does nothing.
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st

if not hasattr(st, "Page") or not hasattr(st, "navigation"):
    class _Page:
        def __init__(self, func, title=None, url_path=None):
            self._func = func
            self.title = title
            self.url_path = url_path
        def run(self):
            self._func()
    class _Navigation:
        def __init__(self, groups_dict):
            self._labels, self._pages = [], []
            for _, pages in groups_dict.items():
                for p in pages:
                    self._labels.append(p.title or "Untitled")
                    self._pages.append(p)
        def run(self):
            choice = st.sidebar.radio("Pages", self._labels, index=0)
            self._pages[self._labels.index(choice)].run()
    def _nav(groups_dict):
        return _Navigation(groups_dict)
    if not hasattr(st, "Page"):
        st.Page = _Page       # type: ignore
    if not hasattr(st, "navigation"):
        st.navigation = _nav  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# HW4: iSchool Student Orgs RAG Chatbot (HTML corpus, in-memory vector DB)
# No files are created; index lives only in memory for this app session.
# Providers: OpenAI, Anthropic (Claude), Google (Gemini)
# ─────────────────────────────────────────────────────────────────────────────
import os
from collections import deque
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from bs4 import BeautifulSoup

st.set_page_config(page_title="HW4 • iSchool Orgs RAG", page_icon="🎓", layout="wide")

# ── Paths / corpus
HTML_DIR = Path("hw4_htmls")                # your folder with the HTML files

# ── Embedding config
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5
MEMORY_TURNS = 5
PROVIDERS = ["OpenAI", "Anthropic", "Google"]

# ── Model menus shown to the user (labels)
OPENAI_MODELS = [
    "gpt-5-nano",
    "gpt-5-chat-latest",
    "gpt-4o",
    "gpt-4o-mini",
]

# Your requested Claude labels (UI) …
ANTHROPIC_MODELS = [
    "Opus 4.1 (2025-08-05)",
    "Opus 4 (2025-05-14)",
    "Sonnet 4 (2025-05-14)",
    "Sonnet 3.7 (2025-02-19)",
    "Haiku 3.5 (2024-10-22)",
    "Haiku 3 (2024-03-07)",
]

# …mapped to **official Anthropic API IDs**
ANTHROPIC_MODEL_ID = {
    "Opus 4.1 (2025-08-05)":   "claude-opus-4-1-20250805",
    "Opus 4 (2025-05-14)":     "claude-opus-4-20250514",
    "Sonnet 4 (2025-05-14)":   "claude-sonnet-4-20250514",
    "Sonnet 3.7 (2025-02-19)": "claude-3-7-sonnet-20250219",
    "Haiku 3.5 (2024-10-22)":  "claude-3-5-haiku-20241022",
    "Haiku 3 (2024-03-07)":    "claude-3-haiku-20240307",
}

# Google models (typo fixed: flash-lite)
GOOGLE_MODELS = [
    "Gemini-2.5-pro",
    "Gemini-2.5-flash",
    "Gemini-2.5-flash-lite",
]
# label → API id mapping (lowercase)
GOOGLE_MODEL_ID = {
    "Gemini-2.5-pro":        "gemini-2.5-pro",
    "Gemini-2.5-flash":      "gemini-2.5-flash",
    "Gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
}

SYSTEM_PROMPT = (
    "You are a helpful iSchool assistant. Use ONLY the retrieved HTML context about student organizations. "
    "If the context fully lacks the answer, say you don't know. If it is partially relevant, answer with what IS "
    "known and clearly state what is unknown. Prefer specific club names, meeting times, locations, eligibility "
    "rules, and links if present."
)

# ─────────────────────────────────────────────────────────────────────────────
# Secrets/env helper (accepts alternate names and falls back to env vars)
# ─────────────────────────────────────────────────────────────────────────────
def get_secret(*names: str) -> str:
    try:
        secrets = getattr(st, "secrets", {})
        for n in names:
            v = secrets.get(n)
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass
    for n in names:
        v = os.environ.get(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# Lazy clients for providers
# ─────────────────────────────────────────────────────────────────────────────
_openai_client = None
_anthropic_client = None
_gemini_model_cache: Dict[str, Any] = {}

def ensure_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=get_secret("OPENAI_API_KEY", "OPENAI_KEY"))
    return _openai_client

def ensure_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY", "ANTHROPIC_KEY"))
    return _anthropic_client

def _google_api_id(label: str) -> str:
    return GOOGLE_MODEL_ID.get(label, label.lower())

def _anthropic_api_id(label: str) -> str:
    return ANTHROPIC_MODEL_ID.get(label, label)

def ensure_gemini(model_label: str):
    api_id = _google_api_id(model_label)
    if api_id not in _gemini_model_cache:
        import google.generativeai as genai
        genai.configure(api_key=get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APIKEY", "GOOGLE_KEY"))
        _gemini_model_cache[api_id] = genai.GenerativeModel(api_id)
    return _gemini_model_cache[api_id]

# ─────────────────────────────────────────────────────────────────────────────
# HTML → text → exactly two chunks (assignment requirement)
# ─────────────────────────────────────────────────────────────────────────────
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

def two_chunk_split(text: str) -> List[str]:
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

# ─────────────────────────────────────────────────────────────────────────────
# In-memory vector DB (no disk writes)
# ─────────────────────────────────────────────────────────────────────────────
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

def build_vecdb_in_memory(html_dir: Path) -> Dict[str, Any]:
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
    return {"embeddings": embs, "chunks": chunks, "metas": metas}

def ensure_vecdb():
    if "hw4_vecdb" not in st.session_state:
        with st.spinner("Indexing HTML corpus (one-time per session)…"):
            st.session_state.hw4_vecdb = build_vecdb_in_memory(HTML_DIR)

def retrieve(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if "hw4_vecdb" not in st.session_state:
        raise RuntimeError("Vector DB not ready.")
    db = st.session_state.hw4_vecdb
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

# ─────────────────────────────────────────────────────────────────────────────
# Provider calls (no forced temperature; robust Anthropic mapping)
# ─────────────────────────────────────────────────────────────────────────────
def call_openai(model: str, sys_prompt: str, history: List[Dict[str, str]]) -> str:
    client = ensure_openai()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt}] + history,
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt}] + history,
            temperature=1,
        )
    return resp.choices[0].message.content

def call_anthropic(model_label: str, sys_prompt: str, history: List[Dict[str, str]]) -> str:
    client = ensure_anthropic()
    api_id = _anthropic_api_id(model_label)
    msgs = [{"role": ("user" if m["role"] == "user" else "assistant"), "content": m["content"]} for m in history]
    try:
        resp = client.messages.create(model=api_id, system=sys_prompt, max_tokens=1200, messages=msgs)
        return "".join([b.text for b in resp.content if hasattr(b, "text")])
    except Exception as e:
        return f"Model error (Anthropic): {e}  [api id tried: {api_id}]"

def call_gemini(model_label: str, sys_prompt: str, history: List[Dict[str, str]]) -> str:
    mdl = ensure_gemini(model_label)
    lines = [f"System:\n{sys_prompt}\n"]
    for m in history:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}:\n{m['content']}\n")
    resp = mdl.generate_content("\n".join(lines))  # defaults (no temp override)
    return (getattr(resp, "text", None) or "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# Session / UI
# ─────────────────────────────────────────────────────────────────────────────
def run():
    if "chat" not in st.session_state:
        st.session_state.chat = deque(maxlen=MEMORY_TURNS)
    if "provider" not in st.session_state:
        st.session_state.provider = "OpenAI"

    ensure_vecdb()

    st.title("🎓 HW4: iSchool Student Orgs RAG Chatbot")
    st.caption("Answers are grounded strictly in your `hw4_htmls/` corpus (exactly two chunks per HTML). No files are written.")

    with st.sidebar:
        st.header("🤖 Provider & Model")
        provider = st.selectbox("Provider", PROVIDERS, index=PROVIDERS.index(st.session_state.provider))
        st.session_state.provider = provider

        if provider == "OpenAI":
            model = st.selectbox("Model", OPENAI_MODELS, index=0)
            if not get_secret("OPENAI_API_KEY", "OPENAI_KEY"):
                st.warning("Missing OPENAI_API_KEY (or OPENAI_KEY).")
        elif provider == "Anthropic":
            model = st.selectbox("Model", ANTHROPIC_MODELS, index=0)
            if not get_secret("ANTHROPIC_API_KEY", "ANTHROPIC_KEY"):
                st.warning("Missing ANTHROPIC_API_KEY (or ANTHROPIC_KEY).")
            st.caption(f"API id → `{_anthropic_api_id(model)}`")
        else:
            model = st.selectbox("Model", GOOGLE_MODELS, index=0)
            if not get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APIKEY", "GOOGLE_KEY"):
                st.warning("Missing GEMINI_API_KEY (or GOOGLE_API_KEY).")
            st.caption(f"API id → `{_google_api_id(model)}`")

        st.divider()
        top_k = TOP_K
        st.caption("The assistant answers using your HTML corpus.")

    # Render past turns
    for u, a, tag in list(st.session_state.chat):
        with st.chat_message("user"):
            st.markdown(u)
        with st.chat_message("assistant"):
            st.markdown(f"_{tag}_\n\n{a}")

    # Current turn
    user_q = st.chat_input("Type your question…")
    if not user_q:
        return
    with st.chat_message("user"):
        st.markdown(user_q)

    # Retrieve & answer
    try:
        hits = retrieve(user_q, k=top_k)
        context = format_context(hits)
    except Exception as e:
        hits, context = [], ""
        st.error(f"Retrieval error: {e}")

    history: List[Dict[str, str]] = []
    if context:
        history.append({"role":"assistant", "content": "Relevant context:\n\n" + context})
    history.append({"role":"user", "content": user_q})

    with st.chat_message("assistant"):
        try:
            if st.session_state.provider == "OpenAI":
                ans = call_openai(model, SYSTEM_PROMPT, history)
            elif st.session_state.provider == "Anthropic":
                ans = call_anthropic(model, SYSTEM_PROMPT, history)
            else:
                ans = call_gemini(model, SYSTEM_PROMPT, history)
        except Exception as e:
            ans = f"⚠️ Model error: {e}"

        # Show sources preview
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

        st.markdown(f"_{st.session_state.provider} · {model}_\n\n{ans}")

    st.session_state.chat.append((user_q, ans, f"{st.session_state.provider} · {model}"))

if __name__ == "__main__":
    run()
