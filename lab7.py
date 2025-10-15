# pages/HW7.py
# ──────────────────────────────────────────────────────────────────────────────
# HW7: News Information Bot (Streamlit + Multi-Vendor + ChromaDB + CSV RAG)
# Ranks "most interesting" news for a global law firm and answers questions.
# Expects CSV with columns: title, summary, content, published_date, source, url
# If columns differ, the app will adapt best-effort.
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
import re
import math
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# Vendor SDKs
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

# HTML parsing is not required here, but kept for parity with HW5 patterns
# from bs4 import BeautifulSoup

# ==============================================================================
# 0) SQLite shim (for older Linux images in Streamlit Cloud)
# ==============================================================================
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    print("⚠️  Falling back to system sqlite3 (may be outdated).")
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: F401

# ==============================================================================
# 1) Constants & Model Options
# ==============================================================================
PAGE_TITLE = "HW7: News Information Bot"
PAGE_ICON = "🗞️"

CHROMA_DB_PATH = "./ChromaDB_RAG"
CHROMA_COLLECTION_NAME = "HW7_NewsCollection"
DEFAULT_CSV_PATH = "hw7pdfs/Example_news_info_for_testing.csv"

MODEL_OPTIONS = {
    "OpenAI": ["gpt-5-nano", "gpt-5-mini", "gpt-5-chat-latest"],
    "Google": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
    "Anthropic": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620", "claude-sonnet-4-20250514"],
}

EMBED_MODEL = "text-embedding-3-small"

# ==============================================================================
# 2) Cached clients & Chroma collection
# ==============================================================================
@st.cache_resource
def get_api_clients():
    clients = {}
    if "OPENAI_API_KEY" in st.secrets:
        clients["OpenAI"] = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        clients["Google"] = genai
    if "ANTHROPIC_API_KEY" in st.secrets:
        clients["Anthropic"] = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    return clients


@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# ==============================================================================
# 3) Data loading & normalization
# ==============================================================================
EXPECTED_COLS = ["title", "summary", "content", "published_date", "source", "url"]

@st.cache_data
def load_news_data(file_bytes: bytes = None, filename: str = None) -> pd.DataFrame:
    if file_bytes:
        df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    else:
        if not os.path.exists(DEFAULT_CSV_PATH):
            st.error("No CSV found. Upload a CSV with news articles.")
            st.stop()
        df = pd.read_csv(DEFAULT_CSV_PATH)

    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols: return cols[c]
        return None

    title_c   = pick("title", "headline")
    summary_c = pick("summary", "abstract", "deck")
    content_c = pick("content", "body", "text", "article")
    date_c    = pick("published_date", "date", "pub_date")
    source_c  = pick("source", "publisher", "outlet")
    url_c     = pick("url", "link")

    if title_c is None:  df["title"] = ""
    else:                df.rename(columns={title_c: "title"}, inplace=True)
    if summary_c is None: df["summary"] = ""
    else:                 df.rename(columns={summary_c: "summary"}, inplace=True)
    if content_c is None: df["content"] = ""
    else:                 df.rename(columns={content_c: "content"}, inplace=True)
    if date_c is None: df["published_date"] = ""
    else:              df.rename(columns={date_c: "published_date"}, inplace=True)
    if source_c is None: df["source"] = ""
    else:                df.rename(columns={source_c: "source"}, inplace=True)
    if url_c is None: df["url"] = ""
    else:             df.rename(columns={url_c: "url"}, inplace=True)

    df = df.fillna("")
    df["text"] = (df["title"].astype(str) + " — " +
                  df["summary"].astype(str) + "\n" +
                  df["content"].astype(str))
    return df[["title", "summary", "content", "published_date", "source", "url", "text"]]

# ==============================================================================
# 4) Vector DB ingest & retrieval
# ==============================================================================
def setup_news_vector_db(collection, openai_client, df: pd.DataFrame, force_rebuild: bool = False):
    if (collection.count() or 0) > 0 and not force_rebuild:
        st.sidebar.success(f"Vector DB ready ({collection.count()} items).")
        return

    if df.empty:
        st.sidebar.error("No rows in CSV.")
        st.stop()

    st.sidebar.warning("Building Vector DB…")
    texts = df["text"].tolist()
    ids = df.index.astype(str).tolist()
    metas = df[["title", "source", "published_date", "url"]].to_dict("records")

    batch = 100
    for i in range(0, len(texts), batch):
        chunk_texts = texts[i:i+batch]
        chunk_ids   = ids[i:i+batch]
        chunk_meta  = metas[i:i+batch]
        try:
            resp = openai_client.embeddings.create(model=EMBED_MODEL, input=chunk_texts)
            embeddings = [d.embedding for d in resp.data]
            collection.add(ids=chunk_ids, documents=chunk_texts,
                           embeddings=embeddings, metadatas=chunk_meta)
        except Exception as e:
            st.error(f"Embedding batch {i}-{i+batch} failed: {e}")

    st.sidebar.success(f"Vector DB built ✅ ({collection.count()} items)")


def get_relevant_context(collection, openai_client, query: str, n_results: int = 4):
    try:
        st.write(f"🔍 DEBUG: Running query → {query}")
        count = collection.count()
        st.write(f"📊 DEBUG: Collection count = {count}")
        if count == 0:
            return "No items in vector DB. Please rebuild it.", []
        
        qemb = openai_client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
        res = collection.query(query_embeddings=[qemb], n_results=n_results)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        if not docs:
            return "No relevant context found.", []
        context_blocks, cites = [], []
        for d, m in zip(docs, metas):
            title = m.get("title", "").strip()
            src = m.get("source", "").strip()
            url = m.get("url", "").strip()
            date = m.get("published_date", "").strip()
            header = f"[{title}] ({src}, {date}) {url}".strip()
            cites.append({"title": title, "source": src, "date": date, "url": url})
            snippet = d[:1200]
            context_blocks.append(f"{header}\n{snippet}")
        return "\n\n---\n\n".join(context_blocks), cites
    except Exception as e:
        st.error(f"Vector search failed: {e}")
        return "Vector search error.", []

# ==============================================================================
# 5) Ranking
# ==============================================================================
LEGAL_RE = re.compile(
    r"(lawsuit|litigation|injunction|arbitration|class action|"
    r"regulation|regulatory|compliance|sanction|gdpr|ccpa|hipaa|"
    r"fine|penalty|settlement|consent decree|enforcement|"
    r"merger|acquisition|m&a|antitrust|sec|doj|ftc|cma|eu|ofac|"
    r"ip infringement|patent|trademark)", flags=re.IGNORECASE)
MONEY_RE = re.compile(r"(\$|USD|€|EUR|£|GBP)\s?\d+([.,]\d{3})*(\s?(million|billion|m|bn))?", re.IGNORECASE)

def safe_days_since(datestr: str) -> float:
    try:
        dt_val = pd.to_datetime(datestr, errors="coerce")
        if pd.isna(dt_val): return 60.0
        return max(0.0, (pd.Timestamp.now(tz=None) - dt_val).days)
    except Exception:
        return 60.0

def compute_interest_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.assign(interest_score=0.0)
    df = df.copy()
    df["recency_days"] = df["published_date"].astype(str).map(safe_days_since)
    df["legal_hits"] = df["text"].str.count(LEGAL_RE)
    df["money_flag"] = df["text"].str.contains(MONEY_RE).astype(int)
    recency = np.exp(-df["recency_days"] / 14.0) * 25
    legal   = np.minimum(df["legal_hits"], 5) / 5.0 * 25
    scale   = df["money_flag"] * 15
    source  = 10.0
    uniq    = 5.0
    df["interest_score"] = recency + legal + scale + source + uniq
    return df.sort_values("interest_score", ascending=False)

# ==============================================================================
# 6) LLM response
# ==============================================================================
def llm_answer(clients, provider: str, model: str, query: str, context: str) -> str:
    system_prompt = (
        "You are a legal-industry news analyst for a global law firm. "
        "Use ONLY the provided CONTEXT. If unsure, say you don't know. "
        "Highlight legal impact, agencies, jurisdictions, monetary amounts, and litigation posture. "
        "Cite specific article titles or URLs present in the context."
    )
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    try:
        if provider == "OpenAI":
            client = clients.get("OpenAI")
            if client is None:
                raise RuntimeError("OpenAI client missing.")

            # --- SPECIAL HANDLING for small models ---
            is_small = model in ["gpt-5-nano", "gpt-5-mini"]

            # Trim context aggressively for small models (token window is tighter)
            compact_context = context if not is_small else (
    "\n\n".join(context.split("\n\n")[:3])[:1000] if context else "")

            user_prompt_local = f"CONTEXT:\n{compact_context}\n\nQUESTION:\n{query}"

            # Build params
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_local},
                ],
            }

            # Small models prefer `max_tokens` and default temperature
            if is_small:
                params["max_completion_tokens"] = 220
                # do NOT set temperature for nano/mini (some versions reject it)
            else:
                params["max_completion_tokens"] = 400
                params["temperature"] = 0.2

            resp = client.chat.completions.create(**params)
            content = (resp.choices[0].message.content or "").strip()

            # Fallback retry if we got a blank response (common with tiny models)
            if not content:
                retry_params = {
    "model": model,
    "messages": [
        {"role": "user",
         "content": f"Answer briefly (1–3 sentences) based only on this text:\n{compact_context}\n\nQuestion: {query}"},],}
                
                if is_small:
                    retry_params["max_completion_tokens"] = 220
                else:
                    retry_params["max_completion_tokens"] = 400
                    retry_params["temperature"] = 0.2

                resp2 = client.chat.completions.create(**retry_params)
                content = (resp2.choices[0].message.content or "").strip()

            return content if content else "Sorry, I couldn't generate an answer for that with this model."

        elif provider == "Google":
            g = clients.get("Google")
            if g is None: raise RuntimeError("Google client missing.")
            model_obj = g.GenerativeModel(model)
            out = model_obj.generate_content(f"{system_prompt}\n\n{user_prompt}")
            return (out.text or "").strip()

        elif provider == "Anthropic":
            a = clients.get("Anthropic")
            if a is None: raise RuntimeError("Anthropic client missing.")
            msg = a.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=900,
                temperature=0.2,
            )
            return msg.content[0].text.strip()
        else:
            return "Unsupported provider selected."
    except Exception as e:
        return f"Model error: {e}"

# ==============================================================================
# 7) UI helpers
# ==============================================================================
def render_ranked_card(row):
    url = row.get("url", "")
    t   = row.get("title", "Untitled")
    src = row.get("source", "Unknown source")
    date = str(row.get("published_date", "")).split(" ")[0]
    score = float(row.get("interest_score", 0.0))
    summary = row.get("summary", "")
    st.markdown(f"**[{t}]({url})**  \n{src} — {date}")
    if summary:
        st.caption(summary[:500] + ("…" if len(summary) > 500 else ""))
    st.progress(min(1.0, score/100.0))
    st.divider()

# ==============================================================================
# 8) Main app
# ==============================================================================
def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.write(
        "Upload or use the default CSV, then explore **Top Interesting News** "
        "for a global law firm or **Ask** questions with RAG answers."
    )

    st.sidebar.header("Settings")
    provider = st.sidebar.selectbox("LLM Provider", list(MODEL_OPTIONS.keys()))
    model = st.sidebar.selectbox("Model", MODEL_OPTIONS[provider])

    uploaded = st.sidebar.file_uploader("Upload news CSV (optional)", type=["csv"])
    df = load_news_data(uploaded.read() if uploaded else None,
                        uploaded.name if uploaded else None)

    clients = get_api_clients()
    if provider not in clients:
        st.sidebar.error(f"API key missing for {provider}. Add it in Streamlit Secrets.")
        st.stop()

    collection = get_chroma_collection()

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("Rebuild Vector DB"):
            try:
                chromadb.PersistentClient(path=CHROMA_DB_PATH).delete_collection(name=CHROMA_COLLECTION_NAME)
            except Exception:
                pass
            st.rerun()
    with col_b:
        if st.button("Clear Chat"):
            st.session_state.pop("hw7_messages", None)
            st.rerun()

    setup_news_vector_db(collection, clients["OpenAI"], df, force_rebuild=(collection.count() == 0))

    tab_ask, tab_rank = st.tabs(["Ask the Bot (RAG)", "Explore Most Interesting"])
    if "hw7_messages" not in st.session_state:
        st.session_state.hw7_messages = []

    with tab_rank:
        st.subheader("Top Interesting News (Legal Impact Ranking)")
        n = st.slider("How many to show", 5, 25, 10)
        if st.button("Compute Ranking"):
            ranked = compute_interest_scores(df).head(n)
            for _, row in ranked.iterrows():
                render_ranked_card(row)
        with st.expander("How scoring works"):
            st.markdown(
                "- **Recency (0–25):** newer articles score higher (exp decay, ~2 weeks half-life)\n"
                "- **Legal Impact (0–25):** counts legal/regulatory keywords (capped)\n"
                "- **Materiality (0/15):** presence of monetary amounts\n"
                "- **Source (10):** constant; replace with outlet trust if you have it\n"
                "- **Uniqueness (5):** placeholder; add near-duplicate penalty if desired\n"
                "- **Total ~ 0–100**"
            )

    with tab_ask:
        st.subheader("Ask about the news (answers cite titles/URLs from context)")
        for m in st.session_state.hw7_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        query = st.chat_input("Ask a question (e.g., 'What are recent SEC crypto enforcement actions?')")
        if query:
            st.session_state.hw7_messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            context, cites = get_relevant_context(collection, clients["OpenAI"], query, n_results=4)
            answer = llm_answer(clients, provider, model, query, context)
            final = f"**Answer (`{provider} / {model}`):**\n\n{answer}"
            st.session_state.hw7_messages.append({"role": "assistant", "content": final})
            with st.chat_message("assistant"):
                st.markdown(final)
            if cites:
                with st.expander("Sources from retrieved context"):
                    for c in cites:
                        title = c.get("title") or "(untitled)"
                        src = c.get("source") or "source unknown"
                        date = c.get("date") or ""
                        url = c.get("url") or ""
                        st.markdown(f"- **[{title}]({url})** — {src} {date}")

    st.caption("Tip: Use the sidebar to switch providers/models and compare quality, cost, and latency.")


if __name__ == "__main__":
    main()
