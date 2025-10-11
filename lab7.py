# streamlit_app.py — timezone-safe version for: company_name, days_since_2000, Date, Document, URL
import math, re
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RECENCY_LAMBDA = 0.03  # ~30-day half-life
TOP_K_CONTEXT = 6

DEFAULT_LEGAL_TERMS = [
    "antitrust","merger","acquisition","litigation","lawsuit","settlement",
    "compliance","regulation","privacy","GDPR","CCPA","SEC","DOJ","sanction",
    "patent","trademark","copyright","arbitration","class action","whistleblower",
    "FCPA","AML","export control","subpoena","injunction"
]

# ---------- Scoring helpers ----------
def recency_score(age_days: float) -> float:
    return float(np.exp(-RECENCY_LAMBDA * float(age_days)))

def engagement_score(x: float, max_eng: float) -> float:
    if max_eng <= 0:
        return 0.0
    return float(np.log1p(x) / np.log1p(max_eng))

def legal_score(text: str, legal_terms) -> float:
    hits = sum(1 for t in legal_terms if re.search(r"\b" + re.escape(t) + r"\b", text, re.I))
    # Saturating transform so 1–2 hits help but don't dominate
    return 1.0 - math.exp(-0.7 * hits)

class TfIdfIndex:
    def __init__(self, docs: List[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
        self.mat = self.vectorizer.fit_transform(docs)

    def query(self, q: str, topn: int = 50):
        v = self.vectorizer.transform([q.lower()])
        sims = cosine_similarity(v, self.mat).ravel()
        idx = sims.argsort()[::-1][:topn]
        return idx.tolist(), sims[idx].tolist()

# ---------- CSV → internal schema ----------
def normalize_from_example_csv(file) -> pd.DataFrame:
    """
    Input columns (your CSV):
      - company_name, days_since_2000, Date, Document, URL
    Output (internal schema):
      id,title,text,date,source,url,engagement,age_days,doc
    """
    df = pd.read_csv(file)
    need = ["company_name", "Date", "Document", "URL"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"CSV must have columns: {need}. Missing: {miss}")

    out = pd.DataFrame()
    out["id"] = np.arange(len(df)) + 1

    docs = df["Document"].fillna("").astype(str)
    title = docs.str.split(".").str[0].str.strip()
    title = np.where(title.astype(str).str.len() == 0, docs.str.slice(0, 120), title)

    out["title"] = pd.Series(title).fillna("Untitled")
    out["text"] = docs

    # --- TIMEZONE-SAFE DATE PARSING ---
    if "Date" in df.columns:
        # Force parse as UTC (tz-aware), then drop tz to become tz-naive
        out["date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        base = pd.Timestamp("2000-01-01")
        days = pd.to_numeric(df.get("days_since_2000", 0), errors="coerce").fillna(0).astype(int)
        out["date"] = base + pd.to_timedelta(days, unit="D")  # already tz-naive

    out["source"] = df["company_name"].fillna("Unknown")
    out["url"] = df["URL"].fillna("")
    out["engagement"] = 0  # not provided; set to zero

    out = out.dropna(subset=["date"]).reset_index(drop=True)

    # Use a tz-naive "now" to match tz-naive dates
    now_ts = pd.Timestamp.utcnow().tz_localize(None)
    out["age_days"] = (now_ts - out["date"]).dt.days.clip(lower=0)
    out["doc"] = (out["title"].str.strip() + " || " + out["text"].str.strip()).str.lower()
    return out

# ---------- Novelty + ranking ----------
def novelty_from_tfidf(index: TfIdfIndex, order_idx: List[int]) -> np.ndarray:
    """Greedy novelty = 1 - max cosine similarity to any previously-seen (newer) doc."""
    M = index.mat
    novelty = np.ones(M.shape[0], dtype=float)
    seen = []
    for ridx in order_idx:
        cand = M[ridx]
        if not seen:
            novelty[ridx] = 1.0
        else:
            sims = cosine_similarity(cand, M[seen]).ravel()
            novelty[ridx] = 1.0 - float(np.max(sims))
        seen.append(ridx)
    return novelty

def score_all(df: pd.DataFrame, index: TfIdfIndex, W: dict, legal_terms) -> pd.DataFrame:
    max_eng = float(max(1.0, df["engagement"].max()))
    df = df.copy()
    df["s_recency"] = df["age_days"].apply(recency_score)
    df["s_engagement"] = df["engagement"].apply(lambda x: engagement_score(x, max_eng))
    df["s_legal"] = df["doc"].apply(lambda t: legal_score(t, legal_terms))
    order = df.sort_values("date", ascending=False).index.tolist()
    nov = novelty_from_tfidf(index, order_idx=order)
    df["s_novelty"] = nov[df.index]
    df["score"] = (
        W["recency"] * df["s_recency"]
        + W["engagement"] * df["s_engagement"]
        + W["novelty"] * df["s_novelty"]
        + W["legal"] * df["s_legal"]
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)

def find_most_interesting(df, index, W, legal_terms, k: int = 10):
    return score_all(df, index, W, legal_terms).head(k)

def find_topic(df, index, W, legal_terms, query: str, k: int = 10, pool: int = 60):
    idxs, sims = index.query(query, topn=pool)
    pool_df = df.iloc[idxs].copy()
    pool_df["qsim"] = sims
    ranked = score_all(pool_df, index, W, legal_terms)
    ranked["score"] = 0.85 * ranked["score"] + 0.15 * pool_df["qsim"]  # small tie-breaker
    return ranked.sort_values("score", ascending=False).head(k)

def make_context(block_df: pd.DataFrame, k: int = TOP_K_CONTEXT) -> str:
    rows = []
    for _, row in block_df.head(k).iterrows():
        snippet = (row["text"][:320] + "…") if len(row["text"]) > 320 else row["text"]
        rows.append(
            f"[{len(rows)+1}] {row['title']} "
            f"({row['date'].date()}, {row['source']}) — {snippet} <{row['url']}>"
        )
    return "\n".join(rows)

def answer_with_stub_llm(prompt: str, vendor: str = "cheap") -> str:
    """Stub so the app runs offline; swap with your real LLM calls."""
    tail = "\n".join(prompt.splitlines()[-12:])
    return f"(Vendor: {vendor})\n" + "—" * 22 + "\n" + tail

# ---------- Streamlit app ----------
def run():
    st.set_page_config(page_title="News Reporting Bot (Example CSV)", layout="wide")
    st.title("📰 News Reporting Bot — adapted to Example_news_info_for_testing.csv")

    # Safe session defaults
    if "vendor" not in st.session_state:
        st.session_state.vendor = "cheap"
    if "weights" not in st.session_state:
        st.session_state.weights = {"recency": 0.35, "engagement": 0.15, "novelty": 0.25, "legal": 0.25}
    if "legal_terms" not in st.session_state:
        st.session_state.legal_terms = DEFAULT_LEGAL_TERMS.copy()

    # Sidebar
    with st.sidebar:
        st.header("Upload CSV")
        up = st.file_uploader("Choose Example_news_info_for_testing.csv", type=["csv"])
        st.caption("Expected columns: company_name, days_since_2000, Date, Document, URL")

        st.header("LLM vendor")
        st.session_state.vendor = st.selectbox(
            "Vendor", ["cheap", "expensive"],
            index=["cheap", "expensive"].index(st.session_state.vendor)
        )

        st.header("Weights (normalized)")
        wR = st.slider("Recency", 0.0, 1.0, float(st.session_state.weights["recency"]), 0.05)
        wE = st.slider("Engagement", 0.0, 1.0, float(st.session_state.weights["engagement"]), 0.05)
        wN = st.slider("Novelty", 0.0, 1.0, float(st.session_state.weights["novelty"]), 0.05)
        wL = st.slider("Legal", 0.0, 1.0, float(st.session_state.weights["legal"]), 0.05)
        total = max(wR + wE + wN + wL, 1e-9)
        st.session_state.weights = {
            "recency": wR / total, "engagement": wE / total, "novelty": wN / total, "legal": wL / total
        }

        st.header("Legal keywords")
        terms_txt = st.text_area("Comma-separated", ", ".join(st.session_state.legal_terms), height=120)
        st.session_state.legal_terms = [t.strip() for t in terms_txt.split(",") if t.strip()]

        st.header("Params")
        k = st.slider("Top-k", 3, 15, 6)
        pool = st.slider("Retriever pool (topic)", 20, 200, 60, step=10)

    if up is None:
        st.info("Upload the CSV to activate the app.")
        return

    # Load & index
    try:
        df = normalize_from_example_csv(up)
    except Exception as e:
        st.error(f"Load/normalize error: {e}")
        return

    @st.cache_resource(show_spinner=False)
    def build_index(docs: List[str]):
        return TfIdfIndex(docs)

    index = build_index(df["doc"].tolist())

    tabs = st.tabs(["⭐ Most interesting", "🔎 Topic search", "🧪 Debug"])

    with tabs[0]:
        st.subheader("Most interesting")
        if st.button("Rank now", type="primary"):
            ranked = find_most_interesting(df, index, st.session_state.weights, st.session_state.legal_terms, k=k)
            ctx = make_context(ranked, k=k)
            prompt = f"""System: You are a precise news analyst for a global law firm.
User: What are the most interesting recent items and why?
Context:
{ctx}

Instructions: Summarize key items (3–6 bullets). Explain legal significance and novelty.
Cite like [#] and end with a short "Why this matters" paragraph."""
            ans = answer_with_stub_llm(prompt, st.session_state.vendor)
            st.markdown("### Answer")
            st.write(ans)
            st.markdown("### Top-k ranked")
            show = ranked[["title","date","source","score","url","s_recency","s_engagement","s_novelty","s_legal"]]
            st.dataframe(show, use_container_width=True)
            st.download_button(
                "Download CSV", show.to_csv(index=False).encode("utf-8"), "most_interesting.csv", "text/csv"
            )

    with tabs[1]:
        st.subheader("Find news about…")
        q = st.text_input("Query (e.g., privacy AND FTC)", "privacy AND FTC")
        if st.button("Search & rank"):
            ranked = find_topic(
                df, index, st.session_state.weights, st.session_state.legal_terms, query=q, k=k, pool=pool
            )
            ctx = make_context(ranked, k=k)
            prompt = f"""System: You are a precise news analyst for a global law firm.
User intent: {q}
Context:
{ctx}

Instructions: Provide a focused answer on the topic, cite [#], and explain legal implications."""
            ans = answer_with_stub_llm(prompt, st.session_state.vendor)
            st.markdown("### Answer")
            st.write(ans)
            st.markdown("### Top-k ranked")
            show = ranked[["title","date","source","score","url","s_recency","s_engagement","s_novelty","s_legal"]]
            st.dataframe(show, use_container_width=True)
            st.download_button(
                "Download CSV", show.to_csv(index=False).encode("utf-8"), "topic_results.csv", "text/csv"
            )

    with tabs[2]:
        st.subheader("Debug")
        st.write("Weights:", st.session_state.weights)
        st.write("Legal terms:", st.session_state.legal_terms)
        st.write("Dataset shape:", df.shape)
        st.dataframe(df.head(20), use_container_width=True)

if __name__ == "__main__":
    run()
