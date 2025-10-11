# streamlit_app.py
import math, re
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Helpers (pure functions) ----------
RECENCY_LAMBDA = 0.03  # ~30-day half-life
TOP_K_CONTEXT = 6

DEFAULT_LEGAL_TERMS = [
    "antitrust","merger","acquisition","litigation","lawsuit","settlement",
    "compliance","regulation","privacy","GDPR","CCPA","SEC","DOJ","sanction",
    "patent","trademark","copyright","arbitration","class action","whistleblower",
    "FCPA","AML","export control","subpoena","injunction"
]

def recency_score(age_days: float) -> float:
    return float(np.exp(-RECENCY_LAMBDA * float(age_days)))

def engagement_score(x: float, max_eng: float) -> float:
    if max_eng <= 0: return 0.0
    return float(np.log1p(x)/np.log1p(max_eng))

def legal_score(text: str, legal_terms) -> float:
    hits = sum(1 for t in legal_terms if re.search(rf"\b{re.escape(t)}\b", text, re.I))
    # Saturating transform so one or two matches help but don't dominate
    return 1.0 - math.exp(-0.7*hits)

class TfIdfIndex:
    def __init__(self, docs: List[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
        self.mat = self.vectorizer.fit_transform(docs)

    def query(self, q: str, topn: int = 50):
        v = self.vectorizer.transform([q.lower()])
        sims = cosine_similarity(v, self.mat).ravel()
        idx = sims.argsort()[::-1][:topn]
        return idx.tolist(), sims[idx].tolist()

def load_news(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    needed = ["id","title","text","date","source","url"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")
    df["text"] = df["text"].fillna("")
    df["title"] = df["title"].fillna("")
    if "engagement" not in df.columns:
        df["engagement"] = 0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["age_days"] = (datetime.utcnow() - df["date"]).dt.days.clip(lower=0)
    df["doc"] = (df["title"].str.strip() + " || " + df["text"].str.strip()).str.lower()
    return df

def novelty_from_tfidf(index: TfIdfIndex, order_idx: List[int]) -> np.ndarray:
    """Greedy novelty = 1 - max cosine sim to any previously-seen (newer) doc."""
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
    df["s_recency"]    = df["age_days"].apply(recency_score)
    df["s_engagement"] = df["engagement"].apply(lambda x: engagement_score(x, max_eng))
    df["s_legal"]      = df["doc"].apply(lambda t: legal_score(t, legal_terms))
    order = df.sort_values("date", ascending=False).index.tolist()
    nov = novelty_from_tfidf(index, order_idx=order)
    df["s_novelty"] = nov[df.index]
    df["score"] = (W["recency"]*df["s_recency"] +
                   W["engagement"]*df["s_engagement"] +
                   W["novelty"]*df["s_novelty"] +
                   W["legal"]*df["s_legal"])
    return df.sort_values("score", ascending=False).reset_index(drop=True)

def find_most_interesting(df, index, W, legal_terms, k=10):
    return score_all(df, index, W, legal_terms).head(k)

def find_topic(df, index, W, legal_terms, query: str, k=10, pool=60):
    idxs, sims = index.query(query, topn=pool)
    pool_df = df.iloc[idxs].copy()
    pool_df["qsim"] = sims
    ranked = score_all(pool_df, index, W, legal_terms)
    ranked["score"] = 0.85*ranked["score"] + 0.15*ranked["qsim"]
    return ranked.sort_values("score", ascending=False).head(k)

def make_context(block_df: pd.DataFrame, k: int = TOP_K_CONTEXT) -> str:
    rows = []
    for _, row in block_df.head(k).iterrows():
        snippet = (row["text"][:320] + "…") if len(row["text"]) > 320 else row["text"]
        rows.append(f"[{len(rows)+1}] {row['title']} "
                    f"({row['date'].date()}, {row['source']}) — {snippet} <{row['url']}>")
    return "\n".join(rows)

def answer_with_stub_llm(prompt: str, vendor: str = "cheap") -> str:
    """Placeholder for real LLM calls. Keeps the app fully local/offline."""
    tail = "\n".join(prompt.splitlines()[-12:])
    return f"(Vendor: {vendor})\n" + "—" * 22 + "\n" + tail


# ---------- App ----------
def run():
    st.set_page_config(page_title="News Reporting Bot", layout="wide")
    st.title("📰 News Reporting Bot (RAG-style retrieval + ranking)")

    # --- safe session defaults ---
    if "vendor" not in st.session_state:
        st.session_state.vendor = "cheap"
    if "weights" not in st.session_state:
        st.session_state.weights = {"recency":0.35,"engagement":0.15,"novelty":0.25,"legal":0.25}
    if "legal_terms" not in st.session_state:
        st.session_state.legal_terms = DEFAULT_LEGAL_TERMS.copy()

    # Sidebar controls
    with st.sidebar:
        st.header("1) Upload CSV")
        up = st.file_uploader(
            "Columns required: id,title,text,date,source,url  (optional: engagement)",
            type=["csv"],
        )
        st.header("2) LLM vendor")
        st.session_state.vendor = st.selectbox(
            "Choose vendor (generation step)",
            ["cheap", "expensive"],
            index=["cheap","expensive"].index(st.session_state.vendor),
        )
        st.header("3) Weights (sum normalized to 1)")
        wR = st.slider("Recency",    0.0, 1.0, float(st.session_state.weights["recency"]),    0.05)
        wE = st.slider("Engagement", 0.0, 1.0, float(st.session_state.weights["engagement"]), 0.05)
        wN = st.slider("Novelty",    0.0, 1.0, float(st.session_state.weights["novelty"]),    0.05)
        wL = st.slider("Legal",      0.0, 1.0, float(st.session_state.weights["legal"]),      0.05)
        total = max(wR + wE + wN + wL, 1e-9)
        st.session_state.weights = {
            "recency": wR/total, "engagement": wE/total, "novelty": wN/total, "legal": wL/total
        }

        st.header("4) Legal keywords")
        terms_txt = st.text_area("Comma-separated", ", ".join(st.session_state.legal_terms), height=120)
        st.session_state.legal_terms = [t.strip() for t in terms_txt.split(",") if t.strip()]

        st.header("5) Parameters")
        k    = st.slider("Top-k results", 3, 15, 6)
        pool = st.slider("Retriever pool (topic)", 20, 200, 60, step=10)

    st.info("Upload your CSV to activate the tabs. (Dates must be parseable, e.g., 2025-09-30.)")

    if up is None:
        st.stop()

    # Load and index
    try:
        df = load_news(up)
    except Exception as e:
        st.error(f"Load error: {e}")
        st.stop()

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
            st.download_button("Download CSV", show.to_csv(index=False).encode("utf-8"),
                               "most_interesting.csv", "text/csv")

    with tabs[1]:
        st.subheader("Find news about…")
        q = st.text_input("Query (e.g., privacy AND FTC)", "privacy AND FTC")
        if st.button("Search & rank"):
            ranked = find_topic(df, index, st.session_state.weights, st.session_state.legal_terms,
                                query=q, k=k, pool=pool)
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
            st.download_button("Download CSV", show.to_csv(index=False).encode("utf-8"),
                               "topic_results.csv", "text/csv")

    with tabs[2]:
        st.subheader("Debug")
        st.write("Weights:", st.session_state.weights)
        st.write("Legal terms:", st.session_state.legal_terms)
        st.write("Dataset shape:", df.shape)
        st.dataframe(df.head(20), use_container_width=True)


if __name__ == "__main__":
    run()
