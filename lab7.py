# streamlit_app.py — News Reporting Bot (Gemini/Claude with secrets), CSV schema adapted
# Expected CSV columns: company_name, days_since_2000, Date, Document, URL

import math, re
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- LLM SDKs ----
try:
    import google.generativeai as genai   # pip install -U google-generativeai>=0.7
except Exception:
    genai = None
try:
    import anthropic                      # pip install -U anthropic>=0.25
except Exception:
    anthropic = None

RECENCY_LAMBDA = 0.03  # ~30-day half-life
TOP_K_CONTEXT = 6

DEFAULT_LEGAL_TERMS = [
    "antitrust","merger","acquisition","litigation","lawsuit","settlement",
    "compliance","regulation","privacy","GDPR","CCPA","SEC","DOJ","sanction",
    "patent","trademark","copyright","arbitration","class action","whistleblower",
    "FCPA","AML","export control","subpoena","injunction"
]

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]
CLAUDE_MODELS = ["claude-opus-4-1-20250805", "claude-sonnet-4-20250514"]


# ---------- Scoring helpers ----------
def recency_score(age_days: float) -> float:
    return float(np.exp(-RECENCY_LAMBDA * float(age_days)))

def engagement_score(x: float, max_eng: float) -> float:
    if max_eng <= 0:
        return 0.0
    return float(np.log1p(x) / np.log1p(max_eng))

def legal_score(text: str, legal_terms) -> float:
    hits = sum(1 for t in legal_terms if re.search(r"\b" + re.escape(t) + r"\b", text, re.I))
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
    Input columns:
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

    # Timezone-safe: parse as UTC and drop tz → tz-naive
    out["date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)

    out["source"] = df["company_name"].fillna("Unknown")
    out["url"] = df["URL"].fillna("")
    out["engagement"] = 0  # not provided

    out = out.dropna(subset=["date"]).reset_index(drop=True)

    now_ts = pd.Timestamp.utcnow().tz_localize(None)  # tz-naive to match 'date'
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
    ranked["score"] = 0.85 * ranked["score"] + 0.15 * pool_df["qsim"]
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


# ---------- LLM clients ----------
def gemini_client():
    if genai is None:
        raise RuntimeError("Gemini SDK not installed. `pip install google-generativeai`")
    api_key = (
        st.secrets.get("GOOGLE_API_KEY")
        or st.secrets.get("GEMINI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not found in secrets.toml")
    genai.configure(api_key=api_key)
    return genai

def anthropic_client():
    if anthropic is None:
        raise RuntimeError("Anthropic SDK not installed. `pip install anthropic`")
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in secrets.toml")
    return anthropic.Anthropic(api_key=api_key)


# ---------- Gemini robust extraction helper ----------
def _gemini_extract_text(resp):
    """
    Robustly extract text from a google-generativeai response.
    Returns (text, finish_reason, safety) where `text` may be "".
    """
    # Try quick accessor first
    try:
        txt = (getattr(resp, "text", None) or "").strip()
    except Exception:
        txt = ""

    finish_reason = None
    safety = None

    try:
        cands = getattr(resp, "candidates", []) or []
        if cands:
            cand0 = cands[0]
            finish_reason = getattr(cand0, "finish_reason", None) or getattr(cand0, "finishReason", None)
            safety = getattr(cand0, "safety_ratings", None) or getattr(cand0, "safetyRatings", None)

            if not txt:
                parts = []
                content = getattr(cand0, "content", None)
                if content and getattr(content, "parts", None):
                    for p in content.parts:
                        t = getattr(p, "text", None)
                        if t:
                            parts.append(t)
                        elif isinstance(p, dict) and "text" in p:
                            parts.append(p["text"])
                txt = "\n".join(parts).strip()
    except Exception:
        pass

    return txt, finish_reason, safety


# ---------- LLM calls ----------
def answer_with_llm(provider: str, model: str, system_prompt: str, user_prompt: str,
                    max_tokens: int = 800, temperature: float = 0.2) -> str:
    """
    Calls the selected LLM.

    Gemini:
      - Construct GenerativeModel with system_instruction.
      - generate_content with explicit contents payload.
      - Robust extraction + fallback on empty text (safety/max_tokens).

    Anthropic:
      - Messages API (max_tokens, temperature).
    """
    if provider == "Gemini":
        client = gemini_client()
        try:
            model_obj = client.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt
            )

            contents = [{"role": "user", "parts": [user_prompt]}]

            resp = model_obj.generate_content(
                contents,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                    "candidate_count": 1,
                },
            )

            text, finish_reason, _safety = _gemini_extract_text(resp)

            # If empty, try a gentler, shorter fallback
            if not text:
                fallback_prompt = (
                    user_prompt
                    + "\n\nPlease summarize in 4–6 concise bullet points with brief legal implications."
                )
                resp2 = model_obj.generate_content(
                    [{"role": "user", "parts": [fallback_prompt]}],
                    generation_config={
                        "max_output_tokens": min(512, max_tokens),
                        "temperature": 0.2,
                        "candidate_count": 1,
                    },
                )
                text2, finish_reason2, _safety2 = _gemini_extract_text(resp2)
                if text2:
                    return text2

                # Final fallback: switch to 2.5-flash
                try:
                    if model != "gemini-2.5-flash":
                        model_obj_flash = client.GenerativeModel(
                            model_name="gemini-2.5-flash",
                            system_instruction=system_prompt
                        )
                        resp3 = model_obj_flash.generate_content(
                            [{"role": "user", "parts": [fallback_prompt]}],
                            generation_config={
                                "max_output_tokens": min(512, max_tokens),
                                "temperature": 0.2,
                                "candidate_count": 1,
                            },
                        )
                        text3, finish_reason3, _safety3 = _gemini_extract_text(resp3)
                        if text3:
                            return text3
                        finish_reason = finish_reason3 or finish_reason
                except Exception:
                    pass

                return (
                    f"⚠️ Gemini returned no text (finish_reason={finish_reason}). "
                    f"This can happen due to safety or token limits. "
                    f"Try a more neutral phrasing or use 'gemini-2.5-flash'."
                )

            # Normal successful path
            if finish_reason and str(finish_reason).upper() not in ("STOP", "FINISH_REASON_STOP"):
                return text + f"\n\n_(Note: model finish_reason={finish_reason})_"

            return text

        except Exception as e:
            return f"⚠️ Gemini error for model '{model}': {e}"

    elif provider == "Claude":
        client = anthropic_client()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}],
            )
            parts = []
            for blk in getattr(resp, "content", []) or []:
                if getattr(blk, "type", None) == "text":
                    parts.append(blk.text)
                elif isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text", ""))
            return "\n".join(parts).strip() or "(empty response)"
        except Exception as e:
            return f"⚠️ Claude error for model '{model}': {e}"

    return "⚠️ Unknown provider."


# ---------- Streamlit app ----------
def run():
    st.set_page_config(page_title="News Reporting Bot", layout="wide")
    st.title("📰 News Reporting Bot")

    # ---- one-time migration to clear stale state when code changes ----
    APP_VERSION = "2025-10-14-gemini-fix"
    if st.session_state.get("_app_version") != APP_VERSION:
        for k in ["provider", "model", "weights", "legal_terms", "adv_k", "adv_pool", "adv_k_slider", "adv_pool_slider"]:
            st.session_state.pop(k, None)
        st.session_state["_app_version"] = APP_VERSION

    # Guard for any legacy/invalid provider values
    valid_providers = {"Gemini", "Claude"}
    if st.session_state.get("provider") not in valid_providers:
        st.session_state.provider = "Gemini"

    # Safe session defaults
    if "weights" not in st.session_state:
        st.session_state.weights = {"recency": 0.35, "engagement": 0.15, "novelty": 0.25, "legal": 0.25}
    if "legal_terms" not in st.session_state:
        st.session_state.legal_terms = DEFAULT_LEGAL_TERMS.copy()
    if "provider" not in st.session_state:
        st.session_state.provider = "Gemini"
    if "model" not in st.session_state:
        st.session_state.model = GEMINI_MODELS[0]

    # Sidebar (minimal)
    with st.sidebar:
        # Reset button for quick full refresh
        if st.button("🔄 Reset app & clear cache"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                st.cache_resource.clear()
            except Exception:
                pass
            st.session_state.clear()
            try:
                st.rerun()  # Streamlit >= 1.30
            except Exception:
                st.experimental_rerun()
        st.markdown("---")

        st.header("Upload CSV")
        up = st.file_uploader("Choose Example_news_info_for_testing.csv", type=["csv"])
        st.caption("Expected columns: company_name, days_since_2000, Date, Document, URL")

        st.header("LLM Provider & Model")
        provider = st.radio("Provider", ["Gemini", "Claude"], index=["Gemini","Claude"].index(st.session_state.provider))
        st.session_state.provider = provider

        if provider == "Gemini":
            model = st.selectbox(
                "Model", GEMINI_MODELS,
                index=GEMINI_MODELS.index(st.session_state.model) if st.session_state.model in GEMINI_MODELS else 0
            )
        else:
            model = st.selectbox(
                "Model", CLAUDE_MODELS,
                index=CLAUDE_MODELS.index(st.session_state.model) if st.session_state.model in CLAUDE_MODELS else 0
            )
        st.session_state.model = model

        # Advanced controls hidden by default
        with st.expander("Advanced (weights, legal keywords, params)", expanded=False):
            w = st.session_state.weights
            wR = st.slider("Recency", 0.0, 1.0, float(w["recency"]), 0.05)
            wE = st.slider("Engagement", 0.0, 1.0, float(w["engagement"]), 0.05)
            wN = st.slider("Novelty", 0.0, 1.0, float(w["novelty"]), 0.05)
            wL = st.slider("Legal", 0.0, 1.0, float(w["legal"]), 0.05)
            total = max(wR + wE + wN + wL, 1e-9)
            st.session_state.weights = {
                "recency": wR / total, "engagement": wE / total, "novelty": wN / total, "legal": wL / total
            }

            terms_txt = st.text_area("Legal keywords (comma-separated)", ", ".join(st.session_state.legal_terms), height=120)
            st.session_state.legal_terms = [t.strip() for t in terms_txt.split(",") if t.strip()]

            # Store params (if user opens expander)
            st.session_state.adv_k = st.slider("Top-k", 3, 15, 6, key="adv_k_slider")
            st.session_state.adv_pool = st.slider("Retriever pool (topic)", 20, 200, 60, step=10, key="adv_pool_slider")

    # Defaults if Advanced never opened
    k = st.session_state.get("adv_k", st.session_state.get("adv_k_slider", 6))
    pool = st.session_state.get("adv_pool", st.session_state.get("adv_pool_slider", 60))

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

    tabs = st.tabs(["⭐ Most interesting", "🔎 Topic search"])

    with tabs[0]:
        st.subheader("Most interesting")
        if st.button("Rank now", type="primary"):
            ranked = find_most_interesting(df, index, st.session_state.weights, st.session_state.legal_terms, k=k)
            ctx = make_context(ranked, k=k)
            system = "You are a precise news analyst for a global law firm."
            user = f"""Summarize the most interesting recent items and why.
Use the context below; add legal significance and novelty. Cite items as [#].
Context:
{ctx}

Output: 3–6 bullets + a short "Why this matters" paragraph."""
            ans = answer_with_llm(st.session_state.provider, st.session_state.model, system, user)
            st.markdown("### Answer")
            st.write(ans)
            st.markdown("### Top-k ranked")
            show = ranked[["title","date","source","score","url","s_recency","s_engagement","s_novelty","s_legal"]]
            st.dataframe(show, use_container_width=True)

    with tabs[1]:
        st.subheader("Find news about…")
        q = st.text_input("Query (e.g., privacy AND FTC)", "privacy AND FTC")
        if st.button("Search & rank"):
            ranked = find_topic(df, index, st.session_state.weights, st.session_state.legal_terms, query=q, k=k, pool=pool)
            ctx = make_context(ranked, k=k)
            system = "You are a precise news analyst for a global law firm."
            user = f"""User intent: {q}
Use the context below to answer succinctly and cite items as [#]. Explain legal implications.
Context:
{ctx}"""
            ans = answer_with_llm(st.session_state.provider, st.session_state.model, system, user)
            st.markdown("### Answer")
            st.write(ans)
            st.markdown("### Top-k ranked")
            show = ranked[["title","date","source","score","url","s_recency","s_engagement","s_novelty","s_legal"]]
            st.dataframe(show, use_container_width=True)


if __name__ == "__main__":
    run()
