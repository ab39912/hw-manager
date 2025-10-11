# lab7.py
# HW 7 — CSV+PDF RAG bot with Chroma VectorDB + chunking
# Provides a `run()` function for use with st.navigation, and a __main__ guard to run standalone.

import os
import io
import re
import json
import math
import datetime as dt
from typing import List, Dict, Any, Tuple, Optional

# ── SQLite shim (Chroma needs sqlite>=3.35; this helps on hosted envs)
try:
    import pysqlite3  # type: ignore
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np

import chromadb
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

# Optional LLM summarization
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DB_DIR = ".chroma_news"
COLLECTION_NAME = "news"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150

LEGAL_KEYWORDS = [
    "lawsuit","regulation","regulatory","compliance","antitrust","merger",
    "acquisition","privacy","gdpr","ccpa","sec","doj","ftc","patent","ip",
    "court","settlement","sanction","fine","subpoena","injunction","litigation",
    "governance","esg","risk","data breach","cybersecurity","contract","arbitration"
]

def _get_openai_client() -> Optional[Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)

def _embed(texts: List[str]) -> np.ndarray:
    vecs = _embedder().encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vecs.astype(np.float32)

def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) + 1 <= chunk_size:
            buf = s if not buf else buf + " " + s
        else:
            if buf:
                chunks.append(buf.strip())
            if chunks:
                tail = chunks[-1][-overlap:]
                buf = (tail + " " + s).strip()
            else:
                buf = s
            if len(buf) > chunk_size * 2:
                for i in range(0, len(buf), chunk_size - overlap):
                    chunks.append(buf[i:i + chunk_size])
                buf = ""
    if buf:
        chunks.append(buf.strip())
    out = []
    for c in chunks:
        if len(c) <= chunk_size:
            out.append(c)
        else:
            for i in range(0, len(c), chunk_size - overlap):
                out.append(c[i:i + chunk_size])
    return [c for c in out if c]

def _read_pdf_pages(file_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        pages.append((i, t))
    return pages

def _infer_row_text(row: pd.Series) -> str:
    preferred = ["title","headline","summary","description","content","body","text"]
    bits = []
    for c in preferred:
        if c in row and isinstance(row[c], str) and row[c].strip():
            bits.append(f"{c.capitalize()}: {row[c].strip()}")
    if not bits:
        for c, v in row.items():
            if isinstance(v, str) and v.strip():
                bits.append(f"{c}: {v.strip()}")
    return " | ".join(bits)

def _parse_date(x: Any):
    try:
        if pd.isna(x):
            return None
        return pd.to_datetime(str(x)).date()
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _chroma() -> Tuple[PersistentClient, Collection]:
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        col = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    except Exception:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        col = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return client, col

def _reset_collection():
    client, _ = _chroma()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def _add(col: Collection, records: List[Dict[str, Any]]):
    if not records:
        return
    docs = [r["text"] for r in records]
    ids  = [r["id"] for r in records]
    metas= [r.get("metadata", {}) for r in records]
    embs = _embed(docs)
    col.add(documents=docs, ids=ids, embeddings=embs.tolist(), metadatas=metas)

def _query(col: Collection, q: str, where: Optional[Dict[str, Any]] = None, k: int = 8) -> Dict[str, Any]:
    q_emb = _embed([q])[0].tolist()
    return col.query(query_embeddings=[q_emb], n_results=k, where=where or {})

def _legal_kw_score(text: str) -> float:
    t = text.lower()
    c = sum(t.count(kw) for kw in LEGAL_KEYWORDS)
    return 1.0 - math.exp(-c)

def _recency_score(date_str: Any) -> float:
    d = _parse_date(date_str)
    if not d:
        return 0.5
    days = (dt.date.today() - d).days
    return float(math.exp(-max(days, 0)/30.0))

def _interest(sim: float, legal: float, rec: float) -> float:
    return 0.60*sim + 0.25*rec + 0.15*legal

def _rank_interest(col: Collection, k: int = 10):
    probe = "Important legal, regulatory, or high-risk news relevant to global law firms and their clients"
    res = _query(col, probe, k=max(k*4, 40))
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids   = res.get("ids", [[]])[0]
    probe_vec = _embed([probe])
    cand_vecs = _embed(docs)
    sims = cosine_similarity(probe_vec, cand_vecs)[0]
    ranked = []
    for doc, meta, _id, s in zip(docs, metas, ids, sims):
        sc = _interest(float(s), _legal_kw_score(doc or ""), _recency_score(meta.get("date")))
        ranked.append({"id": _id, "text": doc, "meta": meta, "score": sc})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:k]

def _meta_str(m: Dict[str, Any]) -> str:
    parts = []
    if m.get("title"): parts.append(f"**{str(m['title']).strip()}**")
    if m.get("date"):  parts.append(f"Date: {str(m['date']).strip()}")
    if m.get("source"):parts.append(f"Source: {str(m['source']).strip()}")
    if m.get("doc_type"): parts.append(f"Type: {m['doc_type']}")
    if m.get("page") is not None: parts.append(f"Page: {m['page']}")
    if m.get("row_id") is not None: parts.append(f"Row: {m['row_id']}")
    return " • ".join(parts)

def _summarize(query: str, contexts: List[str]) -> str:
    client = _get_openai_client()
    if not client:
        return "(No LLM set; showing extracted context)\n\n" + "\n\n".join(contexts)[:2000]
    prompt = f"""You are a news assistant for a global law firm.
Ground your answer ONLY in the provided context.

Question: {query}

Context:
{chr(10).join('- '+c for c in contexts)}

Answer:"""
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM error: {e}; falling back to extracted context)\n\n" + "\n\n".join(contexts)[:2000]

def ingest_csv(col: Collection, df: pd.DataFrame, source_name: str):
    recs = []
    for idx, row in df.iterrows():
        text = _infer_row_text(row)
        if not text:
            continue
        meta = {"doc_type":"csv_row","source":source_name,"row_id":int(idx)}
        for cand in ["date","published","pub_date","time","timestamp"]:
            if cand in df.columns:
                meta["date"] = str(row.get(cand))
                break
        for cand in ["title","headline"]:
            if cand in df.columns:
                meta["title"] = str(row.get(cand))
                break
        if "source" in df.columns:
            meta["source"] = str(row.get("source"))
        recs.append({"id": f"csv-{source_name}-{idx}", "text": text, "metadata": meta})
    _add(col, recs)

def ingest_pdf(col: Collection, file_bytes: bytes, filename: str, chunk_size: int, overlap: int):
    recs = []
    for pgi, ptext in _read_pdf_pages(file_bytes):
        if not ptext.strip():
            continue
        for j, ch in enumerate(_split_text(ptext, chunk_size, overlap)):
            recs.append({
                "id": f"pdf-{filename}-{pgi}-{j}",
                "text": ch,
                "metadata": {"doc_type":"pdf","source":filename,"page":int(pgi),"title":filename}
            })
    _add(col, recs)

def run():
    st.title("📰 HW 7 — CSV + PDF RAG (Chroma VectorDB)")
    with st.sidebar:
        st.header("Ingest Data")
        use_examples = st.toggle("Use example files if found", value=True)
        csv_up = st.file_uploader("Upload news CSV", type=["csv"])
        pdf_ups = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

        st.subheader("Chunking")
        ch_size = st.slider("Chunk size (chars)", 400, 1600, DEFAULT_CHUNK_SIZE, 50)
        ch_overlap = st.slider("Chunk overlap (chars)", 50, 300, DEFAULT_CHUNK_OVERLAP, 10)

        st.subheader("Index")
        _, col = _chroma()
        if st.button("Reset index (danger)", type="secondary"):
            _reset_collection()
            st.success("Index reset.")

        if st.button("Build / Update Index", type="primary"):
            with st.spinner("Indexing..."):
                if use_examples:
                    ex_csv = "/mnt/data/Example_news_info_for_testing.csv"
                    ex_pdf = "/mnt/data/HW 7 (2).pdf"
                    if os.path.exists(ex_csv):
                        try:
                            df = pd.read_csv(ex_csv)
                            ingest_csv(col, df, source_name=os.path.basename(ex_csv))
                            st.info(f"Ingested example CSV: {ex_csv}")
                        except Exception as e:
                            st.warning(f"Example CSV failed: {e}")
                    if os.path.exists(ex_pdf):
                        try:
                            with open(ex_pdf, "rb") as f:
                                ingest_pdf(col, f.read(), os.path.basename(ex_pdf), ch_size, ch_overlap)
                            st.info(f"Ingested example PDF: {ex_pdf}")
                        except Exception as e:
                            st.warning(f"Example PDF failed: {e}")

                if csv_up is not None:
                    try:
                        df = pd.read_csv(csv_up)
                        ingest_csv(col, df, source_name=csv_up.name)
                        st.success(f"Ingested CSV: {csv_up.name} (rows={len(df)})")
                    except Exception as e:
                        st.error(f"CSV ingest failed: {e}")

                if pdf_ups:
                    for pf in pdf_ups:
                        try:
                            ingest_pdf(col, pf.read(), pf.name, ch_size, ch_overlap)
                            st.success(f"Ingested PDF: {pf.name}")
                        except Exception as e:
                            st.error(f"PDF ingest failed for {pf.name}: {e}")

    tab1, tab2, tab3 = st.tabs(["🔎 Ask", "⭐ Most Interesting", "🎯 Topic"])
    _, col = _chroma()

    with tab1:
        st.subheader("Ask a question")
        q = st.text_input("Your question", value="What should a global law firm care about here?")
        topk = st.slider("Top-K passages", 3, 15, 6, 1)
        if st.button("Search & Answer", key="btn_qna"):
            res = _query(col, q.strip(), k=topk)
            docs  = res.get("documents",[[]])[0]
            metas = res.get("metadatas",[[]])[0]
            ids   = res.get("ids",[[]])[0]
            if not docs:
                st.info("No results yet. Build the index first.")
            else:
                for d, m, _ in zip(docs, metas, ids):
                    with st.container(border=True):
                        st.markdown(_meta_str(m))
                        st.write(d)
                st.markdown("#### Answer")
                st.write(_summarize(q.strip(), docs))

    with tab2:
        st.subheader("Top 'Most Interesting' (law-firm lens)")
        k = st.slider("How many items?", 3, 20, 10, 1)
        if st.button("Rank now", key="btn_rank"):
            ranked = _rank_interest(col, k=k)
            for i, r in enumerate(ranked, start=1):
                with st.container(border=True):
                    st.markdown(f"### #{i} — score: {r['score']:.3f}")
                    st.markdown(_meta_str(r["meta"]))
                    st.write(r["text"])

    with tab3:
        st.subheader("Find news about a specific topic")
        topic = st.text_input("Topic (e.g., 'antitrust', 'privacy', 'mergers')", value="privacy")
        where_filter = st.text_input("Optional JSON filter (Chroma 'where')", value="")
        topk2 = st.slider("Top-K", 3, 20, 8, 1, key="topk2")
        if st.button("Search by topic", key="btn_topic"):
            where = {}
            if where_filter.strip():
                try:
                    where = json.loads(where_filter)
                except Exception as e:
                    st.warning(f"Ignoring filter (bad JSON): {e}")
            res = _query(col, topic.strip(), where=where, k=topk2)
            docs  = res.get("documents",[[]])[0]
            metas = res.get("metadatas",[[]])[0]
            ids   = res.get("ids",[[]])[0]
            if not docs:
                st.info("No results yet. Build the index first.")
            else:
                for d, m, _ in zip(docs, metas, ids):
                    with st.container(border=True):
                        st.markdown(_meta_str(m))
                        st.write(d)
                st.markdown("#### Summary")
                st.write(_summarize(f"Summarize topic: {topic}", docs))

# Allow running this page by itself (outside navigation)
if __name__ == "__main__":
    st.set_page_config(page_title="HW 7 — Document QA", layout="wide")
    run()
