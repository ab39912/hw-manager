# HW5.py
# ============================================================
# HW5: Short-Term Memory Chatbot with Vector Search (RAG)
# ============================================================

import os
import streamlit as st
from openai import OpenAI, BadRequestError

# Chroma for vector search
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


try:
    import pysqlite3
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

# ======================
# Config
# ======================
CHAT_MODEL = "gpt-5-nano"
SYSTEM_PROMPT = (
    "You are a helpful assistant for course information.\n"
    "Always use retrieved course context when possible.\n"
    "If no relevant context is found, clearly say so, then answer generally.\n"
    "Keep answers short, clear, and in plain English."
)

PDF_DIR = "lab4_pdfs"
CHROMA_PATH = ".chroma_lab5"
COLLECTION_NAME = "Lab5Collection"
TOP_K = 3


# ======================
# Vector DB Helpers
# ======================
def _embedding_fn(api_key: str):
    return OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")

def _client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def _get_or_create_collection(api_key: str):
    client = _client()
    emb = _embedding_fn(api_key)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb,
        metadata={"hnsw:space": "cosine"},
    )

def initialize_vector_db(api_key: str):
    """
    Ensure collection exists. 
    Unlike Lab4, this HW5 starter does NOT rebuild from PDFs,
    it just assumes they were embedded already or uses an empty collection.
    """
    st.session_state.setdefault("Lab5_collection_name", COLLECTION_NAME)
    st.session_state.setdefault("Lab5_chroma_path", CHROMA_PATH)
    st.session_state["Lab5_vectorDB_ready"] = True


def retrieve_context(query: str, api_key: str, n_results: int = TOP_K):
    """Return retrieved documents concatenated as context text."""
    if not st.session_state.get("Lab5_vectorDB_ready", False):
        initialize_vector_db(api_key)

    collection = _get_or_create_collection(api_key)
    try:
        res = collection.query(query_texts=[query], n_results=n_results)
    except Exception as e:
        st.error(f"Vector DB query error: {e}")
        return ""

    documents = res.get("documents", [[]])[0] if res.get("documents") else []
    return "\n\n".join(documents)


# ======================
# Build LLM Messages
# ======================
def build_messages(query: str, context: str, history: list):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Include short-term memory (chat history)
    for role, text in history:
        messages.append({"role": role, "content": text})
    # Current user turn with retrieved context
    if context:
        user_msg = (
            f"User question: {query}\n\n"
            f"Relevant course information:\n{context}\n\n"
            "Answer the question using ONLY this context."
        )
    else:
        user_msg = (
            f"User question: {query}\n\n"
            "No relevant course information was retrieved. Answer generally."
        )
    messages.append({"role": "user", "content": user_msg})
    return messages


# ======================
# Streamlit App
# ======================
def run():
    st.title("📝 HW5: Intelligent Chatbot with Short-Term Memory")

    # API key
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Missing OPENAI_API_KEY in `.streamlit/secrets.toml`.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Initialize memory
    if "hw5_history" not in st.session_state:
        st.session_state.hw5_history = []  # [(role, content), ...]

    # Show existing transcript
    for role, text in st.session_state.hw5_history:
        with st.chat_message(role):
            st.markdown(text)

    # Take user input
    query = st.chat_input("Ask me about the courses")
    if not query:
        return

    # Save user message
    st.session_state.hw5_history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve context from vector DB
    context = retrieve_context(query, api_key, n_results=TOP_K)

    # Build messages including memory
    messages = build_messages(query, context, st.session_state.hw5_history)

    # Get LLM response
    with st.chat_message("assistant"):
        try:
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
            answer = resp.choices[0].message.content
            st.markdown(answer)
        except BadRequestError:
            st.error("OpenAI BadRequestError — check model name and API key.")
            return

    # Save assistant response
    st.session_state.hw5_history.append(("assistant", answer))


if __name__ == "__main__":
    run()
