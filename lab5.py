# HW5.py
import streamlit as st
from openai import OpenAI, BadRequestError
from lab4 import retrieve_context, initialize_lab4_vector_db, DEFAULT_PDF_DIR, TOP_K

CHAT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a helpful assistant for course information.\n"
    "Use the retrieved context when possible, but if no context is found, say so clearly.\n"
    "Keep answers short and clear."
)

def build_messages(query, context_block, history):
    """Builds a conversation prompt including memory + retrieved context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Add history
    for role, text in history:
        messages.append({"role": role, "content": text})
    # Current user turn
    if context_block:
        user = (
            f"Question: {query}\n\n"
            "Here is relevant course information:\n"
            f"{context_block}\n"
            "Answer the question using this context."
        )
    else:
        user = f"Question: {query}\n\nNo relevant course info was found. Answer generally."
    messages.append({"role": "user", "content": user})
    return messages

def run():
    st.title("📝 HW5: Intelligent Chatbot with Short-Term Memory")

    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Missing OPENAI_API_KEY in `.streamlit/secrets.toml`.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)

    if "hw5_history" not in st.session_state:
        st.session_state.hw5_history = []  # list of (role, content)

    pdf_dir = DEFAULT_PDF_DIR
    if not st.session_state.get("Lab4_vectorDB_ready", False):
        initialize_lab4_vector_db(pdf_dir, openai_api_key)

    # Display chat transcript
    for role, text in st.session_state.hw5_history:
        with st.chat_message(role):
            st.markdown(text)

    query = st.chat_input("Ask me about the courses")
    if not query:
        return

    # Add user turn
    st.session_state.hw5_history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve context
    _, context_block = retrieve_context(query, openai_api_key, pdf_dir, n_results=TOP_K)

    # Build messages with memory
    messages = build_messages(query, context_block, st.session_state.hw5_history)

    # Call LLM
    with st.chat_message("assistant"):
        try:
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
            answer = resp.choices[0].message.content
            st.markdown(answer)
        except BadRequestError:
            st.error("Error calling OpenAI API — check model & key.")
            return

    # Save assistant reply to memory
    st.session_state.hw5_history.append(("assistant", answer))

if __name__ == "__main__":
    run()
