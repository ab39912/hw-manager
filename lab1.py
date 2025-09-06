# hw1.py
import streamlit as st
from openai import OpenAI

def _extract_text(file) -> str:
    """
    Extract text from an uploaded .pdf or .txt file.
    Uses PyMuPDF (fitz) for PDFs, with a friendly ImportError message.
    """
    name = (getattr(file, "name", "") or "").lower()
    ext = "." + name.rsplit(".", 1)[1] if "." in name else ""

    raw_bytes = file.read()  # Read once

    if ext == ".txt":
        # Try utf-8, fallback to latin-1
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="ignore")

    elif ext == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            st.error(
                "PyMuPDF isn't installed. Install it with:\n\n"
                "`pip install PyMuPDF`"
            )
            return ""

        try:
            with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
                texts = [page.get_text() for page in doc]
            return "\n".join(texts).strip()
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            return ""

    else:
        return ""

def run():
    st.title("📄 HW 1: Document QA ChatBot")

    # API key (separate key from other labs)
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="api_key_hw1")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
        return

    client = OpenAI(api_key=openai_api_key)

    # File uploader (PDF or TXT)
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf or .txt)",
        type=("pdf", "txt"),
        key="uploader_hw1"
    )

    # Question box (disabled until a file is uploaded)
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        key="question_hw1",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        with st.spinner("Reading your document..."):
            # Reset pointer in case Streamlit reuses the buffer
            uploaded_file.seek(0)
            document_text = _extract_text(uploaded_file)

        if not document_text:
            st.warning("I couldn't extract any text from that file. Please try another document.")
            return

        messages = [
            {
                "role": "user",
                "content": f"Here's a document:\n\n{document_text}\n\n---\n\n{question}",
            }
        ]

        # Stream the response (same style as lab1)
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)

# Optional alias for callers expecting hw1.main()
main = run
