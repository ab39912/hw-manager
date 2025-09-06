import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF

# Show title and description.
st.title("📄 Ameya's Document QA ChatBot - HW1")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get "
    "[here](https://platform.openai.com/account/api-keys)."
)

# To extract text from uploaded file (PDF or TXT only)
def extract_text(file) -> str:
    name = (getattr(file, "name", "") or "").lower()
    ext = "." + name.rsplit(".", 1)[1] if "." in name else ""

    # Read bytes once
    raw_bytes = file.read()

    if ext == ".txt":
        
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="ignore")

    elif ext == ".pdf":
        try:
            with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
                texts = [page.get_text() for page in doc]  # default text mode
            return "\n".join(texts).strip()
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            return ""

    else:
        return ""  

# Asking user to enter their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file: PDF or TXT only.
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf or .txt only)",
        type=("pdf", "txt")
    )

    # Text area to enter question
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        with st.spinner("Reading your document..."):
            document_text = extract_text(uploaded_file)

        if not document_text:
            st.warning("I couldn't extract any text from that file. Please try another document.")
        else:
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document:\n\n{document_text}\n\n---\n\n{question}",
                }
            ]

            # Generate an answer using the OpenAI API (streaming).
            stream = client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                stream=True,
            )

            # Output the response to the app.
            st.write_stream(stream)