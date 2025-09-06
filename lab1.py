import streamlit as st
from openai import OpenAI

def run():
    st.title("📄 Lab 1: Document QA ChatBot")

    openai_api_key = st.text_input("OpenAI API Key", type="password", key="api_key_lab1")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
        return

    client = OpenAI(api_key=openai_api_key)

    uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"), key="uploader_lab1")
    question = st.text_area("Now ask a question about the document!", placeholder="Can you give me a short summary?", key="question_lab1", disabled=not uploaded_file)

    if uploaded_file and question:
        document = uploaded_file.read().decode()
        messages = [
            {"role": "user", "content": f"Here's a document: {document}\n\n---\n\n{question}"}
        ]

        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)

# Optional alias for callers expecting lab1.main()
main = run
