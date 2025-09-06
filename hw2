import streamlit as st
from openai import OpenAI

def run():
    st.title("📄 Ameya's Document Summarizer (Lab 2)")
    st.write(
        "Upload a document below and choose how you want it summarized. "
        "You can also toggle between different models."
    )

    # Load API key securely from Streamlit secrets
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_api_key)

    # Sidebar: summary options
    st.sidebar.header("Summary Options")
    summary_option = st.sidebar.radio(
        "Choose summary style:",
        [
            "Summarize in 100 words",
            "Summarize in 2 connecting paragraphs",
            "Summarize in 5 bullet points",
        ],
        index=0,
    )

    # Sidebar: model selection
    st.sidebar.header("Model Options")
    use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")
    
    if use_advanced:
        model = "gpt-4o"
    else:
        model = st.sidebar.selectbox(
            "Choose model:",
            ["gpt-5-nano","gpt-5-chat-latest"],
            index=0
        )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md"), key="lab2d_uploader"
    )

    if uploaded_file:
        document = uploaded_file.read().decode()

        # Pick instruction based on summary option
        if summary_option == "Summarize in 100 words":
            instruction = "Summarize the document in about 100 words."
        elif summary_option == "Summarize in 2 connecting paragraphs":
            instruction = "Summarize the document in 2 well-connected paragraphs."
        else:
            instruction = "Summarize the document in 5 clear bullet points."

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Document:\n{document}\n\nTask: {instruction}"},
        ]

        # Generate summary
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        st.subheader(f"Summary (Model: {model})")
        st.write_stream(stream)

main = run