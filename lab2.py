import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# ---------- Helpers ----------
def read_url_content(url: str):
    """Fetch a URL and return visible text content."""
    try:
        # Basic normalization
        url = url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "https://" + url

        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        # Remove script/style tags to reduce noise
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        # Get text with line breaks
        text = soup.get_text(separator="\n")
        # Collapse excessive blank lines
        lines = [ln.strip() for ln in text.splitlines()]
        text = "\n".join(ln for ln in lines if ln)

        return text
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def pick_instruction(summary_option: str) -> str:
    if summary_option == "Summarize in 100 words":
        return "Summarize the document in about 100 words."
    elif summary_option == "Summarize in 2 connecting paragraphs":
        return "Summarize the document in 2 well-connected paragraphs."
    else:
        return "Summarize the document in 5 clear bullet points."

def trim_for_model(text: str, max_chars: int = 20000) -> str:
    """Trim overly long content to keep prompt size reasonable."""
    if text and len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        return head + "\n\n...[content trimmed]...\n\n" + tail
    return text

# ---------- App ----------
def run():
    st.title("🔗 Ameya's URL Summarizer (HW 2)")
    st.write("Enter a URL below and choose how you want it summarized. You can also toggle between different models.")

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
            ["gpt-5-nano", "gpt-5-chat-latest"],
            index=0,
        )

    # Top-of-screen: URL input
    url = st.text_input(
        "Enter a URL to summarize",
        placeholder="https://example.com/article",
    )

    # Optional explicit action button (works on Enter too)
    summarize_clicked = st.button("Summarize")

    if (url and summarize_clicked) or (url and not summarize_clicked and st.session_state.get("auto_run_once") is None):
        # Run once automatically on first valid URL entry (UX nicety)
        st.session_state["auto_run_once"] = True

        with st.spinner("Fetching and summarizing the page..."):
            content = read_url_content(url)
            if not content:
                return

            content = trim_for_model(content)

            instruction = pick_instruction(summary_option)
            messages = [
                {"role": "system", "content": "You are a helpful assistant that writes clear, faithful summaries."},
                {"role": "user", "content": f"URL: {url}\n\nExtracted Text:\n{content}\n\nTask: {instruction}"},
            ]

            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )
                st.subheader(f"Summary (Model: {model})")
                st.write_stream(stream)
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")

# Keep the entry point name the same
main = run
