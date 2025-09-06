import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# Optional provider SDKs
try:
    import anthropic
except Exception:
    anthropic = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import cohere
except Exception:
    cohere = None

# ---------- Helpers ----------
def read_url_content(url: str):
    """Fetch a URL and return visible text content."""
    try:
        url = url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "https://" + url

        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator="\n")
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
    if text and len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        return head + "\n\n...[content trimmed]...\n\n" + tail
    return text

def build_task_prompt(url: str, content: str, base_instruction: str, output_language: str) -> str:
    """Unified task text for non-OpenAI providers (or those that don't use role messages)."""
    return (
        f"Task: {base_instruction}\n"
        f"Write the entire output in {output_language}. Do not include any other language. "
        f"If translation is needed, translate as part of the summary.\n\n"
        f"URL: {url}\n\nExtracted Text:\n{content}\n"
    )

# ---------- Provider: OpenAI ----------
def validate_openai_key(client: OpenAI) -> bool:
    try:
        # Tiny probe to validate key (very low-cost)
        _ = client.chat.completions.create(
            model="gpt-4o-mini",  # small/cheap probe; if unavailable it will still validate auth
            messages=[{"role": "user", "content": "Reply with 'OK'."}],
            max_tokens=2,
        )
        return True
    except Exception as e:
        st.error(f"OpenAI key validation failed: {e}")
        return False

def summarize_with_openai(client: OpenAI, model: str, url: str, content: str, base_instruction: str, output_language: str):
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant that writes clear, faithful summaries. Always respond in {output_language} only.",
        },
        {
            "role": "user",
            "content": f"URL: {url}\n\nExtracted Text:\n{content}\n\nTask: {base_instruction} "
                       f"Write the entire output in {output_language}. Do not include any other languages. "
                       f"If translation is needed, translate as part of the summary."
        },
    ]
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    st.subheader(f"Summary (Model: {model}, Language: {output_language})")
    st.write_stream(stream)

# ---------- Provider: Anthropic (Claude) ----------
def validate_anthropic_key(api_key: str) -> bool:
    if not anthropic:
        st.error("Anthropic SDK is not installed. Add `anthropic` to requirements.txt.")
        return False
    try:
        client = anthropic.Client(api_key=api_key)
        _ = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=5,
            messages=[{"role": "user", "content": "Reply with OK"}],
        )
        return True
    except Exception as e:
        st.error(f"Anthropic key validation failed: {e}")
        return False

def summarize_with_anthropic(api_key: str, url: str, content: str, base_instruction: str, output_language: str):
    client = anthropic.Client(api_key=api_key)
    prompt = build_task_prompt(url, content, base_instruction, output_language)
    try:
        resp = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1500,
            system=f"Always answer in {output_language}.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic SDK returns a list of content blocks; join the text parts
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        st.subheader(f"Summary (Claude, Language: {output_language})")
        st.write(text)
    except Exception as e:
        st.error(f"Claude summarization failed: {e}")

# ---------- Provider: Google Gemini ----------
def validate_gemini_key(api_key: str) -> bool:
    if not genai:
        st.error("Google Generative AI SDK is not installed. Add `google-generativeai` to requirements.txt.")
        return False
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        _ = model.generate_content("Reply with OK")
        return True
    except Exception as e:
        st.error(f"Gemini key validation failed: {e}")
        return False

def summarize_with_gemini(api_key: str, url: str, content: str, base_instruction: str, output_language: str):
    genai.configure(api_key=api_key)
    prompt = build_task_prompt(url, content, base_instruction, output_language)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(
            f"System instruction: Always answer in {output_language}.\n\n{prompt}"
        )
        st.subheader(f"Summary (Gemini, Language: {output_language})")
        st.write(resp.text)
    except Exception as e:
        st.error(f"Gemini summarization failed: {e}")

# ---------- Provider: Cohere ----------
def validate_cohere_key(api_key: str) -> bool:
    if not cohere:
        st.error("Cohere SDK is not installed. Add `cohere` to requirements.txt.")
        return False
    try:
        ch = cohere.Client(api_key)
        _ = ch.chat(message="Reply with OK")
        return True
    except Exception as e:
        st.error(f"Cohere key validation failed: {e}")
        return False

def summarize_with_cohere(api_key: str, url: str, content: str, base_instruction: str, output_language: str):
    ch = cohere.Client(api_key)
    prompt = build_task_prompt(url, content, base_instruction, output_language)
    try:
        resp = ch.chat(
            model="command-r-plus",  # choose an appropriate Cohere chat model
            message=(
                f"System: Always respond in {output_language} only.\n\n{prompt}"
            )
        )
        text = resp.text if hasattr(resp, "text") else (resp.output_text if hasattr(resp, "output_text") else str(resp))
        st.subheader(f"Summary (Cohere, Language: {output_language})")
        st.write(text)
    except Exception as e:
        st.error(f"Cohere summarization failed: {e}")

# ---------- App ----------
def run():
    st.title("🔗 Ameya's URL Summarizer (Lab 2)")
    st.write("Enter a URL below, pick your summary style, choose the **LLM provider** (and model, if OpenAI), and select the output language.")

    # Load all possible keys from secrets (some may be missing)
    openai_api_key    = st.secrets.get("OPENAI_API_KEY", None)
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    gemini_api_key    = st.secrets.get("GEMINI_API_KEY", None)
    cohere_api_key    = st.secrets.get("COHERE_API_KEY", None)

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

    # Sidebar: output language
    st.sidebar.header("Output Language")
    output_language = st.sidebar.selectbox(
        "Select the language to output:",
        ["English", "French", "Spanish", "German", "Hindi"],
        index=0,
    )

    # Sidebar: LLM provider (NEW)
    st.sidebar.header("LLM Provider")
    provider = st.sidebar.selectbox(
        "Choose the LLM to use:",
        ["OpenAI", "Anthropic (Claude)", "Google Gemini", "Cohere"],
        index=0,
    )

    # Keep legacy Lab2 toggle (applies to OpenAI only)
    st.sidebar.header("Model Options (OpenAI)")
    use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")

    # If OpenAI and not advanced, allow model selection (as in Lab2)
    if provider == "OpenAI":
        if use_advanced:
            model = "gpt-4o"
        else:
            model = st.sidebar.selectbox(
                "Choose OpenAI model:",
                ["gpt-5-nano", "gpt-5-chat-latest"],
                index=0,
            )
    else:
        model = None  # not used for non-OpenAI selections

    # Top-of-screen: URL input
    url = st.text_input(
        "Enter a URL to summarize",
        placeholder="https://example.com/article",
    )

    summarize_clicked = st.button("Summarize")

    if (url and summarize_clicked) or (url and not summarize_clicked and st.session_state.get("auto_run_once") is None):
        st.session_state["auto_run_once"] = True

        with st.spinner("Fetching and summarizing the page..."):
            content = read_url_content(url)
            if not content:
                return

            content = trim_for_model(content)
            base_instruction = pick_instruction(summary_option)

            # Route by provider + validate key
            try:
                if provider == "OpenAI":
                    if not openai_api_key:
                        st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
                        return
                    openai_client = OpenAI(api_key=openai_api_key)
                    if not validate_openai_key(openai_client):
                        return
                    summarize_with_openai(openai_client, model, url, content, base_instruction, output_language)

                elif provider == "Anthropic (Claude)":
                    if not anthropic_api_key:
                        st.error("Missing ANTHROPIC_API_KEY in Streamlit secrets.")
                        return
                    if not validate_anthropic_key(anthropic_api_key):
                        return
                    summarize_with_anthropic(anthropic_api_key, url, content, base_instruction, output_language)

                elif provider == "Google Gemini":
                    if not gemini_api_key:
                        st.error("Missing GEMINI_API_KEY in Streamlit secrets.")
                        return
                    if not validate_gemini_key(gemini_api_key):
                        return
                    summarize_with_gemini(gemini_api_key, url, content, base_instruction, output_language)

                elif provider == "Cohere":
                    if not cohere_api_key:
                        st.error("Missing COHERE_API_KEY in Streamlit secrets.")
                        return
                    if not validate_cohere_key(cohere_api_key):
                        return
                    summarize_with_cohere(cohere_api_key, url, content, base_instruction, output_language)

                else:
                    st.error("Unsupported provider selection.")

            except Exception as e:
                st.error(f"Failed to generate summary: {e}")

# Keep the entry point
main = run
