import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

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

ANTHROPIC_MODEL_ALIASES = {
    # ✅ Latest and supported as of 2025:
    "opus":       "claude-opus-4-20250514",       # Opus 4
    "opus-4.1":   "claude-opus-4-1-20250805",     # Opus 4.1 (Aug 2025 release)
    "sonnet":     "claude-sonnet-4-20250514",     # Sonnet 4
    "sonnet-3.7": "claude-3-7-sonnet-20250219",   # Sonnet 3.7 (Feb 2025 release)
    "haiku":      "claude-3-5-haiku-20241022",    # Haiku 3.5
}
def resolve_anthropic_model(selected: str) -> str:
    return ANTHROPIC_MODEL_ALIASES.get(selected, selected)

# ===========================
# Helpers
# ===========================
def read_url_content(url: str):
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
    return (
        f"Task: {base_instruction}\n"
        f"Write the entire output in {output_language}. Do not include any other language. "
        f"If translation is needed, translate as part of the summary.\n\n"
        f"URL: {url}\n\nExtracted Text:\n{content}\n"
    )


# OpenAI

def validate_openai_key(client: OpenAI) -> bool:
    try:
        _ = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with 'OK'."}],
            max_tokens=2,
        )
        return True
    except Exception as e:
        st.error(f"OpenAI key validation failed: {e}")
        return False

def summarize_with_openai(client: OpenAI, model: str, url: str, content: str, base_instruction: str, output_language: str):
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Always respond in {output_language}."},
        {"role": "user", "content": f"URL: {url}\n\nExtracted Text:\n{content}\n\nTask: {base_instruction}"}
    ]
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    st.subheader(f"Summary (OpenAI: {model}, Language: {output_language})")
    st.write_stream(stream)


# Anthropic (Claude)

def validate_anthropic_key(api_key: str, selected_model: str) -> bool:
    if not anthropic:
        st.error("Anthropic SDK is not installed.")
        return False
    try:
        model_id = resolve_anthropic_model(selected_model)
        client = anthropic.Anthropic(api_key=api_key)
        _ = client.messages.create(
            model=model_id,
            max_tokens=5,
            messages=[{"role": "user", "content": "Reply with OK"}],
        )
        return True
    except Exception as e:
        st.error(f"Anthropic key/model validation failed: {e}")
        return False

def summarize_with_anthropic(api_key: str, url: str, content: str, base_instruction: str, output_language: str, selected_model: str):
    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_task_prompt(url, content, base_instruction, output_language)
    model_id = resolve_anthropic_model(selected_model)
    try:
        resp = client.messages.create(
            model=model_id,
            max_tokens=1500,
            system=f"Always answer in {output_language}.",
            messages=[{"role": "user", "content": prompt}],
        )
        text_chunks = []
        for part in getattr(resp, "content", []) or []:
            if hasattr(part, "text") and part.text:
                text_chunks.append(part.text)
        text = "".join(text_chunks).strip() or str(resp)
        st.subheader(f"Summary (Claude: {model_id}, Language: {output_language})")
        st.write(text)
    except Exception as e:
        st.error(f"Claude summarization failed: {e}")


# Google Gemini

def validate_gemini_key(api_key: str, model_name: str) -> bool:
    if not genai:
        st.error("Google Generative AI SDK is not installed.")
        return False
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        _ = model.generate_content("Reply with OK")
        return True
    except Exception as e:
        st.error(f"Gemini key/model validation failed: {e}")
        return False

def summarize_with_gemini(api_key: str, model_name: str, url: str, content: str, base_instruction: str, output_language: str):
    genai.configure(api_key=api_key)
    prompt = build_task_prompt(url, content, base_instruction, output_language)
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(f"System instruction: Always answer in {output_language}.\n\n{prompt}")
        st.subheader(f"Summary (Gemini: {model_name}, Language: {output_language})")
        st.write(resp.text)
    except Exception as e:
        st.error(f"Gemini summarization failed: {e}")


# Cohere

def validate_cohere_key(api_key: str, model_name: str) -> bool:
    if not cohere:
        st.error("Cohere SDK is not installed.")
        return False
    try:
        ch = cohere.Client(api_key)
        _ = ch.chat(model=model_name, message="Reply with OK")
        return True
    except Exception as e:
        st.error(f"Cohere key/model validation failed: {e}")
        return False

def summarize_with_cohere(api_key: str, model_name: str, url: str, content: str, base_instruction: str, output_language: str):
    ch = cohere.Client(api_key)
    prompt = build_task_prompt(url, content, base_instruction, output_language)
    try:
        resp = ch.chat(model=model_name, message=f"System: Always respond in {output_language}.\n\n{prompt}")
        text = resp.text if hasattr(resp, "text") else str(resp)
        st.subheader(f"Summary (Cohere: {model_name}, Language: {output_language})")
        st.write(text)
    except Exception as e:
        st.error(f"Cohere summarization failed: {e}")


# App

def run():
    st.title("🔗 Ameya's URL Summarizer (HW 2)")
    st.write("Enter a URL, choose summary style, provider, and output language.")

    # Load API keys
    openai_api_key    = st.secrets.get("OPENAI_API_KEY")
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY")
    gemini_api_key    = st.secrets.get("GEMINI_API_KEY")
    cohere_api_key    = st.secrets.get("COHERE_API_KEY")

    # Sidebar: summary options
    st.sidebar.header("Summary Options")
    summary_option = st.sidebar.radio("Choose summary style:", [
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs",
        "Summarize in 5 bullet points",
    ], index=0)

    # Sidebar: output language
    st.sidebar.header("Output Language")
    output_language = st.sidebar.selectbox("Select output language:", ["English", "French", "Spanish", "German", "Hindi"])

    # Sidebar: LLM provider
    st.sidebar.header("LLM Provider")
    provider = st.sidebar.selectbox("Choose the LLM:", ["OpenAI", "Anthropic (Claude)", "Google Gemini", "Cohere"])

    # Anthropic
    anthropic_model_choice = None
    if provider == "Anthropic (Claude)":
        st.sidebar.subheader("Anthropic Models")
        use_adv_claude = st.sidebar.checkbox("Use Advanced Claude (Sonnet/Opus family)")
        if use_adv_claude:
            anthropic_model_choice = st.sidebar.selectbox("Advanced Claude model:", ["sonnet", "opus", "opus-4.1", "sonnet-3.7"])
        else:
            anthropic_model_choice = st.sidebar.selectbox("Claude model:", ["haiku", "sonnet", "opus", "opus-4.1", "sonnet-3.7"])

    # Gemini
    gemini_model_choice = None
    if provider == "Google Gemini":
        st.sidebar.subheader("Gemini Models")
        use_adv_gemini = st.sidebar.checkbox("Use Advanced Gemini (1.5 Pro)")
        if use_adv_gemini:
            gemini_model_choice = "gemini-1.5-pro"
        else:
            gemini_model_choice = st.sidebar.selectbox("Gemini model:", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"])

    # Cohere
    cohere_model_choice = None
    if provider == "Cohere":
        st.sidebar.subheader("Cohere Models")
        cohere_model_choice = st.sidebar.selectbox("Cohere model:", ["command-r-plus", "command-r", "command"])

    # OpenAI
    model = None
    if provider == "OpenAI":
        st.sidebar.header("Model Options (OpenAI)")
        use_adv_openai = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")
        if use_adv_openai:
            model = "gpt-4o"
        else:
            model = st.sidebar.selectbox("OpenAI model:", ["gpt-5-nano", "gpt-5-chat-latest"])

    # Input URL
    url = st.text_input("Enter a URL to summarize", placeholder="https://example.com/article")
    summarize_clicked = st.button("Summarize")

    if url and summarize_clicked:
        with st.spinner("Fetching and summarizing..."):
            content = read_url_content(url)
            if not content:
                return
            content = trim_for_model(content)
            base_instruction = pick_instruction(summary_option)

            try:
                if provider == "OpenAI":
                    if not openai_api_key: return st.error("Missing OPENAI_API_KEY")
                    client = OpenAI(api_key=openai_api_key)
                    if not validate_openai_key(client): return
                    summarize_with_openai(client, model, url, content, base_instruction, output_language)

                elif provider == "Anthropic (Claude)":
                    if not anthropic_api_key: return st.error("Missing ANTHROPIC_API_KEY")
                    if not anthropic_model_choice: return st.error("Select a Claude model")
                    if not validate_anthropic_key(anthropic_api_key, anthropic_model_choice): return
                    summarize_with_anthropic(anthropic_api_key, url, content, base_instruction, output_language, anthropic_model_choice)

                elif provider == "Google Gemini":
                    if not gemini_api_key: return st.error("Missing GEMINI_API_KEY")
                    if not gemini_model_choice: return st.error("Select a Gemini model")
                    if not validate_gemini_key(gemini_api_key, gemini_model_choice): return
                    summarize_with_gemini(gemini_api_key, gemini_model_choice, url, content, base_instruction, output_language)

                elif provider == "Cohere":
                    if not cohere_api_key: return st.error("Missing COHERE_API_KEY")
                    if not cohere_model_choice: return st.error("Select a Cohere model")
                    if not validate_cohere_key(cohere_api_key, cohere_model_choice): return
                    summarize_with_cohere(cohere_api_key, cohere_model_choice, url, content, base_instruction, output_language)

            except Exception as e:
                st.error(f"Failed to generate summary: {e}")


main = run
