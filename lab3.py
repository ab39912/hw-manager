# lab3.py
import textwrap
import traceback
from typing import List, Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup
import streamlit as st

# Optional vendor SDKs
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import anthropic
except Exception:
    anthropic = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------------------
# General Configuration
# ---------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly, step by step, and cite facts from the provided sources "
    "when relevant. If a fact is not in the sources, you may use general knowledge but say so."
)

# Token counting (best-effort)
try:
    import tiktoken
    def approx_tokens(text: str) -> int:
        # Generic tokenizer for most chat models
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return max(1, len(text) // 4)
        return len(enc.encode(text))
except Exception:
    def approx_tokens(text: str) -> int:
        return max(1, len(text) // 4)

def messages_token_count(messages: List[Dict[str, str]]) -> int:
    # Super rough chat-format estimate
    total = 0
    for m in messages:
        total += 4 + approx_tokens(m.get("content", ""))
    return total + 2

# ---------------------------
# Sidebar Model Picker
# ---------------------------
VENDOR_PRESETS = {
    "OpenAI": {
        "Budget": "gpt-5-nano",      # fast/cheap
        "Flagship": "gpt-5"          # adjust if you use a different flagship
    },
    "Anthropic": {
        "Budget": "claude-3-haiku-20240307",   # small/cheap
        "Flagship": "claude-3-5-sonnet-latest" # strong general-use
    },
    "Google (Gemini)": {
        "Budget": "gemini-1.5-flash",
        "Flagship": "gemini-1.5-pro"
    },
}

# ---------------------------
# URL fetching / cleaning
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_url_text(url: str, timeout: int = 15) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # strip script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text
    except Exception as e:
        return f"[ERROR fetching {url}: {e}]"

def make_sources_note(url1_text: str, url2_text: str, url1: Optional[str], url2: Optional[str]) -> str:
    blocks = []
    if url1 and url1_text:
        blocks.append(f"[Source 1: {url1}]\n{textwrap.shorten(url1_text, width=4000, placeholder=' …')}")
    if url2 and url2_text:
        blocks.append(f"[Source 2: {url2}]\n{textwrap.shorten(url2_text, width=4000, placeholder=' …')}")
    if not blocks:
        return ""
    return "REFERENCE NOTES FROM URLS:\n\n" + "\n\n".join(blocks)

# ---------------------------
# Memory Builders
# ---------------------------
def build_last_k_pairs(messages: List[Dict[str, str]], k_user: int = 6) -> List[Dict[str, str]]:
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    pick = user_idxs[-k_user:] if k_user > 0 else []
    buf = []
    for ui in pick:
        buf.append(messages[ui])
        # add the next assistant reply if any
        for j in range(ui + 1, len(messages)):
            if messages[j]["role"] == "assistant":
                buf.append(messages[j])
                break
    return buf

def build_token_capped(messages: List[Dict[str, str]], max_tokens: int = 2000) -> List[Dict[str, str]]:
    # Build from end so newest context is kept
    running: List[Dict[str, str]] = []
    for msg in reversed(messages):
        cand = [msg] + running
        if messages_token_count(cand) <= max_tokens:
            running = cand
        else:
            break
    return running

def update_running_summary(current_summary: str, recent_msgs: List[Dict[str, str]], summarize_fn) -> str:
    """
    Calls the selected vendor to refresh a compact running summary.
    summarize_fn: callable(prompt:str)->str (non-streaming)
    """
    snippet = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent_msgs[-8:])  # last 8 turns
    prompt = (
        "You are maintaining a running summary of the conversation. "
        "Keep it concise but complete enough to answer follow-up questions.\n\n"
        f"CURRENT SUMMARY (may be empty):\n{current_summary}\n\n"
        f"NEW EXCHANGES:\n{snippet}\n\n"
        "Update the summary in 4-7 bullet points. Do not include analysis about model behavior."
    )
    try:
        return summarize_fn(prompt)
    except Exception:
        return current_summary  # fail-safe

# ---------------------------
# Vendor wrappers (streaming)
# ---------------------------
def stream_answer_openai(model: str, api_key: str, messages: List[Dict[str, str]]):
    if OpenAI is None:
        raise RuntimeError("openai package not installed.")
    client = OpenAI(api_key=api_key)
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None) if chunk.choices else None
        if delta:
            yield delta

def summarize_openai(model: str, api_key: str, prompt: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package not installed.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You summarize conversations."},
                  {"role": "user", "content": prompt}],
        stream=False,
    )
    return resp.choices[0].message.content or ""

def stream_answer_anthropic(model: str, api_key: str, messages: List[Dict[str, str]]):
    if anthropic is None:
        raise RuntimeError("anthropic package not installed.")
    client = anthropic.Anthropic(api_key=api_key)

    # Convert to Anthropic format
    system = ""
    converted = []
    for m in messages:
        if m["role"] == "system":
            system += ("\n" + m["content"])
        else:
            converted.append({"role": m["role"], "content": m["content"]})

    with client.messages.stream(
        model=model,
        system=system.strip() or None,
        messages=converted,
        max_tokens=1024,
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and getattr(event.delta, "text", None):
                yield event.delta.text

def summarize_anthropic(model: str, api_key: str, prompt: str) -> str:
    if anthropic is None:
        raise RuntimeError("anthropic package not installed.")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        system="You summarize conversations.",
        messages=[{"role": "user", "content": prompt}],
    )
    # Concatenate text parts
    out = []
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            out.append(block.text)
    return "".join(out)

def stream_answer_gemini(model: str, api_key: str, messages: List[Dict[str, str]]):
    if genai is None:
        raise RuntimeError("google-generativeai package not installed.")
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)
    # Convert to Gemini "contents"
    contents = []
    sys = []
    for m in messages:
        if m["role"] == "system":
            sys.append(m["content"])
        else:
            contents.append({"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]})
    # Prepend a system-style instruction
    if sys:
        contents.insert(0, {"role": "user", "parts": ["SYSTEM INSTRUCTIONS:\n" + "\n".join(sys)]})
    resp = gm.generate_content(contents, stream=True)
    for chunk in resp:
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text

def summarize_gemini(model: str, api_key: str, prompt: str) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai package not installed.")
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)
    resp = gm.generate_content([prompt], stream=False)
    try:
        return resp.text or ""
    except Exception:
        return ""

# ---------------------------
# UI
# ---------------------------
def run():
    st.title("🧠 HW-3: Streaming Chat with URL Context")
    st.caption("Ask a question. Optionally add up to two URLs as references. Pick a model and a memory mode. The answer streams live.")

    # --- Sidebar: URLs, model, memory ---
    with st.sidebar:
        st.subheader("Reference URLs")
        url1 = st.text_input("URL 1 (optional)", placeholder="https://example.com/one")
        url2 = st.text_input("URL 2 (optional)", placeholder="https://example.com/two")

        st.subheader("Model Vendor & Tier")
        vendor = st.selectbox("Vendor", list(VENDOR_PRESETS.keys()))
        tier = st.selectbox("Tier", ["Budget", "Flagship"])
        model_id = VENDOR_PRESETS[vendor][tier]

        st.subheader("Conversation Memory")
        mem_mode = st.radio(
            "Memory strategy",
            ["Buffer: last 6 questions", "Conversation summary", "Token buffer (~2000 tokens)"],
            index=0,
            help="Choose how the chat keeps context for each request."
        )
        token_budget = st.number_input(
            "Token buffer size", min_value=500, max_value=8000, value=2000, step=100,
            help="Used only when 'Token buffer' is selected."
        )
        st.markdown("— Example prompts: *“Summarize the pricing section”, “What year did X happen?”, “List 3 key findings”.*")
        st.markdown(f"**Selected model:** `{vendor} / {model_id}`")

    # --- Secrets / API keys availability ---
    openai_key  = st.secrets.get("OPENAI_API_KEY")
    anthro_key  = st.secrets.get("ANTHROPIC_API_KEY")
    google_key  = st.secrets.get("GOOGLE_API_KEY", st.secrets.get("GEMINI_API_KEY"))

    # Vendor dispatch tables
    streamers = {
        "OpenAI":   (lambda msgs: stream_answer_openai(model_id, openai_key, msgs)),
        "Anthropic":(lambda msgs: stream_answer_anthropic(model_id, anthro_key, msgs)),
        "Google (Gemini)": (lambda msgs: stream_answer_gemini(model_id, google_key, msgs)),
    }
    summarizers = {
        "OpenAI":   (lambda prompt: summarize_openai(model_id, openai_key, prompt)),
        "Anthropic":(lambda prompt: summarize_anthropic(model_id, anthro_key, prompt)),
        "Google (Gemini)": (lambda prompt: summarize_gemini(model_id, google_key, prompt)),
    }

    # Validate key for selected vendor
    if vendor == "OpenAI" and not openai_key:
        st.error("Missing OPENAI_API_KEY in `.streamlit/secrets.toml`.")
        st.stop()
    if vendor == "Anthropic" and not anthro_key:
        st.error("Missing ANTHROPIC_API_KEY in `.streamlit/secrets.toml`.")
        st.stop()
    if vendor == "Google (Gemini)" and not google_key:
        st.error("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in `.streamlit/secrets.toml`.")
        st.stop()

    # --- Session State ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list[dict[str, str]]

    if "summary" not in st.session_state:
        st.session_state.summary = ""

    # --- Show transcript ---
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # --- Chat input ---
    user_q = st.chat_input("Type your question…")
    if not user_q:
        return

    # Log user turn
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # --- Build memory context per selected mode ---
    visible_history = st.session_state.messages[:]  # shallow copy

    if mem_mode.startswith("Buffer: last 6"):
        memory = build_last_k_pairs(visible_history, k_user=6)
        memory_label = "Buffer(last 6 Qs)"
    elif mem_mode.startswith("Conversation summary"):
        # Ensure we have/refresh summary using selected vendor
        summarize_fn = summarizers[vendor]
        st.session_state.summary = update_running_summary(
            st.session_state.summary, visible_history, summarize_fn
        )
        memory = [{"role": "system", "content": f"Conversation summary (use as context):\n{st.session_state.summary}"}]
        memory_label = "Summary"
    else:  # Token buffer
        memory = build_token_capped(visible_history, max_tokens=int(token_budget))
        memory_label = f"Token buffer(~{int(token_budget)}t)"

    # --- Fetch URLs (if any) and make a sources note ---
    url1_text = fetch_url_text(url1) if url1 else ""
    url2_text = fetch_url_text(url2) if url2 else ""
    sources_note = make_sources_note(url1_text, url2_text, url1, url2)

    # Compose messages for the model
    final_messages: List[Dict[str, str]] = []

    # System instruction + style
    final_messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # Include memory
    if memory:
        # Memory may already contain a system message; we keep both (order matters: generic system, then memory)
        final_messages.extend(memory)

    # Include sources from URLs
    if sources_note:
        final_messages.append({
            "role": "system",
            "content": (
                "You are given reference notes extracted from URLs. Prefer facts from these notes when relevant.\n\n"
                + sources_note
            )
        })

    # End with the user's latest question (what we want to answer now)
    final_messages.append({"role": "user", "content": user_q})

    # --- Stream the answer using the selected vendor ---
    try:
        with st.chat_message("assistant"):
            streamer = streamers[vendor]
            stream_gen = streamer(final_messages)
            full_text = st.write_stream(stream_gen)
        # Log assistant turn
        st.session_state.messages.append({"role": "assistant", "content": full_text})

        # If mem_mode = Conversation summary, refresh summary including this new answer
        if mem_mode.startswith("Conversation summary"):
            summarize_fn = summarizers[vendor]
            st.session_state.summary = update_running_summary(
                st.session_state.summary,
                st.session_state.messages,
                summarize_fn
            )

        st.caption(f"Context mode: {memory_label} • Messages sent: {len(final_messages)} • ~{messages_token_count(final_messages)} tokens (approx)")

    except Exception as e:
        with st.chat_message("assistant"):
            st.error("There was an error generating a response. See details below.")
            st.code("".join(traceback.format_exception_only(type(e), e)))
