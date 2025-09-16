# lab3.py
import streamlit as st
from openai import OpenAI

# ---------------------------
# Configuration
# ---------------------------
MODEL = "gpt-5-nano"  # default model
LAST_USER_MSGS_TO_KEEP = 20
SYSTEM_PROMPT = (
    "You are a friendly teacher who explains things to a 10-year-old. "
    "Use short sentences, simple words, and clear examples or tiny stories. "
    "Be kind and encouraging."
)

# Try to use tiktoken if available for better token counts
try:
    import tiktoken
    def count_tokens(text: str, model: str = MODEL) -> int:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
except Exception:
    def count_tokens(text: str, model: str = MODEL) -> int:
        # Simple heuristic fallback (~4 chars per token)
        return max(1, int(len(text) / 4))


def messages_token_count(messages, model: str = MODEL) -> int:
    """Approximate token count for [{role, content}] messages."""
    total = 0
    per_msg_overhead = 4
    for m in messages:
        total += per_msg_overhead + count_tokens(m.get("content", ""), model)
    return total + 2


# ---------------------------
# Buffer builders
# ---------------------------
def build_last_k_pairs_buffer(full_history, k_user_msgs=LAST_USER_MSGS_TO_KEEP):
    """
    [system] + last K user messages and each message's next assistant reply (if present),
    in chronological order.
    """
    user_indices = [i for i, m in enumerate(full_history) if m["role"] == "user"]
    pick_user_idxs = user_indices[-k_user_msgs:] if k_user_msgs > 0 else []
    buffer_msgs = []
    for ui in pick_user_idxs:
        buffer_msgs.append(full_history[ui])
        # include the assistant reply right after that user turn, if any
        for j in range(ui + 1, len(full_history)):
            if full_history[j]["role"] == "assistant":
                buffer_msgs.append(full_history[j])
                break
    return [{"role": "system", "content": SYSTEM_PROMPT}] + buffer_msgs


def build_token_capped_buffer(full_history, max_tokens: int, model: str = MODEL):
    """As many recent messages (from tail) as fit within max_tokens, plus the system prompt."""
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    running = [system_msg]
    current_tokens = messages_token_count(running, model=model)

    for msg in reversed(full_history):
        candidate = [msg] + running
        cand_tokens = messages_token_count(candidate, model=model)
        if cand_tokens <= max_tokens:
            running = [msg] + running
            current_tokens = cand_tokens
        else:
            break

    if running[0]["role"] != "system":
        running = [system_msg] + running
        current_tokens = messages_token_count(running, model=model)

    return running, current_tokens


# ---------------------------
# Stream helper
# ---------------------------
def stream_openai_response(client: OpenAI, messages, model: str = MODEL):
    """Stream chunks for st.write_stream and collect the full text."""
    response_text = ""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            response_text += delta
            yield delta
    stream.full_text = response_text  # type: ignore
    return


# ---------------------------
# Helpers for "more info" loop
# ---------------------------
YES_SET = {"y", "yes", "yeah", "yep", "sure", "ok", "okay", "please", "more", "tell me more"}
NO_SET  = {"n", "no", "nope", "nah", "stop", "enough"}

def normalize_reply(s: str) -> str:
    return s.strip().lower()

def ask_more_prompt(original_question: str) -> str:
    return (
        f"Give more information about this question in a way a 10-year-old understands: "
        f"'{original_question}'. Use simple words and short sentences. "
        f"Add one or two kid-friendly examples. Keep it helpful and clear."
    )


# ---------------------------
# UI
# ---------------------------
def run():
    st.title("💬 Lab 3C: Chatbot with Conversation Buffer")
    st.caption("Answers like you’re 10 years old. After each answer, I’ll ask: “DO YOU WANT MORE INFO?”")

    # ✅ Load API key from Streamlit secrets
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Missing OPENAI_API_KEY in `.streamlit/secrets.toml`.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)

    # Sidebar controls
    with st.sidebar:
        st.header("Conversation Buffer")
        buffer_mode = st.radio(
            "Choose buffer mode:",
            ["Last 20 exchanges", "Token-based"],
            index=0,
            help="Last 20 exchanges = last twenty user messages and their replies. Token-based = cap by a token budget."
        )
        max_ctx_tokens = None
        if buffer_mode == "Token-based":
            max_ctx_tokens = st.number_input(
                "Max context tokens",
                min_value=400,
                max_value=8000,
                value=1600,
                step=100,
                help="Maximum tokens to send as context to the model."
            )
        st.divider()
        st.markdown(f"**Model:** `{MODEL}`")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []  # full visible transcript
    if "awaiting_more_info" not in st.session_state:
        st.session_state.awaiting_more_info = False
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "more_info_round" not in st.session_state:
        st.session_state.more_info_round = 0  # optional counter

    # Show transcript so far
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question (or answer yes/no if I asked about more info)")
    if not user_input:
        return

    user_norm = normalize_reply(user_input)

    # Append user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.awaiting_more_info:
        # Expecting a yes/no
        if user_norm in YES_SET:
            followup = ask_more_prompt(st.session_state.last_question)

            # Build buffer for the LLM
            temp_history = st.session_state.messages + [{"role": "user", "content": followup}]
            if buffer_mode == "Token-based":
                to_send, ctx_tokens = build_token_capped_buffer(
                    temp_history, max_tokens=int(max_ctx_tokens), model=MODEL
                )
            else:
                to_send = build_last_k_pairs_buffer(temp_history, LAST_USER_MSGS_TO_KEEP)
                ctx_tokens = messages_token_count(to_send, model=MODEL)

            # Stream response and IMMEDIATELY render the prompt text
            with st.chat_message("assistant"):
                text_stream = stream_openai_response(client, to_send, model=MODEL)
                base_answer = st.write_stream(text_stream)
                st.markdown("\n\n**DO YOU WANT MORE INFO?** (yes/no)")

            combined = base_answer + "\n\n**DO YOU WANT MORE INFO?** (yes/no)"
            st.session_state.messages.append({"role": "assistant", "content": combined})
            st.session_state.more_info_round += 1
            st.session_state.awaiting_more_info = True  # keep looping
            st.caption(f"Context tokens (approx): ~{ctx_tokens}")
            return

        elif user_norm in NO_SET:
            with st.chat_message("assistant"):
                st.markdown("Okay! What question can I help with next?")
            st.session_state.messages.append({"role": "assistant", "content": "Okay! What question can I help with next?"})
            st.session_state.awaiting_more_info = False
            st.session_state.more_info_round = 0
            st.session_state.last_question = ""
            return
        else:
            # If the user typed something else, treat it as a NEW question and reset loop
            st.session_state.awaiting_more_info = False
            st.session_state.more_info_round = 0
            st.session_state.last_question = user_input  # new question below will handle

    # Not awaiting more info → treat input as a NEW QUESTION
    st.session_state.last_question = user_input

    # Build the buffer to send to the LLM
    if buffer_mode == "Token-based":
        to_send, ctx_tokens = build_token_capped_buffer(
            st.session_state.messages, max_tokens=int(max_ctx_tokens) if max_ctx_tokens else 1600, model=MODEL
        )
    else:
        to_send = build_last_k_pairs_buffer(st.session_state.messages, LAST_USER_MSGS_TO_KEEP)
        ctx_tokens = messages_token_count(to_send, model=MODEL)

    # Stream main answer and IMMEDIATELY render the prompt text
    with st.chat_message("assistant"):
        text_stream = stream_openai_response(client, to_send, model=MODEL)
        base_answer = st.write_stream(text_stream)
        st.markdown("\n\n**DO YOU WANT MORE INFO?** (yes/no)")

    combined = base_answer + "\n\n**DO YOU WANT MORE INFO?** (yes/no)"
    st.session_state.messages.append({"role": "assistant", "content": combined})
    st.session_state.awaiting_more_info = True
    st.session_state.more_info_round = 1
    st.caption(f"Context tokens (approx): ~{ctx_tokens}")