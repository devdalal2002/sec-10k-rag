"""
src/app.py - Streamlit chat UI over the SEC 10-K RAG pipeline.

Wraps retrieve() (src/retrieve.py) and generate_answer() (src/generate.py).
Locally this talks to Ollama (qwen2.5:7b). When a GROQ_API_KEY secret is
present (e.g. on Streamlit Community Cloud, where Ollama can't run), it
switches to Groq's free-tier API instead - see README for deploy notes.
"""

import html
import os
import sys
import traceback
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))


def _configure_backend() -> None:
    """Route generation to Groq when a key is available, else default to Ollama."""
    try:
        groq_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_key = None
    groq_key = groq_key or os.environ.get("GROQ_API_KEY")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ.setdefault("LLM_BACKEND", "groq")


_configure_backend()

from config import CACHE_DIR, RERANK_CACHE, LLM_BACKEND, GROQ_MODEL, LLM_MODEL  # noqa: E402
from retrieve import retrieve, load_rerank_cache  # noqa: E402
from generate import generate_answer  # noqa: E402

CACHE_DIR.mkdir(parents=True, exist_ok=True)
load_rerank_cache(RERANK_CACHE)

# Fixed to the headline config from eval/results.md (94.9% recall@5, 100% recall@10) -
# no user-facing retrieval controls, this is the one setup the eval says to use.
# Hosted deploys (Streamlit Cloud, detected via the Groq backend) are memory-constrained
# and can't reliably fit bge-reranker-base (~1.1GB) alongside everything else, so they
# drop to "hybrid_filter" - hybrid retrieval + the ticker/year metadata filter (the
# dominant recall lever per eval/results.md) but no reranker - instead of the full
# headline config. Local/Ollama runs keep the headline config.
COLLECTION = "sec_section_aware"
RETRIEVAL_CONFIG = "hybrid_filter" if LLM_BACKEND == "groq" else "hybrid_rerank_filter"
TOP_K = 5

# Curated from eval/ground_truth.csv (hand-verified against the filings) to span
# question types, plus one out-of-corpus question to demo the refusal behavior.
# Each was spot-checked end-to-end against the actual hosted config (hybrid_filter,
# no reranker) - several ground-truth questions that work under the local headline
# config (hybrid_rerank_filter) don't retrieve cleanly without the reranker on this
# numeric-table-heavy corpus, so this set is deliberately narrower than "spans every
# question type" to keep every button reliable.
EXAMPLE_QUESTIONS = [
    ("Apple FY23 revenue",
     "What were Apple's total net sales in fiscal year 2023?"),
    ("Google Cloud growth",
     "By how much did Google Cloud revenue grow from fiscal year 2023 to fiscal year 2024?"),
    ("Nvidia FY24 revenue",
     "What was Nvidia's total revenue in fiscal year 2024 (fiscal year ending January 28, 2024)?"),
    ("Meta privacy risks",
     "What regulatory risks related to data privacy and content does Meta disclose in its fiscal year 2023 10-K?"),
    ("Out-of-corpus (refusal demo)",
     "How did Microsoft's research and development expense compare to Oracle's R&D spending in fiscal year 2023?"),
]

st.set_page_config(page_title="SEC 10-K RAG", page_icon="\U0001F4C8", layout="centered")

CUSTOM_CSS = """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
}
.block-container {
    max-width: 780px;
    padding-top: 2.5rem;
    padding-bottom: 8rem;
}
.msg-row { display: flex; margin: 1.1rem 0; }
.msg-row.user { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; }
.msg-bubble {
    max-width: 82%;
    padding: 0.65rem 1rem;
    border-radius: 1.15rem;
    line-height: 1.6;
    font-size: 1rem;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.msg-bubble.user {
    background: var(--secondary-background-color);
    border-bottom-right-radius: 0.3rem;
}
.msg-bubble.assistant {
    background: transparent;
    max-width: 100%;
    padding: 0.1rem 0;
}
.source-meta {
    font-size: 0.85rem;
    font-weight: 600;
    opacity: 0.75;
    margin-bottom: 0.15rem;
}
.source-excerpt {
    font-size: 0.88rem;
    line-height: 1.55;
    opacity: 0.85;
    white-space: pre-wrap;
    word-wrap: break-word;
    padding-bottom: 0.6rem;
}
[data-testid="stChatInput"] textarea {
    border-radius: 1.3rem !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

model_name = GROQ_MODEL if LLM_BACKEND == "groq" else LLM_MODEL

title_col, info_col, clear_col = st.columns([7, 1, 1], vertical_alignment="center")
with title_col:
    st.title("SEC 10-K RAG")
with info_col:
    with st.popover("ℹ️", use_container_width=True):
        st.markdown(
            "Retrieval-augmented Q&A over 30 SEC 10-K filings "
            "(AAPL, MSFT, NVDA, META, GOOGL, AMZN, JPM, GS, WMT, TSLA x FY2022-2024).\n\n"
            "Answers are grounded and cited `[N]` against the retrieved excerpts - "
            "see [eval/results.md](https://github.com/devdalal2002/sec-10k-rag/blob/master/eval/results.md) "
            "for the retrieval benchmark behind these defaults.\n\n"
            f"Generation backend: **{LLM_BACKEND}** ({model_name})\n\n"
            f"Retrieval config: **{RETRIEVAL_CONFIG}**"
        )
        if LLM_BACKEND == "groq":
            st.caption("This host rebuilds the retrieval index from precomputed embeddings "
                       "on first use (a few seconds) and skips the cross-encoder reranker to "
                       "fit free-tier resource limits. It keeps the ticker/year metadata "
                       "filter (the biggest recall lever per eval/results.md), just without "
                       "the rerank step, so quality should sit between the 83.1% (hybrid) "
                       "and 94.9% (hybrid_rerank_filter) recall@5 numbers there.")
with clear_col:
    if st.button("Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(role: str, content: str) -> None:
    """Render a chat bubble. Content is HTML-escaped so any markdown-like
    syntax in the LLM's raw output (asterisks, brackets, underscores) shows
    up as plain literal text instead of being reinterpreted as formatting."""
    escaped = html.escape(content)
    st.markdown(
        f'<div class="msg-row {role}"><div class="msg-bubble {role}">{escaped}</div></div>',
        unsafe_allow_html=True,
    )


def render_sources(chunks: list, citations: list) -> None:
    if not chunks:
        return
    cited = {c["chunk_id"] for c in citations}
    with st.expander(f"Sources ({len(chunks)} excerpts retrieved)"):
        for i, chunk in enumerate(chunks, start=1):
            mark = " &middot; cited" if chunk["chunk_id"] in cited else ""
            st.markdown(
                f'<div class="source-meta">[{i}] {chunk["ticker"]} FY{chunk["fiscal_year"]} '
                f'&middot; {chunk["section_id"]} &middot; score {chunk["score"]:.3f}{mark}</div>',
                unsafe_allow_html=True,
            )
            excerpt = chunk["text"][:800] + ("..." if len(chunk["text"]) > 800 else "")
            st.markdown(
                f'<div class="source-excerpt">{html.escape(excerpt)}</div>',
                unsafe_allow_html=True,
            )
            st.divider()


for message in st.session_state.messages:
    render_message(message["role"], message["content"])
    if message["role"] == "assistant":
        render_sources(message.get("chunks", []), message.get("citations", []))

prompt = st.chat_input("Ask a question about the 10-K filings...")

st.caption("Try asking:")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, (label, question) in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(label, help=question):
        prompt = question

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    with st.spinner("Retrieving relevant filings and generating an answer..."):
        try:
            chunks = retrieve(prompt, collection=COLLECTION, config=RETRIEVAL_CONFIG, top_k=TOP_K)
            result = generate_answer(prompt, chunks)
            answer = result["answer"]
            citations = result["citations"]
        except Exception as exc:
            # Print the full traceback to server logs - the chat bubble only
            # shows str(exc), which isn't enough to diagnose intermittent
            # backend errors after the fact.
            print("=== retrieve/generate failed ===", file=sys.stderr)
            traceback.print_exc()
            answer = f"Something went wrong: {exc}"
            citations = []
            chunks = []

    render_message("assistant", answer)
    render_sources(chunks, citations)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "chunks": chunks,
    })
