"""Streamlit chat app for the GNEM Company GraphRAG assistant.

The assistant answers questions about Georgia EV supply-chain companies. It tries
Text2Cypher first by default, falls back to tool-routed retrieval when needed,
generates an answer from the retrieved context, and remembers the conversation so
follow-up questions work.

Run::

    NEO4J_PASSWORD=... uv run --extra ollama \\
        streamlit run examples/apps/company_chat_app.py

Requires a local Ollama server with the generation and embedding models pulled,
and the Company vector/full-text indexes already built
(see examples/database_operations/setup_company_retrieval_ollama.py).
"""

from __future__ import annotations

import os
import pathlib
import sys

import streamlit as st
from neo4j import GraphDatabase

# Reuse the retrieval/engine wiring from the question_answering example.
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[1] / "question_answering")
)

from company_graphrag_tools import (  # noqa: E402
    DEFAULT_TOP_K,
    Settings,
    build_engine,
    load_settings,
)
from neo4j_graphrag.message_history import InMemoryMessageHistory  # noqa: E402


st.set_page_config(page_title="GNEM Company GraphRAG", page_icon="🔋", layout="wide")


@st.cache_resource(show_spinner="Connecting to Neo4j and building the engine...")
def get_engine(password: str, top_k: int):
    """Build (and cache) the driver + tool-routed GraphRAG engine for a session."""
    # Load defaults (URI/user/db/models) from env, then inject the password.
    os.environ["NEO4J_PASSWORD"] = password
    settings: Settings = load_settings()
    driver = GraphDatabase.driver(
        settings.uri, auth=(settings.username, settings.password)
    )
    driver.verify_connectivity()
    rag, router = build_engine(driver, settings, top_k=top_k)
    return rag, settings


def resolve_password() -> str | None:
    """Password from NEO4J_PASSWORD env, else a sidebar input (kept out of code)."""
    env_password = os.getenv("NEO4J_PASSWORD")
    if env_password:
        return env_password
    return st.sidebar.text_input(
        "Neo4j password", type="password", help="Set NEO4J_PASSWORD to skip this."
    )


def render_context(retriever_result) -> None:
    """Show which tool was chosen and the retrieved companies."""
    metadata = retriever_result.metadata or {}
    tools = metadata.get("tools_selected") or ["<none>"]
    items = retriever_result.items
    with st.expander(f"Retrieved context — routed to {', '.join(tools)} ({len(items)} items)"):
        if metadata.get("fallback"):
            st.info(f"Router selected no tool; used fallback: {metadata['fallback']}")
        if metadata.get("llm_response"):
            st.caption(f"Router note: {metadata['llm_response']}")
        for i, item in enumerate(items, start=1):
            st.markdown(f"**{i}.** {item.content}")


# ---- Sidebar ---------------------------------------------------------------
st.sidebar.title("GNEM GraphRAG")
password = resolve_password()
top_k = st.sidebar.slider("top_k (retrieval)", 10, 100, DEFAULT_TOP_K, step=10)
if st.sidebar.button("Clear chat"):
    st.session_state.pop("messages", None)
    st.session_state.pop("history", None)
    st.rerun()

if not password:
    st.info("Enter your Neo4j password in the sidebar (or set NEO4J_PASSWORD) to start.")
    st.stop()

try:
    rag, settings = get_engine(password, top_k)
except Exception as exc:  # connection / model errors
    st.error(f"Could not start the engine: {type(exc).__name__}: {exc}")
    st.stop()

st.sidebar.markdown(
    f"**Generation:** `{settings.generation_model}`  \n"
    f"**Embeddings:** `{settings.embedding_model}`  \n"
    f"**Database:** `{settings.database}`  \n"
    f"**Ollama:** `{settings.ollama_host}`"
)

# ---- Chat state ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "retriever_result"?}
if "history" not in st.session_state:
    st.session_state.history = InMemoryMessageHistory()

st.title("🔋 Georgia EV Supply-Chain Assistant")
st.caption(
    "Ask about companies, tiers, EV supply-chain roles, OEMs, locations, employment "
    "and products. Text2Cypher is tried first; fallback routing handles semantic "
    "questions and follow-ups."
)

# Replay prior turns.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("retriever_result") is not None:
            render_context(message["retriever_result"])

# ---- New question ----------------------------------------------------------
question = st.chat_input("Ask a question about Georgia EV supply-chain companies...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Routing, retrieving and generating..."):
            try:
                # Pass PRIOR turns so follow-ups resolve; the current question is
                # added to history afterwards.
                result = rag.search(
                    question,
                    message_history=st.session_state.history,
                    retriever_config={},
                    return_context=True,
                )
                answer = result.answer.strip() or "_No answer produced._"
                st.markdown(answer)
                render_context(result.retriever_result)
            except Exception as exc:
                answer = f"Error: {type(exc).__name__}: {exc}"
                st.error(answer)
                result = None

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "retriever_result": result.retriever_result if result else None,
        }
    )
    # Record the turn for future follow-up questions.
    st.session_state.history.add_message({"role": "user", "content": question})
    st.session_state.history.add_message({"role": "assistant", "content": answer})
