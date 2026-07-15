import sys
from pathlib import Path

import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from scripts.legal_rag import generate_answer as rag_generate_answer
    from scripts.legal_rag import retrieve_context
except Exception as exc:  # pragma: no cover - runtime fallback
    rag_generate_answer = None
    retrieve_context = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = None

st.set_page_config(page_title="Judgement Bot", page_icon="⚖️", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I can help answer questions about the legal judgments in this workspace.",
        }
    ]


def get_answer(query: str) -> str:
    if rag_generate_answer and retrieve_context:
        try:
            context_docs = retrieve_context(query, k=5)
            result = rag_generate_answer(query, context_docs)
            return result.get("answer", "")
        except Exception as exc:
            return f"Sorry, I could not generate an answer right now. Error: {exc}"

    return (
        "The legal retrieval backend is not available yet. "
        "Please make sure the RAG pipeline is configured correctly."
    )


st.title("⚖️ Judgement Bot")
st.caption("A ChatGPT-style assistant for legal judgments and case-related questions")

with st.sidebar:
    st.header("Options")
    st.info("This app uses your legal retrieval pipeline when available.")

    if IMPORT_ERROR:
        st.warning(f"Backend issue: {IMPORT_ERROR}")

    if st.button("Clear chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I can help answer questions about the legal judgments in this workspace.",
            }
        ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about a judgment, case, or legal topic...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_answer(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
