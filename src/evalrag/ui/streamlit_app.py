import os

import httpx
import streamlit as st

API = os.environ.get("EVALRAG_API", "http://localhost:8000")

st.set_page_config(page_title="EvalRAG", layout="wide")
st.title("EvalRAG — single-doc Q&A with live trust scores")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("1. Upload a document")
    up = st.file_uploader("PDF / DOCX / MD / TXT", type=["pdf", "docx", "md", "txt"])
    if up and st.button("Ingest"):
        with st.spinner("Embedding…"):
            r = httpx.post(f"{API}/docs",
                           files={"file": (up.name, up.getvalue(), up.type)},
                           timeout=120)
        if r.status_code == 200:
            body = r.json()
            st.session_state.doc_id = body["id"]
            st.success(f"Ingested {body['chunks']} chunks. L2 eval running in background.")
        else:
            st.error(f"{r.status_code}: {r.text}")

    st.header("2. Toggles")
    use_reranker = st.checkbox("Use cross-encoder reranker", value=True)

    if st.session_state.doc_id:
        st.header("3. Doc dashboard")
        d = httpx.get(f"{API}/docs/{st.session_state.doc_id}").json()
        st.json(d)

st.header("Chat")
placeholder = (
    "Ask your next question"
    if st.session_state.history
    else "Ask the document a question"
)
disabled = st.session_state.doc_id is None
q = st.chat_input(placeholder, disabled=disabled)
if q and st.session_state.doc_id:
    with st.spinner("Thinking…"):
        r = httpx.post(f"{API}/query", json={
            "doc_id": st.session_state.doc_id,
            "question": q,
            "use_reranker": use_reranker,
        }, timeout=60)
    if r.status_code != 200:
        st.error(f"{r.status_code}: {r.text}")
    else:
        body = r.json()
        st.session_state.history.append((q, body))
        st.rerun()

for q_text, body in reversed(st.session_state.history):
    st.markdown(f"**Q:** {q_text}")
    st.markdown(f"**A:** {body['answer']}")
    ts = body.get("trust_score")
    if ts:
        color = {"green": "🟢", "amber": "🟡", "red": "🔴"}.get(ts["band"], "⚪")
        st.markdown(f"{color} **Trust: {ts['overall']}/100** "
                    f"(faith {ts['breakdown']['faithfulness']:.2f} · "
                    f"relev {ts['breakdown']['context_relevance']:.2f} · "
                    f"cite {ts['breakdown']['citation_coverage']:.2f})")
    else:
        st.markdown("⚪ Trust score unavailable")
    with st.expander("Retrieved chunks"):
        st.json(body["retrieval_trace"])
    st.caption(f"latency {body['latency_ms']} ms · cost ${body['cost_usd']:.4f}")
    st.divider()
