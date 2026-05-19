"""Streamlit dashboard for CodeReviewBot metrics.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
import streamlit as st

from config import settings

st.set_page_config(page_title="CodeReviewBot Dashboard", page_icon="🤖", layout="wide")


@st.cache_resource
def _get_collection():
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(settings.CHROMA_COLLECTION)


def main() -> None:
    st.title("CodeReviewBot — Review Memory Dashboard")

    col = _get_collection()
    total = col.count()

    st.metric("Total review memories stored", total)

    if total == 0:
        st.info("No reviews stored yet. Trigger a PR review to populate the memory.")
        return

    st.subheader("Recent Review Memories")
    results = col.get(limit=50, include=["documents", "metadatas"])

    rows = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        rows.append({
            "Repo": meta.get("repo", ""),
            "PR": meta.get("pr_number", ""),
            "File": meta.get("filename", ""),
            "Comment (truncated)": meta.get("comment", "")[:120],
            "Snippet (truncated)": doc[:80],
        })

    st.dataframe(rows, use_container_width=True)

    st.subheader("Semantic Search")
    query = st.text_area("Paste a code snippet to find similar past reviews:")
    if st.button("Search") and query.strip():
        from memory.vector_store import query_similar
        hits = query_similar(query, n_results=5)
        for i, h in enumerate(hits, 1):
            with st.expander(f"#{i} — {h['repo']} PR#{h['pr_number']} | {h['filename']} (dist={h['distance']:.3f})"):
                st.code(h["document"], language="python")
                st.markdown(h["comment"])


if __name__ == "__main__":
    main()
