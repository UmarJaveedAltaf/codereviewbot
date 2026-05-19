from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from memory.embedder import embed_text

logger = logging.getLogger(__name__)


@dataclass
class ReviewMemory:
    id: str
    code_snippet: str
    comment: str
    repo: str
    pr_number: int
    filename: str


_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def store_review(memory: ReviewMemory) -> None:
    """Persist a review comment alongside its code snippet embedding.

    Args:
        memory: ReviewMemory dataclass with all required fields.
    """
    col = _get_collection()
    embedding = embed_text(memory.code_snippet)
    col.upsert(
        ids=[memory.id],
        embeddings=[embedding],
        documents=[memory.code_snippet],
        metadatas=[{
            "comment": memory.comment,
            "repo": memory.repo,
            "pr_number": memory.pr_number,
            "filename": memory.filename,
        }],
    )
    logger.debug("Stored review memory %s", memory.id)


def query_similar(code_snippet: str, n_results: int = 5) -> list[dict]:
    """Retrieve the most similar past review comments for a code snippet.

    Args:
        code_snippet: The code fragment to find similar reviews for.
        n_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: document, comment, repo, pr_number, filename, distance.
    """
    col = _get_collection()
    if col.count() == 0:
        return []

    embedding = embed_text(code_snippet)
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(n_results, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "document": doc,
            "comment": meta["comment"],
            "repo": meta["repo"],
            "pr_number": meta["pr_number"],
            "filename": meta["filename"],
            "distance": dist,
        })
    return output
