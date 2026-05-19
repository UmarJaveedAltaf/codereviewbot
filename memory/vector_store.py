"""Two-collection ChromaDB vector store for CodeReviewBot.

Collections
-----------
past_reviews
    One document per review comment posted.
    document  : the code snippet that was reviewed
    metadata  : file, line, severity, category, accepted, repo, timestamp, comment

team_conventions
    One document per team coding rule.
    document  : the human-readable rule text
    metadata  : repo, language, category, added_by

Acceptance tracking
-------------------
``accepted`` is stored as an integer because ChromaDB metadata supports only
str / int / float / bool (not None / null):

    -1  →  unrated (default)
     0  →  rejected by the developer
     1  →  accepted / acted upon

``get_acceptance_rate()`` excludes unrated entries from the denominator so it
measures "of the reviews developers evaluated, what fraction did they agree with?"

Test isolation
--------------
``_get_chroma_client()`` is a named module-level function so tests can
monkeypatch it to return a ``chromadb.EphemeralClient()`` and the module-level
singletons ``_reviews_col`` / ``_conventions_col`` can be reset to ``None``
between tests.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from memory.embedder import embed_text

logger = logging.getLogger(__name__)

# ── Module-level singletons (patchable for tests) ─────────────────────────────

_chroma_client: chromadb.PersistentClient | None = None
_reviews_col = None
_conventions_col = None

_REVIEWS_COLLECTION = "past_reviews"
_CONVENTIONS_COLLECTION = "team_conventions"


# ── Public data models ─────────────────────────────────────────────────────────


@dataclass
class ReviewMetadata:
    """Metadata attached to a stored review comment."""

    file: str
    line: int = 0
    severity: str = "WARNING"      # CRITICAL | WARNING | SUGGESTION
    category: str = "general"      # security | style | performance | error_handling | testing
    repo: str = ""
    timestamp: str = ""            # ISO-8601; auto-filled when empty
    accepted: int = -1             # -1 = unrated, 0 = rejected, 1 = accepted


@dataclass
class ReviewResult:
    """A past review comment retrieved by similarity search."""

    review_id: str
    code_snippet: str
    comment: str
    file: str
    line: int
    severity: str
    category: str
    accepted: int                  # -1, 0, or 1
    repo: str
    timestamp: str
    distance: float                # cosine distance — lower is more similar


@dataclass
class ConventionMetadata:
    """Metadata attached to a team coding convention."""

    repo: str = ""                 # "" = applies to all repos
    language: str = ""             # "" = applies to all languages
    category: str = "general"      # security | python_style | error_handling | performance | testing
    added_by: str = "system"


# ── ChromaDB client / collection accessors ────────────────────────────────────


def _get_chroma_client():
    """Return the shared ChromaDB client, creating it lazily.

    Client selection
    ----------------
    When the ``CHROMA_HOST`` environment variable is set (e.g. in Docker where
    a standalone ChromaDB container is running) an ``HttpClient`` is returned
    so all requests go over HTTP to that service.

    When ``CHROMA_HOST`` is absent (local development, tests) a
    ``PersistentClient`` is returned, writing to ``CHROMA_PERSIST_DIR``.

    Exposed as a named function so tests can monkeypatch it to return a
    ``chromadb.EphemeralClient()`` for full in-memory isolation.
    """
    global _chroma_client
    if _chroma_client is None:
        chroma_host = os.environ.get("CHROMA_HOST", "")
        if chroma_host:
            chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
            logger.info("ChromaDB: connecting to HTTP server %s:%d", chroma_host, chroma_port)
            _chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
            )
        else:
            logger.debug("ChromaDB: using PersistentClient at %s", settings.CHROMA_PERSIST_DIR)
            _chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
    return _chroma_client


def _get_reviews_collection():
    """Return the ``past_reviews`` collection, creating it if needed."""
    global _reviews_col
    if _reviews_col is None:
        _reviews_col = _get_chroma_client().get_or_create_collection(
            name=_REVIEWS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _reviews_col


def _get_conventions_collection():
    """Return the ``team_conventions`` collection, creating it if needed."""
    global _conventions_col
    if _conventions_col is None:
        _conventions_col = _get_chroma_client().get_or_create_collection(
            name=_CONVENTIONS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _conventions_col


def _safe_n(col, where_filter: dict | None, requested: int) -> int:
    """Return ``min(requested, available)`` where *available* respects the filter.

    ChromaDB raises if ``n_results`` exceeds the number of matching documents.
    This helper pre-counts the filtered subset so callers never hit that error.
    """
    if where_filter:
        matching = col.get(where=where_filter, include=[])
        available = len(matching.get("ids", []))
    else:
        available = col.count()
    return min(requested, available)


# ── past_reviews public API ────────────────────────────────────────────────────


def add_review(
    code_snippet: str,
    comment: str,
    metadata: ReviewMetadata,
) -> str:
    """Store a review comment and its code-snippet embedding.

    Args:
        code_snippet: The code fragment that was reviewed (used as the document
                      and as the source for the embedding vector).
        comment:      The review comment text generated by the LLM.
        metadata:     Structured metadata; ``timestamp`` is auto-set if empty.

    Returns:
        The UUID string assigned to this review (usable with
        ``mark_review_accepted`` / ``mark_review_rejected``).
    """
    if not metadata.timestamp:
        metadata.timestamp = datetime.now(timezone.utc).isoformat()

    review_id = str(uuid.uuid4())
    col = _get_reviews_collection()
    embedding = embed_text(code_snippet)

    col.upsert(
        ids=[review_id],
        embeddings=[embedding],
        documents=[code_snippet],
        metadatas=[{
            "comment": comment,
            "file": metadata.file,
            "line": metadata.line,
            "severity": metadata.severity,
            "category": metadata.category,
            "accepted": metadata.accepted,
            "repo": metadata.repo,
            "timestamp": metadata.timestamp,
        }],
    )
    logger.debug("Stored review %s for %s / %s", review_id[:8], metadata.repo, metadata.file)
    return review_id


def get_similar_reviews(
    code_snippet: str,
    repo: str = "",
    n: int = 5,
) -> list[ReviewResult]:
    """Retrieve the most semantically similar past review comments.

    When *repo* is provided, results are scoped to that repository so the
    context in prompts reflects this team's conventions rather than reviews
    from unrelated codebases.

    Args:
        code_snippet: Code to find similar past reviews for.
        repo:         Restrict to this repo (``"owner/name"``).  Pass ``""``
                      to search across all repos.
        n:            Maximum number of results.

    Returns:
        List of ``ReviewResult`` ordered by ascending cosine distance (most
        similar first).  Empty list when the store is empty or no matches exist.
    """
    col = _get_reviews_collection()
    where_filter = {"repo": repo} if repo else None

    available = _safe_n(col, where_filter, n)
    if available == 0:
        return []

    embedding = embed_text(code_snippet)
    results = col.query(
        query_embeddings=[embedding],
        n_results=available,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    output: list[ReviewResult] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(ReviewResult(
            review_id=results["ids"][0][len(output)],
            code_snippet=doc,
            comment=meta["comment"],
            file=meta.get("file", ""),
            line=int(meta.get("line", 0)),
            severity=meta.get("severity", "WARNING"),
            category=meta.get("category", "general"),
            accepted=int(meta.get("accepted", -1)),
            repo=meta.get("repo", ""),
            timestamp=meta.get("timestamp", ""),
            distance=dist,
        ))
    return output


def mark_review_accepted(review_id: str) -> None:
    """Record that a developer accepted / acted on this review comment.

    Fetches the current metadata, patches the ``accepted`` field to ``1``,
    and re-upserts without touching the stored embedding or document.

    Args:
        review_id: UUID returned by ``add_review()``.
    """
    _set_accepted(review_id, 1)


def mark_review_rejected(review_id: str) -> None:
    """Record that a developer dismissed / rejected this review comment.

    Args:
        review_id: UUID returned by ``add_review()``.
    """
    _set_accepted(review_id, 0)


def _set_accepted(review_id: str, value: int) -> None:
    col = _get_reviews_collection()
    result = col.get(ids=[review_id], include=["metadatas"])
    if not result.get("ids"):
        logger.warning("mark_review: review_id %s not found", review_id)
        return
    existing_meta = result["metadatas"][0]
    updated_meta = {**existing_meta, "accepted": value}
    col.update(ids=[review_id], metadatas=[updated_meta])
    logger.debug("Review %s marked accepted=%d", review_id[:8], value)


def get_acceptance_rate(repo: str) -> float:
    """Fraction of rated reviews for *repo* that were accepted.

    Unrated reviews (``accepted == -1``) are excluded from both numerator
    and denominator.  Returns ``0.0`` when no reviews exist or none are rated.

    Args:
        repo: ``"owner/name"`` repository identifier.

    Returns:
        Float in ``[0.0, 1.0]``.
    """
    col = _get_reviews_collection()
    where_filter: dict | None = {"repo": repo} if repo else None

    result = col.get(where=where_filter, include=["metadatas"])
    if not result.get("ids"):
        return 0.0

    rated = [m for m in result["metadatas"] if int(m.get("accepted", -1)) != -1]
    if not rated:
        return 0.0

    accepted_count = sum(1 for m in rated if int(m["accepted"]) == 1)
    return accepted_count / len(rated)


# ── team_conventions public API ───────────────────────────────────────────────


def add_convention(rule_text: str, metadata: ConventionMetadata) -> str:
    """Store a team coding convention for future retrieval.

    Args:
        rule_text: Human-readable description of the rule.
        metadata:  Scope and classification of the rule.

    Returns:
        UUID string assigned to this convention.
    """
    convention_id = str(uuid.uuid4())
    col = _get_conventions_collection()
    embedding = embed_text(rule_text)

    col.upsert(
        ids=[convention_id],
        embeddings=[embedding],
        documents=[rule_text],
        metadatas=[{
            "repo": metadata.repo,
            "language": metadata.language,
            "category": metadata.category,
            "added_by": metadata.added_by,
        }],
    )
    logger.debug("Stored convention %s [%s]", convention_id[:8], metadata.category)
    return convention_id


def get_relevant_conventions(
    code_snippet: str,
    language: str = "",
    n: int = 3,
) -> list[str]:
    """Retrieve the most relevant team conventions for a code snippet.

    Language filtering uses an ``$or`` so that language-agnostic conventions
    (``language == ""``) are always eligible alongside language-specific ones.

    Args:
        code_snippet: Code to find relevant conventions for.
        language:     Programming language of the snippet (``"python"``,
                      ``"javascript"``, etc.).  Pass ``""`` to fetch global
                      conventions only.
        n:            Maximum number of convention strings to return.

    Returns:
        List of rule-text strings, ordered by descending relevance.
        Empty list when the store is empty.
    """
    col = _get_conventions_collection()

    # Match exact language AND global (language="") conventions.
    if language:
        where_filter: dict | None = {
            "$or": [
                {"language": language},
                {"language": ""},
            ]
        }
    else:
        where_filter = None  # fetch all when no language specified

    available = _safe_n(col, where_filter, n)
    if available == 0:
        return []

    embedding = embed_text(code_snippet)
    results = col.query(
        query_embeddings=[embedding],
        n_results=available,
        where=where_filter,
        include=["documents"],
    )

    return results["documents"][0]


# ── Backward-compatibility shims ──────────────────────────────────────────────
# Keep the old ReviewMemory / store_review / query_similar names so any code
# that hasn't migrated yet continues to work.  New code should use add_review()
# and get_similar_reviews() directly.


@dataclass
class ReviewMemory:
    """Deprecated: use ReviewMetadata + add_review() instead."""

    id: str
    code_snippet: str
    comment: str
    repo: str
    pr_number: int
    filename: str


def store_review(memory: ReviewMemory) -> None:
    """Deprecated shim around add_review().  Will be removed in a future version."""
    add_review(
        code_snippet=memory.code_snippet,
        comment=memory.comment,
        metadata=ReviewMetadata(
            file=memory.filename,
            repo=memory.repo,
        ),
    )


def query_similar(code_snippet: str, n_results: int = 5) -> list[dict]:
    """Deprecated shim around get_similar_reviews().  Returns raw dicts for
    backward compatibility with callers that index results by key name."""
    results = get_similar_reviews(code_snippet, repo="", n=n_results)
    return [
        {
            "document": r.code_snippet,
            "comment": r.comment,
            "repo": r.repo,
            "pr_number": 0,
            "filename": r.file,
            "distance": r.distance,
        }
        for r in results
    ]
