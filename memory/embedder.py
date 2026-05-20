"""Google Generative AI text embedding wrapper.

All embedding calls go through this module so there is one place to:
  - Manage the API client lifecycle (lazy singleton)
  - Wrap transient failures in EmbedderError
  - Expose reset_client() for test isolation
  - Chunk oversized batches automatically

Constants
---------
EMBEDDING_DIM : int
    Output dimension for gemini-embedding-001 (3072).  ChromaDB collection
    creation can use this value to set dimensionality upfront.
"""

from __future__ import annotations

import logging
from typing import Final

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM: Final[int] = 3072       # gemini-embedding-001 default output size
_MAX_BATCH: Final[int] = 100           # conservative batch limit for Google API

_client: GoogleGenerativeAIEmbeddings | None = None


# ── Error type ─────────────────────────────────────────────────────────────────


class EmbedderError(RuntimeError):
    """Raised when a Google embedding call fails."""


# ── Client management ──────────────────────────────────────────────────────────


def _get_client() -> GoogleGenerativeAIEmbeddings:
    """Return the cached embeddings client, creating it on first call."""
    global _client
    if _client is None:
        _client = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )
    return _client


def reset_client() -> None:
    """Discard the cached client.

    Useful in tests to inject a fresh API key or mock client without
    restarting the process.
    """
    global _client
    _client = None


# ── Public API ─────────────────────────────────────────────────────────────────


def embed_text(text: str) -> list[float]:
    """Embed a single string using the configured Google model.

    Args:
        text: Text to embed.

    Returns:
        Dense float vector of length ``EMBEDDING_DIM`` (768).

    Raises:
        EmbedderError: If the API call fails for any reason.
    """
    try:
        return _get_client().embed_query(text)
    except Exception as exc:
        raise EmbedderError(f"embed_text failed: {exc}") from exc


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, chunking automatically at ``_MAX_BATCH``.

    Args:
        texts: Strings to embed.  An empty list returns ``[]`` immediately.

    Returns:
        Vectors in the same order as the input.

    Raises:
        EmbedderError: If any batch call fails.
    """
    if not texts:
        return []

    results: list[list[float]] = []
    for offset in range(0, len(texts), _MAX_BATCH):
        chunk = texts[offset : offset + _MAX_BATCH]
        try:
            chunk_vectors = _get_client().embed_documents(chunk)
            results.extend(chunk_vectors)
        except Exception as exc:
            raise EmbedderError(
                f"embed_batch failed at offset {offset}: {exc}"
            ) from exc

    return results
