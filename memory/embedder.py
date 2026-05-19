"""OpenAI text embedding wrapper.

All embedding calls go through this module so there is one place to:
  - Manage the API client lifecycle (lazy singleton)
  - Wrap transient failures in EmbedderError
  - Expose reset_client() for test isolation
  - Chunk oversized batches automatically

Constants
---------
EMBEDDING_DIM : int
    Output dimension for text-embedding-3-small (1536).  ChromaDB collection
    creation can use this value to set dimensionality upfront.
"""

from __future__ import annotations

import logging
from typing import Final

from openai import OpenAI, OpenAIError

from config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM: Final[int] = 1536       # text-embedding-3-small output size
_MAX_BATCH: Final[int] = 2_048         # OpenAI hard batch limit

_client: OpenAI | None = None


# ── Error type ─────────────────────────────────────────────────────────────────


class EmbedderError(RuntimeError):
    """Raised when an OpenAI embedding call fails."""


# ── Client management ──────────────────────────────────────────────────────────


def _get_client() -> OpenAI:
    """Return the cached OpenAI client, creating it on first call."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
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
    """Embed a single string using the configured OpenAI model.

    Args:
        text: Text to embed.  Texts longer than ~8 192 tokens are silently
              truncated by the API on the server side.

    Returns:
        Dense float vector of length ``EMBEDDING_DIM``.

    Raises:
        EmbedderError: If the API call fails for any reason.
    """
    try:
        response = _get_client().embeddings.create(
            input=text,
            model=settings.EMBEDDING_MODEL,
        )
        return response.data[0].embedding
    except OpenAIError as exc:
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
            response = _get_client().embeddings.create(
                input=chunk,
                model=settings.EMBEDDING_MODEL,
            )
            chunk_vectors = [
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            ]
            results.extend(chunk_vectors)
        except OpenAIError as exc:
            raise EmbedderError(
                f"embed_batch failed at offset {offset}: {exc}"
            ) from exc

    return results
