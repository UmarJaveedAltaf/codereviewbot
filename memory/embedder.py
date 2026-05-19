from __future__ import annotations

from openai import OpenAI

from config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed_text(text: str) -> list[float]:
    """Embed a single string using the configured OpenAI embedding model.

    Args:
        text: The text to embed. Long texts are silently truncated by the API.

    Returns:
        Dense float vector.
    """
    response = _get_client().embeddings.create(
        input=text,
        model=settings.EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings in a single API call.

    Args:
        texts: List of strings. Empty list returns empty list.

    Returns:
        List of float vectors, same order as input.
    """
    if not texts:
        return []
    response = _get_client().embeddings.create(
        input=texts,
        model=settings.EMBEDDING_MODEL,
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
