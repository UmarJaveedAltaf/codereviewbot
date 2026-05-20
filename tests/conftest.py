"""Shared pytest fixtures for CodeReviewBot tests."""

import hashlib
import json
from pathlib import Path
from typing import Generator

import chromadb
import pytest

from config import settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# The HMAC secret must be identical between the test's _sign() helper and the
# validator.  We pin it here with autouse=True so every test in the suite gets
# it automatically without each test needing its own patch() call.
# The value "test-secret" is also the default used by _sign() in
# test_webhook.py, so the two stay in sync without further coordination.
# ---------------------------------------------------------------------------
TEST_WEBHOOK_SECRET = "test-secret"


@pytest.fixture(autouse=True)
def _patch_webhook_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin GITHUB_WEBHOOK_SECRET to a known value for every test."""
    monkeypatch.setattr(settings, "GITHUB_WEBHOOK_SECRET", TEST_WEBHOOK_SECRET)


@pytest.fixture()
def mock_pr_payload() -> dict:
    """Full GitHub pull_request webhook payload that satisfies PullRequestPayload."""
    return {
        "action": "opened",
        "number": 42,
        "pull_request": {
            "number": 42,
            "title": "Add divide function",
            "body": "Implements safe integer division with a zero-guard.",
            "state": "open",
            "draft": False,
            "html_url": "https://github.com/testorg/testrepo/pull/42",
            "user": {"login": "dev-user", "id": 1, "type": "User"},
            "head": {
                "label": "testorg:feature/divide",
                "ref": "feature/divide",
                "sha": "abc123def456abc123def456abc123def456abc1",
            },
            "base": {
                "label": "testorg:main",
                "ref": "main",
                "sha": "000000aabbcc000000aabbcc000000aabbcc0000",
            },
            "additions": 15,
            "deletions": 2,
            "changed_files": 1,
        },
        "repository": {
            "id": 12345,
            "name": "testrepo",
            "full_name": "testorg/testrepo",
            "private": False,
            "html_url": "https://github.com/testorg/testrepo",
            "owner": {"login": "testorg", "id": 99},
            "default_branch": "main",
        },
        "sender": {"login": "dev-user", "id": 1, "type": "User"},
    }


@pytest.fixture()
def sample_python_diff() -> str:
    return (
        "@@ -1,5 +1,10 @@\n"
        "+def divide(a, b):\n"
        "+    if b == 0:\n"
        "+        raise ValueError('division by zero')\n"
        "+    return a / b\n"
        "+\n"
        " def add(x, y):\n"
        "     return x + y\n"
    )


@pytest.fixture()
def sample_changed_file() -> dict:
    return {
        "filename": "math_utils.py",
        "patch": (
            "@@ -1,5 +1,10 @@\n"
            "+def divide(a, b):\n"
            "+    return a / b\n"
            "+\n"
            " def add(x, y):\n"
            "     return x + y\n"
        ),
        "status": "modified",
        "additions": 3,
        "deletions": 0,
    }


# ── Vector store fixture ───────────────────────────────────────────────────────


def _fake_embed(text: str) -> list[float]:
    """Deterministic 8-dimensional fake embedding derived from the text's MD5.

    Dimension 8 is far smaller than the real 768 (text-embedding-004) but
    ChromaDB accepts any consistent dimensionality, making tests fast without
    mocking the entire Google API client.
    """
    digest = hashlib.md5(text.encode()).digest()  # 16 bytes
    return [b / 255.0 for b in digest[:8]]


@pytest.fixture()
def mem(monkeypatch: pytest.MonkeyPatch):
    """Isolated in-memory vector store with deterministic fake embeddings.

    Isolation strategy
    ------------------
    Rather than patching the ChromaDB client (where EphemeralClient instances
    can share an in-memory SQLite backend across instances in chromadb 0.5.x),
    we patch ``_get_reviews_collection`` and ``_get_conventions_collection``
    directly.  Each test gets collections with a UUID-suffixed name on a fresh
    EphemeralClient, guaranteeing zero state leakage between tests.

    Patches applied
    ---------------
    - ``_get_reviews_collection``    → fresh isolated collection
    - ``_get_conventions_collection`` → fresh isolated collection
    - ``embed_text``                  → ``_fake_embed`` (MD5-based, 8-dim, no API)

    Yields the ``memory.vector_store`` module so tests call public functions
    directly::

        def test_foo(mem):
            rid = mem.add_review("code", "comment",
                                 mem.ReviewMetadata(file="f.py", repo="r"))
            assert rid
    """
    import uuid
    from memory import vector_store as vs

    client = chromadb.EphemeralClient()
    uid = uuid.uuid4().hex[:8]   # unique suffix prevents collection name reuse

    # Cache so repeated calls within the same test return the same object.
    _cache: dict[str, object] = {}

    def _get_reviews():
        if "r" not in _cache:
            _cache["r"] = client.get_or_create_collection(
                f"reviews_{uid}", metadata={"hnsw:space": "cosine"}
            )
        return _cache["r"]

    def _get_conventions():
        if "c" not in _cache:
            _cache["c"] = client.get_or_create_collection(
                f"conventions_{uid}", metadata={"hnsw:space": "cosine"}
            )
        return _cache["c"]

    monkeypatch.setattr(vs, "_get_reviews_collection", _get_reviews)
    monkeypatch.setattr(vs, "_get_conventions_collection", _get_conventions)
    monkeypatch.setattr(vs, "embed_text", _fake_embed)

    yield vs
