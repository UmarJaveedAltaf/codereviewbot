"""Shared pytest fixtures for CodeReviewBot tests."""

import json
from pathlib import Path

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
