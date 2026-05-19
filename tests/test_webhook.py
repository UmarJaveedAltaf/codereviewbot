"""Tests for webhook/handler.py and webhook/validator.py."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import settings
from webhook.handler import (
    GitHubCommitRef,
    GitHubPullRequest,
    GitHubRepository,
    GitHubRepoOwner,
    GitHubUser,
    PRContext,
    PullRequestPayload,
    REVIEWABLE_ACTIONS,
    router,
)

# ── Test app ──────────────────────────────────────────────────────────────────

app = FastAPI()
app.include_router(router, prefix="/webhook")
client = TestClient(app, raise_server_exceptions=False)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sign(body: bytes, secret: str = "test-secret") -> str:
    digest = hmac.new(secret.encode(), msg=body, digestmod=hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _pr_payload(action: str = "opened", draft: bool = False) -> dict:
    return {
        "action": action,
        "number": 42,
        "pull_request": {
            "number": 42,
            "title": "Add divide function",
            "body": "Implements safe division.",
            "state": "open",
            "draft": draft,
            "html_url": "https://github.com/org/repo/pull/42",
            "user": {"login": "dev-user", "id": 1},
            "head": {"label": "org:feature", "ref": "feature", "sha": "abc123def456abc123def456abc123def456abc1"},
            "base": {"label": "org:main", "ref": "main", "sha": "000000aabbcc000000aabbcc000000aabbcc0000"},
            "additions": 20,
            "deletions": 3,
            "changed_files": 2,
        },
        "repository": {
            "id": 9999,
            "name": "repo",
            "full_name": "org/repo",
            "private": False,
            "html_url": "https://github.com/org/repo",
            "owner": {"login": "org", "id": 2},
            "default_branch": "main",
        },
        "sender": {"login": "dev-user", "id": 1, "type": "User"},
    }


def _post(payload: dict, secret: str = "test-secret", event: str = "pull_request"):
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": event,
            "X-GitHub-Delivery": "delivery-123",
            "X-Hub-Signature-256": _sign(body, secret),
        },
    )


# ── Pydantic model unit tests ─────────────────────────────────────────────────


class TestGitHubCommitRef:
    def test_sha_normalised_to_lowercase(self):
        ref = GitHubCommitRef(
            label="org:branch",
            ref="branch",
            sha="ABC123DEF456ABC123DEF456ABC123DEF456ABC1",
        )
        assert ref.sha == ref.sha.lower()

    def test_sha_too_short_raises(self):
        with pytest.raises(Exception):
            GitHubCommitRef(label="a:b", ref="b", sha="abc")


class TestPullRequestPayload:
    def test_valid_payload_parses(self, mock_pr_payload):
        payload = PullRequestPayload.model_validate(mock_pr_payload)
        assert payload.action == "opened"
        assert payload.pr_number == 42
        assert payload.repo_full_name == "testorg/testrepo"

    def test_pr_author_property(self, mock_pr_payload):
        payload = PullRequestPayload.model_validate(mock_pr_payload)
        assert payload.pull_request.author == "dev-user"

    def test_pr_description_strips_and_caps(self):
        raw = _pr_payload()
        raw["pull_request"]["body"] = "x" * 2000
        payload = PullRequestPayload.model_validate(raw)
        assert len(payload.pull_request.description) <= 1000


class TestPRContext:
    def test_log_extra_has_required_keys(self):
        import time
        import uuid
        ctx = PRContext(
            review_id=str(uuid.uuid4()),
            received_at=time.monotonic(),
            pr_number=1,
            pr_title="Test",
            pr_body="",
            pr_author="user",
            pr_url="https://github.com/x/y/pull/1",
            is_draft=False,
            repo_full_name="x/y",
            repo_owner="x",
            repo_name="y",
            repo_private=False,
            base_sha="a" * 40,
            head_sha="b" * 40,
            base_ref="main",
            head_ref="feature",
        )
        extra = ctx.log_extra()
        for key in ("review_id", "repo", "pr", "author", "action", "head_sha", "base_sha"):
            assert key in extra


# ── Handler endpoint tests ─────────────────────────────────────────────────────


class TestSignatureValidation:
    def test_missing_signature_returns_401(self):
        body = json.dumps(_pr_payload()).encode()
        resp = client.post(
            "/webhook/github",
            content=body,
            headers={"Content-Type": "application/json", "X-GitHub-Event": "pull_request"},
        )
        assert resp.status_code == 401

    def test_wrong_signature_returns_401(self):
        resp = _post(_pr_payload(), secret="wrong-secret")
        assert resp.status_code == 401

    def test_valid_signature_passes(self):
        with patch("webhook.handler.run_review"), \
             patch("config.settings.GITHUB_WEBHOOK_SECRET", "test-secret"):
            resp = _post(_pr_payload())
        assert resp.status_code == 200


class TestEventFiltering:
    def test_ping_event_acknowledged(self):
        body = json.dumps({"zen": "Keep it logically awesome."}).encode()
        resp = client.post(
            "/webhook/github",
            content=body,
            headers={
                "X-GitHub-Event": "ping",
                "X-Hub-Signature-256": _sign(body),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["event"] == "ping"

    def test_non_pr_event_ignored(self):
        body = json.dumps({"action": "created"}).encode()
        resp = client.post(
            "/webhook/github",
            content=body,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": _sign(body),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    @pytest.mark.parametrize("action", ["closed", "labeled", "assigned", "edited"])
    def test_non_reviewable_actions_ignored(self, action):
        with patch("webhook.handler.run_review"):
            resp = _post(_pr_payload(action=action))
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    @pytest.mark.parametrize("action", list(REVIEWABLE_ACTIONS))
    def test_reviewable_actions_queued(self, action):
        with patch("webhook.handler.run_review"):
            resp = _post(_pr_payload(action=action))
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    def test_draft_pr_skipped(self):
        with patch("webhook.handler.run_review"):
            resp = _post(_pr_payload(draft=True))
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"
        assert "draft" in resp.json()["reason"]


class TestQueuedResponse:
    def test_response_shape(self):
        with patch("webhook.handler.run_review"):
            resp = _post(_pr_payload())
        data = resp.json()
        assert data["status"] == "queued"
        assert "review_id" in data
        assert data["pr"] == 42
        assert data["repo"] == "org/repo"
        assert len(data["head_sha"]) == 12   # truncated for safety

    def test_background_task_receives_pr_context(self):
        captured = {}

        def fake_run(ctx):
            captured["ctx"] = ctx

        with patch("webhook.handler.run_review", side_effect=fake_run):
            _post(_pr_payload())

        # BackgroundTasks runs synchronously in TestClient
        assert "ctx" in captured
        ctx: PRContext = captured["ctx"]
        assert ctx.pr_number == 42
        assert ctx.repo_full_name == "org/repo"
        assert ctx.pr_author == "dev-user"
        assert ctx.action == "opened"
        assert ctx.base_ref == "main"
        assert ctx.head_ref == "feature"
        assert ctx.is_draft is False
