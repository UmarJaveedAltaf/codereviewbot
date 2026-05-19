"""GitHub webhook receiver for CodeReviewBot.

Entry point: POST /webhook/github

Flow:
    1.  HMAC-SHA256 signature validated by Depends(verify_github_signature).
    2.  Raw body decoded and validated against PullRequestPayload Pydantic model.
    3.  Non-PR events and irrelevant actions (merged, closed, …) are ignored and
        acknowledged with 200 so GitHub doesn't retry them.
    4.  Draft PRs are skipped (configurable).
    5.  A PRContext dataclass is built from the validated payload.
    6.  run_review(ctx) is handed off to FastAPI BackgroundTasks so the HTTP
        response returns immediately — GitHub expects a response within 10 s.
    7.  Every step is logged with structured key=value context for easy grep / log
        aggregation (no external logging library needed).

Pydantic models mirror the GitHub webhook schema for pull_request events:
    https://docs.github.com/en/webhooks/webhook-events-and-payloads#pull_request
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from agent.reviewer import run_review
from webhook.validator import verify_github_signature

logger = logging.getLogger(__name__)
router = APIRouter()

# Actions that should trigger a review.  Everything else is a no-op.
REVIEWABLE_ACTIONS: frozenset[str] = frozenset({"opened", "synchronize", "reopened"})


# ── Pydantic payload models ────────────────────────────────────────────────────


class GitHubUser(BaseModel):
    """Minimal GitHub user/actor object."""

    login: str
    id: int
    type: str = "User"


class GitHubRepoOwner(BaseModel):
    login: str
    id: int = 0


class GitHubRepository(BaseModel):
    """Relevant fields from the repository object in a webhook payload."""

    id: int
    name: str
    full_name: str
    private: bool = False
    html_url: str
    owner: GitHubRepoOwner
    default_branch: str = "main"


class GitHubCommitRef(BaseModel):
    """One side of a pull request comparison (head or base)."""

    label: str          # "owner:branch-name"
    ref: str            # "branch-name"
    sha: str            # full 40-char commit SHA

    @field_validator("sha")
    @classmethod
    def sha_looks_valid(cls, v: str) -> str:
        if len(v) < 7:
            raise ValueError(f"SHA too short to be valid: {v!r}")
        return v.lower()


class GitHubPullRequest(BaseModel):
    """Fields we care about from a pull_request webhook object."""

    number: int
    title: str
    body: str | None = Field(default=None)
    state: str                             # "open" | "closed"
    draft: bool = False
    html_url: str
    user: GitHubUser                       # PR author
    head: GitHubCommitRef
    base: GitHubCommitRef
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0

    @property
    def author(self) -> str:
        return self.user.login

    @property
    def description(self) -> str:
        """Non-null body, stripped, capped at 1 000 chars."""
        return (self.body or "").strip()[:1_000]


class PullRequestPayload(BaseModel):
    """Top-level pull_request webhook payload."""

    action: str
    number: int
    pull_request: GitHubPullRequest
    repository: GitHubRepository
    sender: GitHubUser                     # actor who triggered the event

    @property
    def repo_full_name(self) -> str:
        return self.repository.full_name

    @property
    def pr_number(self) -> int:
        """Alias for .number — mirrors the PR-centric naming used everywhere else."""
        return self.number


# ── Internal context object passed to background tasks ─────────────────────────


@dataclass
class PRContext:
    """Immutable snapshot of everything the reviewer needs.

    Built once in the webhook handler and passed to run_review() so the
    background worker never has to touch the raw HTTP request.
    """

    # Identity
    review_id: str                # UUID for log correlation
    received_at: float            # time.monotonic() timestamp

    # PR metadata
    pr_number: int
    pr_title: str
    pr_body: str
    pr_author: str
    pr_url: str
    is_draft: bool

    # Repository
    repo_full_name: str           # "owner/repo"
    repo_owner: str
    repo_name: str
    repo_private: bool

    # Git refs
    base_sha: str
    head_sha: str
    base_ref: str                 # branch name
    head_ref: str

    # Stats from the PR object (may be 0 on synchronize events)
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0

    # Webhook provenance
    action: str = ""
    sender: str = ""
    event_id: str = ""            # X-GitHub-Delivery header

    def log_extra(self) -> dict:
        """Return a dict of structured fields safe to pass to logger.extra."""
        return {
            "review_id": self.review_id,
            "repo": self.repo_full_name,
            "pr": self.pr_number,
            "author": self.pr_author,
            "action": self.action,
            "head_sha": self.head_sha[:12],
            "base_sha": self.base_sha[:12],
            "is_draft": self.is_draft,
        }


# ── Route ──────────────────────────────────────────────────────────────────────


@router.post(
    "/github",
    status_code=status.HTTP_200_OK,
    summary="Receive GitHub pull_request webhook events",
    response_description="Acknowledgement; review runs asynchronously",
)
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    # HMAC validation — raises 401 on bad/missing signature; returns raw body
    raw_body: Annotated[bytes, Depends(verify_github_signature)],
    # GitHub headers (optional=True so FastAPI doesn't 422 on missing ones)
    x_github_event: Annotated[str | None, Header()] = None,
    x_github_delivery: Annotated[str | None, Header()] = None,
) -> dict:
    """Receive and dispatch GitHub webhook events.

    Returns 200 immediately for all valid, signed requests.
    Only pull_request events with action in {opened, synchronize, reopened}
    trigger a background review job.  Everything else is acknowledged and
    dropped so GitHub stops retrying.
    """
    delivery_id = x_github_delivery or "unknown"
    event_type = x_github_event or "unknown"

    log = logger.getChild("handler")
    log.info(
        "Webhook received",
        extra={"delivery_id": delivery_id, "event": event_type},
    )

    # ── 1. Filter by event type ──────────────────────────────────────────────
    if event_type == "ping":
        log.info("GitHub ping acknowledged", extra={"delivery_id": delivery_id})
        return {"status": "ok", "event": "ping"}

    if event_type != "pull_request":
        log.debug(
            "Ignoring non-PR event",
            extra={"event": event_type, "delivery_id": delivery_id},
        )
        return {"status": "ignored", "event": event_type, "delivery_id": delivery_id}

    # ── 2. Decode and validate payload ───────────────────────────────────────
    try:
        raw_dict: dict = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        log.error("Malformed JSON in webhook body: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON: {exc}",
        ) from exc

    try:
        payload = PullRequestPayload.model_validate(raw_dict)
    except Exception as exc:
        # Pydantic validation error — log the action so we can add models later
        action = raw_dict.get("action", "?")
        log.warning(
            "Payload validation failed for action=%r: %s",
            action,
            exc,
            extra={"delivery_id": delivery_id},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Payload schema error: {exc}",
        ) from exc

    pr = payload.pull_request
    repo = payload.repository

    # ── 3. Filter by action ──────────────────────────────────────────────────
    if payload.action not in REVIEWABLE_ACTIONS:
        log.info(
            "Action %r is not reviewable — skipping",
            payload.action,
            extra={
                "action": payload.action,
                "repo": repo.full_name,
                "pr": pr.number,
                "delivery_id": delivery_id,
            },
        )
        return {
            "status": "ignored",
            "reason": f"action '{payload.action}' does not trigger review",
            "pr": pr.number,
            "repo": repo.full_name,
        }

    # ── 4. Skip draft PRs ────────────────────────────────────────────────────
    if pr.draft:
        log.info(
            "PR #%d is a draft — skipping review",
            pr.number,
            extra={"repo": repo.full_name, "pr": pr.number, "delivery_id": delivery_id},
        )
        return {
            "status": "ignored",
            "reason": "draft PR",
            "pr": pr.number,
            "repo": repo.full_name,
        }

    # ── 5. Build PRContext ───────────────────────────────────────────────────
    review_id = str(uuid.uuid4())
    ctx = PRContext(
        review_id=review_id,
        received_at=time.monotonic(),
        pr_number=pr.number,
        pr_title=pr.title,
        pr_body=pr.description,
        pr_author=pr.author,
        pr_url=pr.html_url,
        is_draft=pr.draft,
        repo_full_name=repo.full_name,
        repo_owner=repo.owner.login,
        repo_name=repo.name,
        repo_private=repo.private,
        base_sha=pr.base.sha,
        head_sha=pr.head.sha,
        base_ref=pr.base.ref,
        head_ref=pr.head.ref,
        additions=pr.additions,
        deletions=pr.deletions,
        changed_files=pr.changed_files,
        action=payload.action,
        sender=payload.sender.login,
        event_id=delivery_id,
    )

    log.info(
        "Queuing review",
        extra=ctx.log_extra() | {"delivery_id": delivery_id},
    )

    # ── 6. Dispatch background task ─────────────────────────────────────────
    background_tasks.add_task(_review_task_wrapper, ctx)

    return {
        "status": "queued",
        "review_id": review_id,
        "pr": pr.number,
        "repo": repo.full_name,
        "action": payload.action,
        "head_sha": pr.head.sha[:12],
    }


# ── Background task wrapper ────────────────────────────────────────────────────


def _review_task_wrapper(ctx: PRContext) -> None:
    """Thin wrapper around run_review for error isolation and timing logs.

    Catches all exceptions so a failing review never crashes the worker
    process.  Logs duration so you can spot expensive reviews in your logs.
    """
    start = time.monotonic()
    log = logger.getChild("background")

    log.info("Review task started", extra=ctx.log_extra())

    try:
        run_review(ctx)
    except Exception:
        log.exception(
            "Unhandled error in review task",
            extra=ctx.log_extra(),
        )
    finally:
        elapsed = round(time.monotonic() - start, 2)
        log.info(
            "Review task finished",
            extra=ctx.log_extra() | {"elapsed_s": elapsed},
        )
