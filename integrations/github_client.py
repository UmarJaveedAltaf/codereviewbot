"""PyGithub wrapper for all GitHub API operations.

Every function is stateless and accepts plain Python values so callers
(especially tests) never need to touch a Github object directly.

Authentication uses a personal access token (GITHUB_TOKEN) with:
    repo scope          — read repo content, post review comments
    pull_requests scope — read PR metadata and files

New in Phase 5
--------------
post_review_comments  Post all line-level findings as a single GitHub Review.
post_pr_summary       Post a formatted severity-table summary comment.
mark_comment_resolved Record developer feedback on a specific review comment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from github import Github, GithubException
from github.Commit import Commit
from github.Repository import Repository

from config import settings

if TYPE_CHECKING:
    from agent.reviewer import FileReview

logger = logging.getLogger(__name__)

# Module-level singleton — reuse the connection pool across requests.
_github: Github | None = None

# Severity ordering for the summary table (worst first).
_SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]
_SEVERITY_ICONS = {
    "critical": "🚨",
    "high":     "❗",
    "medium":   "⚠️",
    "low":      "💡",
    "info":     "ℹ️",
}


# ── Client / repo accessors ───────────────────────────────────────────────────


def _get_github() -> Github:
    """Return (or create) the shared PyGithub client."""
    global _github
    if _github is None:
        _github = Github(
            login_or_token=settings.GITHUB_TOKEN,
            per_page=100,
            retry=3,
        )
    return _github


def _get_repo(repo_full_name: str) -> Repository:
    """Fetch a Repository object, raising RuntimeError on unknown repos."""
    try:
        return _get_github().get_repo(repo_full_name)
    except GithubException as exc:
        raise RuntimeError(
            f"Could not fetch repo {repo_full_name!r}: {exc}"
        ) from exc


# ── Diff / file fetching ──────────────────────────────────────────────────────


def get_pr_diff(repo_full_name: str, base_sha: str, head_sha: str) -> list[dict]:
    """Fetch the full diff between two commits using ``repo.compare()``.

    Preferred over ``get_pr_files()`` because ``compare()`` returns patch text
    even when the PR has more than 300 changed files (the PR files API limit).

    Args:
        repo_full_name: ``"owner/repo"``
        base_sha:       Base commit SHA (PR target branch tip).
        head_sha:       Head commit SHA (PR source branch tip).

    Returns:
        List of dicts with keys: ``filename``, ``patch``, ``status``,
        ``additions``, ``deletions``.
    """
    repo = _get_repo(repo_full_name)
    try:
        comparison = repo.compare(base_sha, head_sha)
    except GithubException as exc:
        logger.error(
            "repo.compare() failed for %s %s..%s: %s",
            repo_full_name, base_sha[:12], head_sha[:12], exc,
        )
        raise

    files: list[dict] = []
    for f in comparison.files:
        files.append({
            "filename":  f.filename,
            "patch":     f.patch or "",
            "status":    f.status,
            "additions": f.additions,
            "deletions": f.deletions,
        })

    logger.info(
        "Fetched diff for %s %s..%s — %d file(s)",
        repo_full_name, base_sha[:12], head_sha[:12], len(files),
    )
    return files


def get_pr_files(repo_full_name: str, pr_number: int) -> tuple[list[dict], str]:
    """Return ``(changed_files, pr_title)`` via the PR files API.

    Returns changed files with their full patch text and the PR title.

    .. deprecated::
        Prefer :func:`get_pr_diff` which uses ``repo.compare()`` and avoids
        the 300-file pagination cap of the PR files endpoint.

    Args:
        repo_full_name: ``"owner/repo"``
        pr_number:      Pull request number.

    Returns:
        Tuple of ``(files, pr_title)`` where each file dict has keys:
        ``filename``, ``patch``, ``status``, ``additions``, ``deletions``.
    """
    repo = _get_repo(repo_full_name)
    pr   = repo.get_pull(pr_number)
    files = [
        {
            "filename":  f.filename,
            "patch":     f.patch or "",
            "status":    f.status,
            "additions": f.additions,
            "deletions": f.deletions,
        }
        for f in pr.get_files()
        if f.patch
    ]
    return files, pr.title


def get_pr_metadata(repo_full_name: str, pr_number: int) -> tuple[str, str]:
    """Return ``(title, body)`` for a PR.

    Useful when the webhook payload may be stale or incomplete.
    """
    repo = _get_repo(repo_full_name)
    pr   = repo.get_pull(pr_number)
    return pr.title, pr.body or ""


# ── Structured review submission ──────────────────────────────────────────────


def post_review_comments(
    repo_full_name: str,
    pr_number: int,
    commit_sha: str,
    file_review: "FileReview",
) -> None:
    """Post all line-level findings as a **single** GitHub Review submission.

    Groups every ``LineComment`` from *file_review* into one
    ``PullRequest.create_review()`` call so GitHub shows them as a cohesive
    review thread rather than scattered inline comments.

    Review event selection:
        ``REQUEST_CHANGES`` — when any finding has ``severity == "critical"``
        ``COMMENT``          — for all other cases (informational review)

    Args:
        repo_full_name: ``"owner/repo"``
        pr_number:      Pull request number.
        commit_sha:     Head commit SHA used for diff positioning.
        file_review:    Structured ``FileReview`` from the LLM agent.
    """
    has_critical = any(c.severity == "critical" for c in file_review.comments)
    event = "REQUEST_CHANGES" if has_critical else "COMMENT"

    inline_comments: list[dict] = []
    for c in file_review.comments:
        body_lines = [
            f"**{_SEVERITY_ICONS.get(c.severity, '•')} [{c.severity.upper()}]** "
            f"`{c.category}`",
            "",
            c.issue,
            "",
            f"**Suggestion:** {c.suggestion}",
        ]
        if c.code_fix:
            body_lines += ["", f"```\n{c.code_fix}\n```"]
        inline_comments.append({
            "path": file_review.filename,
            "line": c.line,
            "side": "RIGHT",
            "body": "\n".join(body_lines),
        })

    review_body = (
        f"## 🤖 CodeReviewBot — `{file_review.filename}`\n\n"
        f"{file_review.summary}"
    )

    try:
        repo   = _get_repo(repo_full_name)
        commit: Commit = repo.get_commit(commit_sha)
        pr     = repo.get_pull(pr_number)
        pr.create_review(
            commit=commit,
            body=review_body,
            event=event,
            comments=inline_comments,
        )
        logger.info(
            "Posted %s review on %s PR #%d / %s (%d inline comment(s))",
            event, repo_full_name, pr_number,
            file_review.filename, len(inline_comments),
        )
    except GithubException as exc:
        logger.error(
            "Failed to post review on %s PR #%d / %s: %s",
            repo_full_name, pr_number, file_review.filename, exc,
        )


def post_pr_summary(
    repo_full_name: str,
    pr_number: int,
    summary_text: str,
    file_reviews: list | None = None,
) -> None:
    """Post the overall PR review summary as an issue comment.

    When *file_reviews* is supplied, prepends a Markdown severity-breakdown
    table so reviewers get a quick at-a-glance count before reading details.

    Severity table format::

        | Severity     | Count |
        |--------------|-------|
        | 🚨 Critical  |     1 |
        | ❗ High      |     3 |
        | ⚠️ Medium    |     2 |

    Args:
        repo_full_name: ``"owner/repo"``
        pr_number:      Pull request number.
        summary_text:   Markdown body from the LLM summarizer or fallback.
        file_reviews:   Optional list of ``FileReview`` objects used to build
                        the severity count table.
    """
    body_parts: list[str] = []

    if file_reviews:
        counts: dict[str, int] = {s: 0 for s in _SEVERITY_ORDER}
        for fr in file_reviews:
            for c in fr.comments:
                if c.severity in counts:
                    counts[c.severity] += 1

        if any(v > 0 for v in counts.values()):
            table_rows = ["| Severity | Count |", "|----------|-------|"]
            for sev in _SEVERITY_ORDER:
                if counts[sev] > 0:
                    icon = _SEVERITY_ICONS[sev]
                    table_rows.append(
                        f"| {icon} {sev.title()} | {counts[sev]} |"
                    )
            body_parts.append("\n".join(table_rows))

    body_parts.append(summary_text)
    full_body = "\n\n".join(body_parts)

    try:
        repo = _get_repo(repo_full_name)
        pr   = repo.get_pull(pr_number)
        pr.create_issue_comment(full_body)
        logger.info(
            "Posted PR summary on %s PR #%d", repo_full_name, pr_number
        )
    except GithubException as exc:
        logger.error(
            "Failed to post summary on %s PR #%d: %s",
            repo_full_name, pr_number, exc,
        )


def mark_comment_resolved(
    repo_full_name: str,
    pr_number: int,
    comment_id: int,
) -> None:
    """Record that a developer resolved / dismissed a review comment.

    GitHub's REST API does not expose a direct "mark resolved" endpoint for
    pull-request review comments (that action is only available via GraphQL).
    This function logs the feedback so it can be used for acceptance-rate
    analytics and optionally persists the signal via the vector store.

    Args:
        repo_full_name: ``"owner/repo"``
        pr_number:      Pull request number.
        comment_id:     GitHub review comment ID (from the webhook payload or
                        the comment URL).
    """
    logger.info(
        "Comment resolved — %s PR #%d comment_id=%d",
        repo_full_name, pr_number, comment_id,
    )
    # Future: call mark_review_accepted(review_id) on the associated vector
    # store entry once the comment_id → review_id mapping is stored.


# ── Legacy single-file comment helpers ───────────────────────────────────────


def post_review_comment(
    repo_full_name: str,
    pr_number: int,
    commit_sha: str,
    filename: str,
    body: str,
) -> None:
    """Post a single pull-request review (COMMENT event) on a specific file.

    Uses ``create_review()`` so the comment appears as a review rather than
    a standalone comment.  For line-level structured output prefer
    :func:`post_review_comments` (plural).

    Args:
        repo_full_name: ``"owner/repo"``
        pr_number:      PR number.
        commit_sha:     Head commit SHA.
        filename:       File path within the repo.
        body:           Markdown comment body.
    """
    try:
        repo   = _get_repo(repo_full_name)
        commit: Commit = repo.get_commit(commit_sha)
        pr     = repo.get_pull(pr_number)
        pr.create_review(commit=commit, body=body, event="COMMENT")
        logger.info(
            "Posted review comment on %s PR #%d / %s",
            repo_full_name, pr_number, filename,
        )
    except GithubException as exc:
        logger.error(
            "Failed to post review comment on %s PR #%d: %s",
            repo_full_name, pr_number, exc,
        )


def post_summary_comment(
    repo_full_name: str,
    pr_number: int,
    body: str,
) -> None:
    """Post the overall review summary as a PR issue comment (timeline entry).

    Args:
        repo_full_name: ``"owner/repo"``
        pr_number:      PR number.
        body:           Markdown summary body.
    """
    try:
        repo = _get_repo(repo_full_name)
        pr   = repo.get_pull(pr_number)
        pr.create_issue_comment(body)
        logger.info(
            "Posted summary comment on %s PR #%d", repo_full_name, pr_number
        )
    except GithubException as exc:
        logger.error(
            "Failed to post summary comment on %s PR #%d: %s",
            repo_full_name, pr_number, exc,
        )
