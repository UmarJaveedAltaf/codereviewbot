"""PyGithub wrapper for all GitHub API operations.

Every function is stateless and accepts plain Python values so callers
(especially tests) never need to touch a Github object directly.

Authentication uses a personal access token (GITHUB_TOKEN) with:
    repo scope — read repo content, post review comments
    pull_requests scope — read PR metadata and files
"""

from __future__ import annotations

import logging

from github import Github, GithubException
from github.Commit import Commit
from github.Repository import Repository

from config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — reuse the connection pool across requests.
_github: Github | None = None


def _get_github() -> Github:
    """Return (or create) the shared PyGithub client."""
    global _github
    if _github is None:
        _github = Github(
            login_or_token=settings.GITHUB_TOKEN,
            per_page=100,               # max files per page
            retry=3,                    # auto-retry transient errors
        )
    return _github


def _get_repo(repo_full_name: str) -> Repository:
    """Fetch a Repository object, raising RuntimeError on unknown repos."""
    try:
        return _get_github().get_repo(repo_full_name)
    except GithubException as exc:
        raise RuntimeError(f"Could not fetch repo {repo_full_name!r}: {exc}") from exc


# ── Diff / file fetching ──────────────────────────────────────────────────────


def get_pr_diff(repo_full_name: str, base_sha: str, head_sha: str) -> list[dict]:
    """Fetch the full diff between two commits using repo.compare().

    This is preferred over pr.get_files() because compare() gives us the
    actual patch text even when there are more than 300 changed files (the
    GitHub PR files API caps at 300).

    Args:
        repo_full_name: "owner/repo"
        base_sha:       Base commit SHA (the PR target branch tip).
        head_sha:       Head commit SHA (the PR source branch tip).

    Returns:
        List of dicts, each with keys:
            filename (str)   — repo-relative path
            patch    (str)   — unified diff text, empty string for binary files
            status   (str)   — "added" | "modified" | "removed" | "renamed"
            additions (int)
            deletions (int)
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
        patch = f.patch or ""          # None for binary files
        files.append({
            "filename": f.filename,
            "patch": patch,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
        })

    logger.info(
        "Fetched diff for %s %s..%s — %d file(s)",
        repo_full_name, base_sha[:12], head_sha[:12], len(files),
    )
    return files


def get_pr_metadata(repo_full_name: str, pr_number: int) -> tuple[str, str]:
    """Return (title, body) for a PR — useful when the webhook payload is stale.

    Args:
        repo_full_name: "owner/repo"
        pr_number:      PR number.

    Returns:
        (title, body) tuple.  body may be an empty string.
    """
    repo = _get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)
    return pr.title, pr.body or ""


# ── Kept for backward compatibility with existing callers ─────────────────────

def get_pr_files(repo_full_name: str, pr_number: int) -> tuple[list[dict], str]:
    """Fetch changed files and PR title via the PR files API.

    .. deprecated::
        Prefer get_pr_diff() which uses repo.compare() and avoids the
        300-file pagination cap of the PR files endpoint.

    Returns:
        (files, pr_title)
    """
    repo = _get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)
    files = [
        {
            "filename": f.filename,
            "patch": f.patch or "",
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
        }
        for f in pr.get_files()
        if f.patch  # skip binary / move-only entries
    ]
    return files, pr.title


# ── Posting comments ──────────────────────────────────────────────────────────


def post_review_comment(
    repo_full_name: str,
    pr_number: int,
    commit_sha: str,
    filename: str,
    body: str,
) -> None:
    """Post a pull request review (COMMENT type) on a specific file.

    Uses create_review() rather than create_review_comment() so all
    file-level findings are grouped under a single review thread.

    Args:
        repo_full_name: "owner/repo"
        pr_number:      PR number.
        commit_sha:     Head commit SHA for diff positioning.
        filename:       File path within the repo.
        body:           Markdown comment body.
    """
    try:
        repo = _get_repo(repo_full_name)
        commit: Commit = repo.get_commit(commit_sha)
        pr = repo.get_pull(pr_number)
        pr.create_review(
            commit=commit,
            body=body,
            event="COMMENT",
        )
        logger.info(
            "Posted review comment on %s PR #%d / %s",
            repo_full_name, pr_number, filename,
        )
    except GithubException as exc:
        logger.error(
            "Failed to post review comment on %s PR #%d: %s",
            repo_full_name, pr_number, exc,
        )


def post_summary_comment(repo_full_name: str, pr_number: int, body: str) -> None:
    """Post the overall review summary as a PR issue comment (appears in timeline).

    Args:
        repo_full_name: "owner/repo"
        pr_number:      PR number.
        body:           Markdown summary.
    """
    try:
        repo = _get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        pr.create_issue_comment(body)
        logger.info("Posted summary comment on %s PR #%d", repo_full_name, pr_number)
    except GithubException as exc:
        logger.error(
            "Failed to post summary comment on %s PR #%d: %s",
            repo_full_name, pr_number, exc,
        )
