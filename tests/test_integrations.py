"""Tests for integrations/github_client.py, integrations/slack_client.py,
and the dashboard data-loading helpers in dashboard/app.py.

All external calls are mocked — no real GitHub token, Slack URL,
or ChromaDB persistence needed.

Test classes
------------
TestPostReviewComments   post_review_comments(): review event selection,
                         inline comment formatting, error handling.
TestPostPrSummary        post_pr_summary(): severity table generation,
                         passthrough when no file_reviews supplied.
TestMarkCommentResolved  mark_comment_resolved(): logs without raising.
TestPostReviewComment    Legacy post_review_comment() sanity check.
TestPostSummaryComment   Legacy post_summary_comment() sanity check.
TestGetPrFiles           get_pr_files(): data shape returned from PyGithub.
TestSendReviewSummary    send_review_summary(): Block Kit payload, colour
                         logic, skips when URL unset.
TestNotifySlack          notify_slack(): happy path and URL-unset skip.
TestNotifySlackRich      notify_slack_rich(): block structure.
TestDashboardDataLoaders load_reviews() and load_conventions() with a
                         mocked ChromaDB client.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

# ── Helpers shared across GitHub tests ───────────────────────────────────────

def _make_github_mocks():
    """Return (mock_repo, mock_commit, mock_pr) all wired together."""
    mock_repo   = MagicMock()
    mock_commit = MagicMock()
    mock_pr     = MagicMock()
    mock_repo.get_commit.return_value = mock_commit
    mock_repo.get_pull.return_value   = mock_pr
    return mock_repo, mock_commit, mock_pr


def _make_file_review(severity: str = "high", with_code_fix: bool = False):
    """Return a minimal FileReview-like object suitable for github_client tests."""
    from agent.reviewer import FileReview, LineComment

    code_fix = "return a / b if b != 0 else None" if with_code_fix else None
    return FileReview(
        filename="math_utils.py",
        summary="Division-by-zero risk detected.",
        has_issues=True,
        comments=[
            LineComment(
                line=5,
                severity=severity,
                category="bug",
                issue="No guard against b=0 before dividing.",
                suggestion="Raise ValueError when b is zero.",
                code_fix=code_fix,
            )
        ],
    )


# ── post_review_comments ──────────────────────────────────────────────────────


class TestPostReviewComments:
    @patch("integrations.github_client._get_repo")
    def test_uses_request_changes_for_critical(self, mock_get_repo) -> None:
        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comments
        post_review_comments("org/repo", 1, "abc123", _make_file_review("critical"))

        mock_pr.create_review.assert_called_once()
        kwargs = mock_pr.create_review.call_args[1]
        assert kwargs["event"] == "REQUEST_CHANGES"

    @patch("integrations.github_client._get_repo")
    def test_uses_comment_for_high_severity(self, mock_get_repo) -> None:
        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comments
        post_review_comments("org/repo", 1, "abc123", _make_file_review("high"))

        kwargs = mock_pr.create_review.call_args[1]
        assert kwargs["event"] == "COMMENT"

    @patch("integrations.github_client._get_repo")
    def test_uses_comment_for_medium_severity(self, mock_get_repo) -> None:
        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comments
        post_review_comments("org/repo", 1, "abc123", _make_file_review("medium"))

        kwargs = mock_pr.create_review.call_args[1]
        assert kwargs["event"] == "COMMENT"

    @patch("integrations.github_client._get_repo")
    def test_inline_comment_has_correct_path_and_line(self, mock_get_repo) -> None:
        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comments
        post_review_comments("org/repo", 1, "abc123", _make_file_review("high"))

        kwargs   = mock_pr.create_review.call_args[1]
        comments = kwargs["comments"]
        assert len(comments) == 1
        assert comments[0]["path"] == "math_utils.py"
        assert comments[0]["line"] == 5
        assert comments[0]["side"] == "RIGHT"

    @patch("integrations.github_client._get_repo")
    def test_inline_comment_body_contains_severity(self, mock_get_repo) -> None:
        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comments
        post_review_comments("org/repo", 1, "abc123", _make_file_review("high"))

        comment_body = mock_pr.create_review.call_args[1]["comments"][0]["body"]
        assert "HIGH" in comment_body

    @patch("integrations.github_client._get_repo")
    def test_inline_comment_includes_code_fix(self, mock_get_repo) -> None:
        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comments
        post_review_comments(
            "org/repo", 1, "abc123",
            _make_file_review("high", with_code_fix=True),
        )

        comment_body = mock_pr.create_review.call_args[1]["comments"][0]["body"]
        assert "return a / b" in comment_body

    @patch("integrations.github_client._get_repo")
    def test_review_body_contains_filename(self, mock_get_repo) -> None:
        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comments
        post_review_comments("org/repo", 1, "abc123", _make_file_review())

        body = mock_pr.create_review.call_args[1]["body"]
        assert "math_utils.py" in body

    @patch("integrations.github_client._get_repo")
    def test_github_exception_logged_not_raised(self, mock_get_repo) -> None:
        from github import GithubException

        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_pr.create_review.side_effect = GithubException(422, "Unprocessable", {})
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comments
        # Must not raise
        post_review_comments("org/repo", 1, "abc123", _make_file_review())

    @patch("integrations.github_client._get_repo")
    def test_commit_fetched_with_correct_sha(self, mock_get_repo) -> None:
        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comments
        post_review_comments("org/repo", 1, "deadbeef", _make_file_review())

        mock_repo.get_commit.assert_called_once_with("deadbeef")

    @patch("integrations.github_client._get_repo")
    def test_mixed_severities_request_changes_when_any_critical(
        self, mock_get_repo,
    ) -> None:
        from agent.reviewer import FileReview, LineComment

        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        fr = FileReview(
            filename="app.py",
            summary="Two issues.",
            has_issues=True,
            comments=[
                LineComment(line=1, severity="low",      category="style",
                            issue="naming issue here", suggestion="rename it"),
                LineComment(line=8, severity="critical", category="security",
                            issue="injection risk here", suggestion="sanitise"),
            ],
        )

        from integrations.github_client import post_review_comments
        post_review_comments("org/repo", 1, "sha", fr)

        kwargs = mock_pr.create_review.call_args[1]
        assert kwargs["event"] == "REQUEST_CHANGES"
        assert len(kwargs["comments"]) == 2


# ── post_pr_summary ───────────────────────────────────────────────────────────


class TestPostPrSummary:
    @patch("integrations.github_client._get_repo")
    def test_posts_summary_text_without_file_reviews(self, mock_get_repo) -> None:
        mock_repo, _, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_pr_summary
        post_pr_summary("org/repo", 1, "## Summary\n\nLooks good.")

        mock_pr.create_issue_comment.assert_called_once()
        body = mock_pr.create_issue_comment.call_args[0][0]
        assert "Looks good" in body

    @patch("integrations.github_client._get_repo")
    def test_prepends_severity_table_when_file_reviews_given(
        self, mock_get_repo,
    ) -> None:
        from agent.reviewer import FileReview, LineComment

        mock_repo, _, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        fr = FileReview(
            filename="x.py",
            summary="s",
            has_issues=True,
            comments=[
                LineComment(line=1, severity="critical", category="security",
                            issue="sql injection risk", suggestion="sanitise"),
                LineComment(line=2, severity="high",     category="bug",
                            issue="null deref risk", suggestion="add null check"),
                LineComment(line=3, severity="medium",   category="performance",
                            issue="n+1 query", suggestion="use bulk fetch"),
            ],
        )

        from integrations.github_client import post_pr_summary
        post_pr_summary("org/repo", 1, "Summary text.", file_reviews=[fr])

        body = mock_pr.create_issue_comment.call_args[0][0]
        assert "Critical" in body
        assert "High" in body
        assert "Medium" in body
        assert "1" in body     # critical count
        assert "Summary text." in body

    @patch("integrations.github_client._get_repo")
    def test_no_table_when_all_counts_zero(self, mock_get_repo) -> None:
        from agent.reviewer import FileReview

        mock_repo, _, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        fr = FileReview(filename="x.py", summary="clean", has_issues=False, comments=[])

        from integrations.github_client import post_pr_summary
        post_pr_summary("org/repo", 1, "All clean.", file_reviews=[fr])

        body = mock_pr.create_issue_comment.call_args[0][0]
        # No table rows (no issues) — just the summary text
        assert "|" not in body
        assert "All clean." in body

    @patch("integrations.github_client._get_repo")
    def test_github_exception_does_not_raise(self, mock_get_repo) -> None:
        from github import GithubException

        mock_repo, _, mock_pr = _make_github_mocks()
        mock_pr.create_issue_comment.side_effect = GithubException(500, "error", {})
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_pr_summary
        post_pr_summary("org/repo", 1, "text")  # must not raise


# ── mark_comment_resolved ─────────────────────────────────────────────────────


class TestMarkCommentResolved:
    def test_does_not_raise(self) -> None:
        from integrations.github_client import mark_comment_resolved
        mark_comment_resolved("org/repo", 42, 99_999)  # just logs

    def test_accepts_arbitrary_comment_ids(self) -> None:
        from integrations.github_client import mark_comment_resolved
        for cid in [1, 100, 999_999_999]:
            mark_comment_resolved("org/repo", 1, cid)


# ── Legacy helpers ────────────────────────────────────────────────────────────


class TestPostReviewComment:
    @patch("integrations.github_client._get_repo")
    def test_calls_create_review_with_comment_event(self, mock_get_repo) -> None:
        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comment
        post_review_comment("org/repo", 1, "sha", "file.py", "looks good")

        mock_pr.create_review.assert_called_once()
        assert mock_pr.create_review.call_args[1]["event"] == "COMMENT"

    @patch("integrations.github_client._get_repo")
    def test_github_exception_logged_not_raised(self, mock_get_repo) -> None:
        from github import GithubException

        mock_repo, mock_commit, mock_pr = _make_github_mocks()
        mock_pr.create_review.side_effect = GithubException(403, "Forbidden", {})
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_review_comment
        post_review_comment("org/repo", 1, "sha", "f.py", "body")


class TestPostSummaryComment:
    @patch("integrations.github_client._get_repo")
    def test_calls_create_issue_comment(self, mock_get_repo) -> None:
        mock_repo, _, mock_pr = _make_github_mocks()
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_summary_comment
        post_summary_comment("org/repo", 1, "summary body")

        mock_pr.create_issue_comment.assert_called_once_with("summary body")

    @patch("integrations.github_client._get_repo")
    def test_github_exception_logged_not_raised(self, mock_get_repo) -> None:
        from github import GithubException

        mock_repo, _, mock_pr = _make_github_mocks()
        mock_pr.create_issue_comment.side_effect = GithubException(404, "Not Found", {})
        mock_get_repo.return_value = mock_repo

        from integrations.github_client import post_summary_comment
        post_summary_comment("org/repo", 1, "body")


class TestGetPrFiles:
    @patch("integrations.github_client._get_repo")
    def test_returns_files_and_title(self, mock_get_repo) -> None:
        mock_file        = MagicMock()
        mock_file.filename   = "src/app.py"
        mock_file.patch      = "@@ -1 +1 @@\n+x = 1"
        mock_file.status     = "modified"
        mock_file.additions  = 1
        mock_file.deletions  = 0

        mock_pr = MagicMock()
        mock_pr.title             = "Add feature"
        mock_pr.get_files.return_value = [mock_file]

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_get_repo.return_value      = mock_repo

        from integrations.github_client import get_pr_files
        files, title = get_pr_files("org/repo", 1)

        assert title == "Add feature"
        assert len(files) == 1
        assert files[0]["filename"] == "src/app.py"
        assert files[0]["patch"] == "@@ -1 +1 @@\n+x = 1"


# ── send_review_summary (Slack) ───────────────────────────────────────────────


class TestSendReviewSummary:
    def _call(self, critical=0, high=0, medium=0, low=0, info=0):
        from integrations.slack_client import send_review_summary
        send_review_summary(
            pr_url="https://github.com/org/repo/pull/1",
            pr_title="Add divide function",
            critical=critical,
            high=high,
            medium=medium,
            low=low,
            repo="org/repo",
            info=info,
        )

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_skips_when_url_not_configured(
        self, mock_post, mock_settings,
    ) -> None:
        mock_settings.SLACK_WEBHOOK_URL = ""
        self._call(critical=1)
        mock_post.assert_not_called()

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_red_attachment_for_critical(
        self, mock_post, mock_settings,
    ) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        self._call(critical=2, high=1)

        payload = mock_post.call_args[0][0]
        assert payload["attachments"][0]["color"] == "#FF4444"

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_amber_attachment_for_high_no_critical(
        self, mock_post, mock_settings,
    ) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        self._call(critical=0, high=3, medium=1)

        payload = mock_post.call_args[0][0]
        assert payload["attachments"][0]["color"] == "#FFB800"

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_green_attachment_when_clean(
        self, mock_post, mock_settings,
    ) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        self._call(critical=0, high=0, medium=2, low=5)

        payload = mock_post.call_args[0][0]
        assert payload["attachments"][0]["color"] == "#36A64F"

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_payload_contains_pr_link(
        self, mock_post, mock_settings,
    ) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        self._call(high=1)

        payload   = mock_post.call_args[0][0]
        all_text  = json.dumps(payload)
        assert "https://github.com/org/repo/pull/1" in all_text

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_payload_contains_repo_name(
        self, mock_post, mock_settings,
    ) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        self._call()
        payload  = mock_post.call_args[0][0]
        all_text = json.dumps(payload)
        assert "org/repo" in all_text

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_severity_counts_in_fields(
        self, mock_post, mock_settings,
    ) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        self._call(critical=1, high=2, medium=3, low=4, info=5)

        payload  = mock_post.call_args[0][0]
        all_text = json.dumps(payload)
        # All counts appear somewhere in the blocks
        for n in ["1", "2", "3", "4", "5"]:
            assert n in all_text

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_verdict_critical_text(
        self, mock_post, mock_settings,
    ) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        self._call(critical=1)

        payload  = mock_post.call_args[0][0]
        all_text = json.dumps(payload)
        assert "CRITICAL" in all_text.upper()

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_has_view_pr_button(
        self, mock_post, mock_settings,
    ) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        self._call()

        payload  = mock_post.call_args[0][0]
        all_text = json.dumps(payload)
        assert "View PR" in all_text


# ── notify_slack ──────────────────────────────────────────────────────────────


class TestNotifySlack:
    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_sends_text_payload(self, mock_post, mock_settings) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        from integrations.slack_client import notify_slack
        notify_slack("hello from bot")

        mock_post.assert_called_once()
        payload = mock_post.call_args[0][0]
        assert payload["text"] == "hello from bot"

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_skips_when_url_empty(self, mock_post, mock_settings) -> None:
        mock_settings.SLACK_WEBHOOK_URL = ""
        from integrations.slack_client import notify_slack
        notify_slack("ignored message")
        mock_post.assert_not_called()


# ── notify_slack_rich ─────────────────────────────────────────────────────────


class TestNotifySlackRich:
    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_blocks_contain_pr_url(self, mock_post, mock_settings) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        from integrations.slack_client import notify_slack_rich
        notify_slack_rich(
            repo="org/repo", pr_number=7,
            pr_url="https://github.com/org/repo/pull/7",
            verdict="APPROVED", files_reviewed=3,
        )
        payload  = mock_post.call_args[0][0]
        all_text = json.dumps(payload)
        assert "https://github.com/org/repo/pull/7" in all_text

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_skips_when_url_empty(self, mock_post, mock_settings) -> None:
        mock_settings.SLACK_WEBHOOK_URL = ""
        from integrations.slack_client import notify_slack_rich
        notify_slack_rich("r", 1, "url", "APPROVED", 1)
        mock_post.assert_not_called()

    @patch("integrations.slack_client.settings")
    @patch("integrations.slack_client._post_webhook")
    def test_verdict_appears_in_payload(self, mock_post, mock_settings) -> None:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/fake"
        from integrations.slack_client import notify_slack_rich
        notify_slack_rich(
            repo="r", pr_number=1,
            pr_url="https://github.com/r/pull/1",
            verdict="NEEDS CHANGES", files_reviewed=2,
        )
        payload  = mock_post.call_args[0][0]
        all_text = json.dumps(payload)
        assert "NEEDS CHANGES" in all_text


# ── Dashboard data loaders ────────────────────────────────────────────────────


class TestDashboardDataLoaders:
    """Tests for load_reviews() and load_conventions() in dashboard/app.py.

    Both functions use @st.cache_data; we clear the cache before each test
    and mock the ChromaDB client so no real persistence layer is needed.
    """

    def _make_chroma_col(self, docs, metadatas, count=None):
        """Build a mock ChromaDB collection that returns *docs* and *metadatas*."""
        col = MagicMock()
        col.count.return_value = count if count is not None else len(docs)
        col.get.return_value   = {"documents": docs, "metadatas": metadatas}
        return col

    @patch("dashboard.app.chromadb")
    def test_load_reviews_returns_dataframe(self, mock_chromadb) -> None:
        col = self._make_chroma_col(
            docs=["def foo(): pass"],
            metadatas=[{
                "repo": "org/repo", "file": "foo.py", "severity": "WARNING",
                "category": "style", "accepted": -1,
                "timestamp": "2024-06-01T12:00:00+00:00",
                "comment": "Missing type hints.",
            }],
        )
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = col

        import dashboard.app as app
        app.load_reviews.clear()
        df = app.load_reviews()

        assert not df.empty
        assert "repo" in df.columns
        assert df.iloc[0]["repo"] == "org/repo"
        assert df.iloc[0]["file"] == "foo.py"
        assert "date" in df.columns

    @patch("dashboard.app.chromadb")
    def test_load_reviews_empty_collection_returns_empty_df(
        self, mock_chromadb,
    ) -> None:
        col = MagicMock()
        col.count.return_value = 0
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = col

        import dashboard.app as app
        app.load_reviews.clear()
        df = app.load_reviews()

        assert df.empty

    @patch("dashboard.app.chromadb")
    def test_load_conventions_returns_dataframe(self, mock_chromadb) -> None:
        col = self._make_chroma_col(
            docs=["Never store secrets in source code."],
            metadatas=[{
                "category": "security", "language": "", "repo": "", "added_by": "security-team",
            }],
        )
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = col

        import dashboard.app as app
        app.load_conventions.clear()
        df = app.load_conventions()

        assert not df.empty
        assert "rule" in df.columns
        assert "Never store secrets" in df.iloc[0]["rule"]
        assert df.iloc[0]["language"] == "all"     # empty string → "all"
        assert df.iloc[0]["repo"]     == "global"  # empty string → "global"

    @patch("dashboard.app.chromadb")
    def test_load_conventions_empty_returns_empty_df(self, mock_chromadb) -> None:
        col = MagicMock()
        col.count.return_value = 0
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = col

        import dashboard.app as app
        app.load_conventions.clear()
        df = app.load_conventions()

        assert df.empty

    @patch("dashboard.app.chromadb")
    def test_load_reviews_multiple_rows(self, mock_chromadb) -> None:
        col = self._make_chroma_col(
            docs=["x = 1", "y = 2", "z = 3"],
            metadatas=[
                {"repo": "a/b", "file": "f1.py", "severity": "CRITICAL",
                 "category": "security", "accepted": 1,
                 "timestamp": "2024-01-01T00:00:00+00:00", "comment": "c1"},
                {"repo": "a/b", "file": "f2.py", "severity": "WARNING",
                 "category": "bug",      "accepted": 0,
                 "timestamp": "2024-01-02T00:00:00+00:00", "comment": "c2"},
                {"repo": "c/d", "file": "f3.py", "severity": "SUGGESTION",
                 "category": "style",    "accepted": -1,
                 "timestamp": "2024-01-03T00:00:00+00:00", "comment": "c3"},
            ],
        )
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = col

        import dashboard.app as app
        app.load_reviews.clear()
        df = app.load_reviews()

        assert len(df) == 3
        assert set(df["repo"].unique()) == {"a/b", "c/d"}


# ── Dashboard KPI helpers ─────────────────────────────────────────────────────


class TestDashboardHelpers:
    """Unit tests for pure-Python helper functions in dashboard/app.py."""

    def _make_df(self):
        import pandas as pd
        return pd.DataFrame([
            {"repo": "a/b", "severity": "CRITICAL",   "accepted": 1,  "category": "security"},
            {"repo": "a/b", "severity": "WARNING",    "accepted": 0,  "category": "bug"},
            {"repo": "a/b", "severity": "SUGGESTION", "accepted": -1, "category": "style"},
            {"repo": "c/d", "severity": "WARNING",    "accepted": 1,  "category": "security"},
        ])

    def test_acceptance_rate_all_repos(self) -> None:
        from dashboard.app import _acceptance_rate
        df   = self._make_df()
        rate = _acceptance_rate(df)
        # Rated: 3 (accepted=1: 2, rejected=0: 1). Rate = 2/3
        assert abs(rate - 2 / 3) < 1e-9

    def test_acceptance_rate_scoped_to_repo(self) -> None:
        from dashboard.app import _acceptance_rate
        df = self._make_df()
        # a/b: 1 accepted, 1 rejected, 1 unrated → 1/2
        rate_ab = _acceptance_rate(df, "a/b")
        assert abs(rate_ab - 0.5) < 1e-9
        # c/d: 1 accepted → 1.0
        rate_cd = _acceptance_rate(df, "c/d")
        assert abs(rate_cd - 1.0) < 1e-9

    def test_acceptance_rate_empty_returns_zero(self) -> None:
        import pandas as pd
        from dashboard.app import _acceptance_rate
        assert _acceptance_rate(pd.DataFrame()) == 0.0

    def test_severity_counts_normalises_legacy_labels(self) -> None:
        from dashboard.app import _severity_counts
        df    = self._make_df()
        counts = _severity_counts(df)
        # CRITICAL → "critical", WARNING → "high", SUGGESTION → "low"
        assert counts.get("critical", 0) == 1
        assert counts.get("high", 0)     == 2
        assert counts.get("low", 0)      == 1
