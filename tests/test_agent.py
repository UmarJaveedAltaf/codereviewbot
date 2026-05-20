"""Tests for agent/reviewer.py — Phase 4 structured LLM review agent.

All LLM calls are mocked — no Google API key required.

Test classes
------------
TestLineComment          Pydantic validation of the LineComment model.
TestFileReview           Pydantic validation of the FileReview model.
TestPRSummary            Pydantic validation of the PRSummary model.
TestCountTokens          _count_tokens() happy path and fallback behaviour.
TestInvokeWithRetry      Retry logic: success paths, back-off, exhaustion.
TestFormatFileComment    Markdown rendering of a FileReview.
TestFileReviewToText     Plain-text conversion used by the summary prompt.
TestExtractSeverity      Legacy severity string mapping.
TestFallbackSummary      Fallback summary when LLM summarizer fails.
TestReviewFile           _review_file() integration with mocked LLM.
TestSummarize            _summarize() integration with mocked LLM.
TestRunReview            run_review() end-to-end with all integrations mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from agent.reviewer import (
    FileReview,
    LineComment,
    PRSummary,
    _count_tokens,
    _extract_severity,
    _fallback_summary,
    _file_review_to_text,
    _format_file_comment,
    _invoke_with_retry,
    _review_file,
    _summarize,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_line_comment() -> LineComment:
    return LineComment(
        line=2,
        severity="high",
        category="bug",
        issue="Division by zero is not guarded — calling divide(x, 0) raises ZeroDivisionError.",
        suggestion="Add `if b == 0: raise ValueError('division by zero')` before returning.",
        code_fix="if b == 0:\n    raise ValueError('division by zero')\nreturn a / b",
    )


@pytest.fixture()
def sample_file_review(sample_line_comment: LineComment) -> FileReview:
    return FileReview(
        filename="math_utils.py",
        summary="The divide() function lacks a zero-division guard.",
        comments=[sample_line_comment],
        has_issues=True,
    )


@pytest.fixture()
def clean_file_review() -> FileReview:
    return FileReview(
        filename="utils.py",
        summary="No issues found; code is clean and well-structured.",
        comments=[],
        has_issues=False,
    )


@pytest.fixture()
def sample_cf():
    """A minimal ChangedFile that the _review_file helper accepts."""
    from parser.diff_extractor import ChangedFile, HunkRange

    return ChangedFile(
        filename="math_utils.py",
        language="python",
        patch="@@ -1,3 +1,5 @@\n+def divide(a, b):\n+    return a / b\n",
        added_lines=["def divide(a, b):", "    return a / b"],
        removed_lines=[],
        hunks=[HunkRange(start=1, length=5)],
    )


# ── LineComment validation ────────────────────────────────────────────────────


class TestLineComment:
    def test_all_fields_stored(self, sample_line_comment: LineComment) -> None:
        assert sample_line_comment.line == 2
        assert sample_line_comment.severity == "high"
        assert sample_line_comment.category == "bug"
        assert "zero" in sample_line_comment.issue.lower()
        assert "raise" in sample_line_comment.suggestion.lower()
        assert sample_line_comment.code_fix is not None

    def test_code_fix_is_optional(self) -> None:
        c = LineComment(
            line=5,
            severity="info",
            category="style",
            issue="Missing return-type annotation on public function.",
            suggestion="Add `-> None` return type hint.",
        )
        assert c.code_fix is None

    def test_line_must_be_at_least_one(self) -> None:
        with pytest.raises(Exception):
            LineComment(
                line=0, severity="info", category="style",
                issue="x" * 10, suggestion="y" * 10,
            )

    @pytest.mark.parametrize(
        "severity",
        ["critical", "high", "medium", "low", "info"],
    )
    def test_all_valid_severities_accepted(self, severity: str) -> None:
        c = LineComment(
            line=1, severity=severity, category="bug",
            issue="some issue description", suggestion="some fix suggestion",
        )
        assert c.severity == severity

    @pytest.mark.parametrize(
        "category",
        ["security", "bug", "logic", "style", "performance", "test_coverage"],
    )
    def test_all_valid_categories_accepted(self, category: str) -> None:
        c = LineComment(
            line=1, severity="low", category=category,
            issue="some issue description", suggestion="some fix suggestion",
        )
        assert c.category == category

    def test_invalid_severity_rejected(self) -> None:
        with pytest.raises(Exception):
            LineComment(
                line=1, severity="urgent", category="bug",
                issue="x" * 10, suggestion="y" * 10,
            )

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(Exception):
            LineComment(
                line=1, severity="low", category="unknown",
                issue="x" * 10, suggestion="y" * 10,
            )


# ── FileReview validation ─────────────────────────────────────────────────────


class TestFileReview:
    def test_has_issues_true_when_comments_present(
        self, sample_file_review: FileReview
    ) -> None:
        assert sample_file_review.has_issues is True
        assert len(sample_file_review.comments) == 1

    def test_has_issues_false_when_no_comments(
        self, clean_file_review: FileReview
    ) -> None:
        assert clean_file_review.has_issues is False
        assert clean_file_review.comments == []

    def test_comments_default_to_empty_list(self) -> None:
        fr = FileReview(filename="x.py", summary="ok", has_issues=False)
        assert fr.comments == []

    def test_multiple_comments_stored_in_order(self) -> None:
        c1 = LineComment(line=1, severity="high", category="bug",
                         issue="issue one long enough", suggestion="fix one long enough")
        c2 = LineComment(line=5, severity="low", category="style",
                         issue="issue two long enough", suggestion="fix two long enough")
        fr = FileReview(filename="a.py", summary="two issues", comments=[c1, c2], has_issues=True)
        assert fr.comments[0].line == 1
        assert fr.comments[1].line == 5


# ── PRSummary validation ──────────────────────────────────────────────────────


class TestPRSummary:
    @pytest.mark.parametrize(
        "verdict",
        ["approved", "needs_changes", "critical_issues"],
    )
    def test_all_valid_verdicts(self, verdict: str) -> None:
        ps = PRSummary(overall_verdict=verdict, summary="Some summary text here.")
        assert ps.overall_verdict == verdict

    def test_invalid_verdict_rejected(self) -> None:
        with pytest.raises(Exception):
            PRSummary(overall_verdict="maybe", summary="text")


# ── Token counting ────────────────────────────────────────────────────────────


class TestCountTokens:
    def test_returns_positive_int_for_normal_text(self) -> None:
        count = _count_tokens("def foo(): return 42")
        assert isinstance(count, int)
        assert count > 0

    def test_longer_text_produces_more_tokens(self) -> None:
        short = _count_tokens("x = 1")
        long  = _count_tokens("x = 1\n" * 200)
        assert long > short

    def test_empty_string_returns_zero(self) -> None:
        assert _count_tokens("") == 0

    def test_unknown_model_falls_back_gracefully(self) -> None:
        # The model parameter is ignored; the char-based estimator always
        # returns a non-negative int regardless of the model name supplied.
        count = _count_tokens("hello world hello world", model="no-such-model-xyz")
        assert isinstance(count, int)
        assert count >= 0


# ── Retry logic ───────────────────────────────────────────────────────────────


class TestInvokeWithRetry:
    def test_succeeds_on_first_attempt(self) -> None:
        chain = MagicMock()
        chain.invoke.return_value = "result"
        assert _invoke_with_retry(chain, {}) == "result"
        chain.invoke.assert_called_once_with({})

    def test_retries_after_transient_failure_and_succeeds(self) -> None:
        chain = MagicMock()
        chain.invoke.side_effect = [ValueError("transient"), "success"]
        with patch("agent.reviewer.time.sleep"):
            result = _invoke_with_retry(chain, {}, max_retries=3)
        assert result == "success"
        assert chain.invoke.call_count == 2

    def test_raises_after_max_retries_exhausted(self) -> None:
        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("permanent failure")
        with patch("agent.reviewer.time.sleep"), \
             pytest.raises(RuntimeError, match="permanent failure"):
            _invoke_with_retry(chain, {}, max_retries=3)
        assert chain.invoke.call_count == 3

    def test_sleeps_with_exponential_backoff(self) -> None:
        chain = MagicMock()
        chain.invoke.side_effect = [ValueError("a"), ValueError("b"), "ok"]
        with patch("agent.reviewer.time.sleep") as mock_sleep:
            _invoke_with_retry(chain, {}, max_retries=3)
        # Two sleeps: after attempt 1 (2^1=2s) and after attempt 2 (2^2=4s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2)
        mock_sleep.assert_any_call(4)

    def test_no_sleep_on_first_success(self) -> None:
        chain = MagicMock()
        chain.invoke.return_value = "ok"
        with patch("agent.reviewer.time.sleep") as mock_sleep:
            _invoke_with_retry(chain, {})
        mock_sleep.assert_not_called()

    def test_passes_inputs_dict_unchanged(self) -> None:
        chain = MagicMock()
        chain.invoke.return_value = "ok"
        inputs = {"repo": "org/repo", "pr_number": 42}
        _invoke_with_retry(chain, inputs)
        chain.invoke.assert_called_once_with(inputs)

    def test_single_retry_allowed(self) -> None:
        chain = MagicMock()
        chain.invoke.side_effect = [ValueError("fail"), "ok"]
        with patch("agent.reviewer.time.sleep"):
            result = _invoke_with_retry(chain, {}, max_retries=2)
        assert result == "ok"

    def test_max_retries_one_raises_immediately(self) -> None:
        chain = MagicMock()
        chain.invoke.side_effect = ValueError("boom")
        with pytest.raises(ValueError):
            _invoke_with_retry(chain, {}, max_retries=1)
        assert chain.invoke.call_count == 1


# ── _format_file_comment ──────────────────────────────────────────────────────


class TestFormatFileComment:
    def test_contains_filename(self, sample_file_review: FileReview) -> None:
        assert "math_utils.py" in _format_file_comment(sample_file_review)

    def test_contains_file_summary(self, sample_file_review: FileReview) -> None:
        body = _format_file_comment(sample_file_review)
        assert "zero-division guard" in body

    def test_contains_line_number(self, sample_file_review: FileReview) -> None:
        assert "Line 2" in _format_file_comment(sample_file_review)

    def test_contains_severity_tag(self, sample_file_review: FileReview) -> None:
        assert "HIGH" in _format_file_comment(sample_file_review)

    def test_contains_category(self, sample_file_review: FileReview) -> None:
        assert "bug" in _format_file_comment(sample_file_review)

    def test_contains_code_fix(self, sample_file_review: FileReview) -> None:
        assert "division by zero" in _format_file_comment(sample_file_review)

    def test_clean_file_shows_no_issues_message(
        self, clean_file_review: FileReview
    ) -> None:
        assert "No issues" in _format_file_comment(clean_file_review)

    def test_includes_disclaimer(self, sample_file_review: FileReview) -> None:
        assert "Not a substitute for human review" in _format_file_comment(
            sample_file_review
        )

    def test_comments_sorted_by_line(self) -> None:
        fr = FileReview(
            filename="x.py",
            summary="two issues",
            has_issues=True,
            comments=[
                LineComment(line=10, severity="low", category="style",
                            issue="style issue found here", suggestion="fix the style issue"),
                LineComment(line=3,  severity="high", category="bug",
                            issue="bug issue found here", suggestion="fix the bug issue"),
            ],
        )
        body = _format_file_comment(fr)
        assert body.index("Line 3") < body.index("Line 10")


# ── _file_review_to_text ──────────────────────────────────────────────────────


class TestFileReviewToText:
    def test_contains_filename(self, sample_file_review: FileReview) -> None:
        assert "math_utils.py" in _file_review_to_text(sample_file_review)

    def test_contains_issue_description(
        self, sample_file_review: FileReview
    ) -> None:
        assert "zero" in _file_review_to_text(sample_file_review).lower()

    def test_contains_suggestion(self, sample_file_review: FileReview) -> None:
        text = _file_review_to_text(sample_file_review)
        assert "Suggestion:" in text

    def test_contains_code_fix(self, sample_file_review: FileReview) -> None:
        assert "division by zero" in _file_review_to_text(sample_file_review)

    def test_clean_file_has_no_comments_block(
        self, clean_file_review: FileReview
    ) -> None:
        text = _file_review_to_text(clean_file_review)
        assert "utils.py" in text
        # No bullet-point findings
        assert "- [" not in text


# ── _extract_severity ────────────────────────────────────────────────────────


class TestExtractSeverity:
    def test_critical_overrides_lower_severities(self) -> None:
        fr = FileReview(
            filename="f.py",
            summary="s",
            has_issues=True,
            comments=[
                LineComment(line=1, severity="critical", category="security",
                            issue="injection risk here", suggestion="sanitise input"),
                LineComment(line=2, severity="low", category="style",
                            issue="naming convention issue", suggestion="rename variable"),
            ],
        )
        assert _extract_severity([fr]) == "CRITICAL"

    def test_high_maps_to_warning(self) -> None:
        fr = FileReview(
            filename="f.py",
            summary="s",
            has_issues=True,
            comments=[
                LineComment(line=1, severity="high", category="bug",
                            issue="null pointer risk", suggestion="add null check"),
            ],
        )
        assert _extract_severity([fr]) == "WARNING"

    def test_medium_maps_to_suggestion(self) -> None:
        fr = FileReview(
            filename="f.py",
            summary="s",
            has_issues=True,
            comments=[
                LineComment(line=1, severity="medium", category="performance",
                            issue="unnecessary allocation", suggestion="use generator"),
            ],
        )
        assert _extract_severity([fr]) == "SUGGESTION"

    def test_no_comments_returns_suggestion(
        self, clean_file_review: FileReview
    ) -> None:
        assert _extract_severity([clean_file_review]) == "SUGGESTION"

    def test_multiple_files_worst_wins(self) -> None:
        fr1 = FileReview(
            filename="a.py", summary="s", has_issues=True,
            comments=[LineComment(line=1, severity="low", category="style",
                                  issue="style issue here", suggestion="fix style")],
        )
        fr2 = FileReview(
            filename="b.py", summary="s", has_issues=True,
            comments=[LineComment(line=1, severity="critical", category="security",
                                  issue="security issue here", suggestion="fix security")],
        )
        assert _extract_severity([fr1, fr2]) == "CRITICAL"


# ── _fallback_summary ────────────────────────────────────────────────────────


class TestFallbackSummary:
    def test_approved_verdict_when_no_issues(
        self, clean_file_review: FileReview
    ) -> None:
        text = _fallback_summary([clean_file_review])
        assert "APPROVED" in text

    def test_needs_changes_verdict_when_issues_present(
        self, sample_file_review: FileReview
    ) -> None:
        text = _fallback_summary([sample_file_review])
        assert "NEEDS CHANGES" in text

    def test_includes_all_filenames(
        self, sample_file_review: FileReview, clean_file_review: FileReview
    ) -> None:
        text = _fallback_summary([sample_file_review, clean_file_review])
        assert "math_utils.py" in text
        assert "utils.py" in text

    def test_includes_codereviewbot_attribution(
        self, clean_file_review: FileReview
    ) -> None:
        assert "CodeReviewBot" in _fallback_summary([clean_file_review])


# ── _review_file integration ──────────────────────────────────────────────────


class TestReviewFile:
    """Tests for _review_file() — LLM calls mocked via _invoke_with_retry."""

    @patch("agent.reviewer.get_similar_reviews",    return_value=[])
    @patch("agent.reviewer.get_relevant_conventions", return_value=[])
    @patch("agent.reviewer.extract_functions",       return_value=[])
    def test_returns_file_review_instance(
        self, _mock_ast, _mock_conv, _mock_reviews,
        sample_cf, sample_file_review: FileReview,
    ) -> None:
        mock_llm = MagicMock()
        with patch("agent.reviewer._invoke_with_retry", return_value=sample_file_review):
            result = _review_file(mock_llm, "org/repo", 42, "Test PR", sample_cf)
        assert isinstance(result, FileReview)
        assert result.filename == "math_utils.py"

    @patch("agent.reviewer.get_similar_reviews",    return_value=[])
    @patch("agent.reviewer.get_relevant_conventions", return_value=["Always guard division"])
    @patch("agent.reviewer.extract_functions",       return_value=[])
    def test_conventions_appear_in_prompt_inputs(
        self, _mock_ast, _mock_conv, _mock_reviews,
        sample_cf, sample_file_review: FileReview,
    ) -> None:
        captured: dict = {}

        def fake_invoke(chain, inputs, **kwargs):
            captured.update(inputs)
            return sample_file_review

        mock_llm = MagicMock()
        with patch("agent.reviewer._invoke_with_retry", side_effect=fake_invoke):
            _review_file(mock_llm, "org/repo", 42, "title", sample_cf)

        assert "Always guard division" in captured.get("past_conventions", "")

    @patch("agent.reviewer.get_similar_reviews")
    @patch("agent.reviewer.get_relevant_conventions", return_value=[])
    @patch("agent.reviewer.extract_functions",       return_value=[])
    def test_past_reviews_appear_in_prompt_inputs(
        self, _mock_ast, _mock_conv, mock_reviews,
        sample_cf, sample_file_review: FileReview,
    ) -> None:
        from memory.vector_store import ReviewResult

        mock_reviews.return_value = [
            ReviewResult(
                review_id="abc-123",
                code_snippet="def divide(a, b): return a / b",
                comment="No zero guard — will raise ZeroDivisionError.",
                file="math_utils.py",
                line=1,
                severity="WARNING",
                category="bug",
                accepted=1,
                repo="org/repo",
                timestamp="2024-01-01T00:00:00+00:00",
                distance=0.05,
            )
        ]
        captured: dict = {}

        def fake_invoke(chain, inputs, **kwargs):
            captured.update(inputs)
            return sample_file_review

        mock_llm = MagicMock()
        with patch("agent.reviewer._invoke_with_retry", side_effect=fake_invoke):
            _review_file(mock_llm, "org/repo", 42, "title", sample_cf)

        past = captured.get("past_conventions", "")
        assert "No zero guard" in past
        assert "✓ accepted" in past   # accepted=1

    @patch("agent.reviewer.get_similar_reviews",    return_value=[])
    @patch("agent.reviewer.get_relevant_conventions", return_value=[])
    @patch("agent.reviewer.extract_functions",       return_value=[])
    def test_patch_truncated_to_max_chars(
        self, _mock_ast, _mock_conv, _mock_reviews, sample_file_review: FileReview,
    ) -> None:
        from parser.diff_extractor import ChangedFile

        long_patch = "@@ -1 +1 @@\n" + "+x = 1\n" * 1_000
        big_cf = ChangedFile(
            filename="big.py",
            language="python",
            patch=long_patch,
            added_lines=["x = 1"] * 1_000,
            removed_lines=[],
            hunks=[],
        )
        captured: dict = {}

        def fake_invoke(chain, inputs, **kwargs):
            captured.update(inputs)
            return sample_file_review

        mock_llm = MagicMock()
        with patch("agent.reviewer._invoke_with_retry", side_effect=fake_invoke):
            _review_file(mock_llm, "org/repo", 1, "title", big_cf)

        assert len(captured.get("patch", "")) <= 4_000

    @patch("agent.reviewer.get_similar_reviews",    return_value=[])
    @patch("agent.reviewer.get_relevant_conventions", return_value=[])
    @patch("agent.reviewer.extract_functions",       return_value=[])
    def test_with_structured_output_called_on_llm(
        self, _mock_ast, _mock_conv, _mock_reviews,
        sample_cf, sample_file_review: FileReview,
    ) -> None:
        mock_llm = MagicMock()
        with patch("agent.reviewer._invoke_with_retry", return_value=sample_file_review):
            _review_file(mock_llm, "org/repo", 1, "title", sample_cf)
        mock_llm.with_structured_output.assert_called_once_with(FileReview)

    @patch("agent.reviewer.get_similar_reviews",    return_value=[])
    @patch("agent.reviewer.get_relevant_conventions", return_value=[])
    @patch("agent.reviewer.extract_functions",       return_value=[])
    def test_no_context_shows_fallback_message(
        self, _mock_ast, _mock_conv, _mock_reviews,
        sample_cf, sample_file_review: FileReview,
    ) -> None:
        captured: dict = {}

        def fake_invoke(chain, inputs, **kwargs):
            captured.update(inputs)
            return sample_file_review

        mock_llm = MagicMock()
        with patch("agent.reviewer._invoke_with_retry", side_effect=fake_invoke):
            _review_file(mock_llm, "org/repo", 1, "title", sample_cf)

        assert "No prior context" in captured.get("past_conventions", "")


# ── _summarize ────────────────────────────────────────────────────────────────


class TestSummarize:
    def test_returns_pr_summary_instance(
        self, sample_file_review: FileReview
    ) -> None:
        expected = PRSummary(
            overall_verdict="needs_changes",
            summary="## Summary\n\n⚠️ NEEDS CHANGES\n\nFix the divide() zero guard.",
        )
        mock_llm = MagicMock()
        with patch("agent.reviewer._invoke_with_retry", return_value=expected):
            result = _summarize(mock_llm, [sample_file_review])
        assert isinstance(result, PRSummary)
        assert result.overall_verdict == "needs_changes"

    def test_approved_verdict_passthrough(
        self, clean_file_review: FileReview
    ) -> None:
        expected = PRSummary(
            overall_verdict="approved",
            summary="✅ APPROVED — no issues found.",
        )
        mock_llm = MagicMock()
        with patch("agent.reviewer._invoke_with_retry", return_value=expected):
            result = _summarize(mock_llm, [clean_file_review])
        assert result.overall_verdict == "approved"

    def test_with_structured_output_called_on_llm(
        self, sample_file_review: FileReview
    ) -> None:
        expected = PRSummary(overall_verdict="approved", summary="ok")
        mock_llm = MagicMock()
        with patch("agent.reviewer._invoke_with_retry", return_value=expected):
            _summarize(mock_llm, [sample_file_review])
        mock_llm.with_structured_output.assert_called_once_with(PRSummary)

    def test_findings_text_passed_to_invoke(
        self, sample_file_review: FileReview
    ) -> None:
        expected = PRSummary(overall_verdict="needs_changes", summary="details")
        captured: dict = {}

        def fake_invoke(chain, inputs, **kwargs):
            captured.update(inputs)
            return expected

        mock_llm = MagicMock()
        with patch("agent.reviewer._invoke_with_retry", side_effect=fake_invoke):
            _summarize(mock_llm, [sample_file_review])

        assert "math_utils.py" in captured.get("findings", "")


# ── run_review end-to-end ─────────────────────────────────────────────────────


class TestRunReview:
    """run_review() wired end-to-end with all external calls patched."""

    def _make_ctx(self):
        ctx = MagicMock()
        ctx.repo_full_name = "org/repo"
        ctx.pr_number      = 42
        ctx.pr_title       = "Add divide function"
        ctx.pr_url         = "https://github.com/org/repo/pull/42"
        ctx.base_sha       = "base000"
        ctx.head_sha       = "head111"
        ctx.review_id      = "rev-uuid"
        ctx.log_extra.return_value = {}
        return ctx

    def _make_changed_file(self):
        from parser.diff_extractor import ChangedFile, HunkRange
        return ChangedFile(
            filename="math_utils.py",
            language="python",
            patch="@@ -1 +1 @@\n+def divide(a, b): return a / b\n",
            added_lines=["def divide(a, b): return a / b"],
            removed_lines=[],
            hunks=[HunkRange(start=1, length=1)],
        )

    @patch("agent.reviewer.notify_slack_rich")
    @patch("agent.reviewer.post_summary_comment")
    @patch("agent.reviewer.post_review_comment")
    @patch("agent.reviewer.add_review")
    @patch("agent.reviewer._summarize")
    @patch("agent.reviewer._review_file")
    @patch("agent.reviewer.extract_changed_files")
    @patch("agent.reviewer.get_pr_diff")
    @patch("agent.reviewer._build_llm")
    def test_happy_path_posts_comment_and_summary(
        self,
        mock_build, mock_diff, mock_extract, mock_review,
        mock_summarize, mock_add, mock_post_review,
        mock_post_summary, mock_slack,
    ) -> None:
        ctx = self._make_ctx()
        cf  = self._make_changed_file()
        mock_diff.return_value    = [{}]
        mock_extract.return_value = [cf]
        mock_review.return_value  = FileReview(
            filename="math_utils.py",
            summary="No zero guard.",
            comments=[LineComment(line=2, severity="high", category="bug",
                                  issue="zero div risk issue", suggestion="guard it")],
            has_issues=True,
        )
        mock_summarize.return_value = PRSummary(
            overall_verdict="needs_changes",
            summary="⚠️ NEEDS CHANGES",
        )

        from agent.reviewer import run_review
        run_review(ctx)

        mock_post_review.assert_called_once()
        mock_post_summary.assert_called_once()
        mock_slack.assert_called_once()
        mock_add.assert_called_once()

    @patch("agent.reviewer.post_summary_comment")
    @patch("agent.reviewer.extract_changed_files")
    @patch("agent.reviewer.get_pr_diff")
    @patch("agent.reviewer._build_llm")
    def test_empty_diff_posts_no_reviewable_files_message(
        self, mock_build, mock_diff, mock_extract, mock_summary,
    ) -> None:
        ctx = self._make_ctx()
        mock_diff.return_value    = []
        mock_extract.return_value = []

        from agent.reviewer import run_review
        run_review(ctx)

        mock_summary.assert_called_once()
        body = mock_summary.call_args[0][2]
        assert "No reviewable" in body

    @patch("agent.reviewer.notify_slack")
    @patch("agent.reviewer.notify_slack_rich")
    @patch("agent.reviewer.post_summary_comment")
    @patch("agent.reviewer.post_review_comment")
    @patch("agent.reviewer.add_review")
    @patch("agent.reviewer._summarize",    side_effect=RuntimeError("summarize failed"))
    @patch("agent.reviewer._review_file")
    @patch("agent.reviewer.extract_changed_files")
    @patch("agent.reviewer.get_pr_diff")
    @patch("agent.reviewer._build_llm")
    def test_summary_failure_posts_fallback(
        self,
        mock_build, mock_diff, mock_extract, mock_review,
        mock_summarize, mock_add, mock_post_review,
        mock_post_summary, mock_slack_rich, mock_slack,
    ) -> None:
        ctx = self._make_ctx()
        cf  = self._make_changed_file()
        mock_diff.return_value    = [{}]
        mock_extract.return_value = [cf]
        mock_review.return_value  = FileReview(
            filename="math_utils.py",
            summary="Issue found.",
            comments=[LineComment(line=1, severity="low", category="style",
                                  issue="style issue found", suggestion="fix it")],
            has_issues=True,
        )

        from agent.reviewer import run_review
        run_review(ctx)

        # Should still post a summary (the fallback) — no crash
        mock_post_summary.assert_called_once()
        body = mock_post_summary.call_args[0][2]
        assert "CodeReviewBot" in body

    @patch("agent.reviewer.notify_slack")
    @patch("agent.reviewer.get_pr_diff", side_effect=ConnectionError("GitHub down"))
    @patch("agent.reviewer._build_llm")
    def test_pipeline_crash_sends_slack_alert(
        self, mock_build, mock_diff, mock_slack,
    ) -> None:
        ctx = self._make_ctx()

        from agent.reviewer import run_review
        run_review(ctx)   # must not raise

        mock_slack.assert_called_once()
        msg = mock_slack.call_args[0][0]
        assert "org/repo" in msg
        assert "42" in msg

    @patch("agent.reviewer.notify_slack_rich")
    @patch("agent.reviewer.post_summary_comment")
    @patch("agent.reviewer.post_review_comment")
    @patch("agent.reviewer.add_review")
    @patch("agent.reviewer._summarize")
    @patch("agent.reviewer._review_file", side_effect=RuntimeError("LLM exploded"))
    @patch("agent.reviewer.extract_changed_files")
    @patch("agent.reviewer.get_pr_diff")
    @patch("agent.reviewer._build_llm")
    def test_per_file_failure_skipped_gracefully(
        self,
        mock_build, mock_diff, mock_extract,
        mock_review, mock_summarize, mock_add,
        mock_post_review, mock_post_summary, mock_slack,
    ) -> None:
        ctx = self._make_ctx()
        cf  = self._make_changed_file()
        mock_diff.return_value    = [{}]
        mock_extract.return_value = [cf]

        from agent.reviewer import run_review
        run_review(ctx)

        # No comment posted since file failed; no summary either
        mock_post_review.assert_not_called()
        mock_post_summary.assert_not_called()
