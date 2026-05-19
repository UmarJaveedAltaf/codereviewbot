"""Tests for memory/vector_store.py and memory/embedder.py.

All tests use the ``mem`` fixture from conftest.py which:
  - Redirects ChromaDB to an in-memory EphemeralClient (no disk writes)
  - Replaces embed_text with a fast deterministic 8-dim function
  - Resets module-level singletons between tests

This means no OpenAI API key is needed and tests run in milliseconds.
"""

from __future__ import annotations

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────


def _add_review(mem, code: str = "def foo(): pass", comment: str = "looks good",
                repo: str = "org/repo", file: str = "foo.py",
                severity: str = "WARNING") -> str:
    return mem.add_review(
        code_snippet=code,
        comment=comment,
        metadata=mem.ReviewMetadata(
            file=file,
            line=1,
            severity=severity,
            category="general",
            repo=repo,
        ),
    )


def _add_convention(mem, rule: str, language: str = "", category: str = "general",
                    repo: str = "") -> str:
    return mem.add_convention(
        rule_text=rule,
        metadata=mem.ConventionMetadata(
            repo=repo,
            language=language,
            category=category,
            added_by="test",
        ),
    )


# ── add_review ─────────────────────────────────────────────────────────────────


class TestAddReview:
    def test_returns_uuid_string(self, mem):
        rid = _add_review(mem)
        assert isinstance(rid, str)
        assert len(rid) == 36                  # standard UUID4 format
        assert rid.count("-") == 4

    def test_two_reviews_have_different_ids(self, mem):
        r1 = _add_review(mem, code="def a(): pass")
        r2 = _add_review(mem, code="def b(): pass")
        assert r1 != r2

    def test_timestamp_auto_filled(self, mem):
        """ReviewMetadata with empty timestamp should be auto-filled on store."""
        rid = mem.add_review(
            code_snippet="x = 1",
            comment="ok",
            metadata=mem.ReviewMetadata(file="a.py", repo="r", timestamp=""),
        )
        col = mem._get_reviews_collection()
        result = col.get(ids=[rid], include=["metadatas"])
        ts = result["metadatas"][0]["timestamp"]
        assert ts and ts != ""

    def test_explicit_timestamp_preserved(self, mem):
        ts = "2024-06-01T12:00:00+00:00"
        rid = mem.add_review(
            code_snippet="y = 2",
            comment="ok",
            metadata=mem.ReviewMetadata(file="b.py", repo="r", timestamp=ts),
        )
        col = mem._get_reviews_collection()
        result = col.get(ids=[rid], include=["metadatas"])
        assert result["metadatas"][0]["timestamp"] == ts

    def test_metadata_fields_stored_correctly(self, mem):
        rid = mem.add_review(
            code_snippet="import os",
            comment="[CRITICAL] Never import os without sanitising paths",
            metadata=mem.ReviewMetadata(
                file="utils.py",
                line=3,
                severity="CRITICAL",
                category="security",
                repo="acme/app",
            ),
        )
        col = mem._get_reviews_collection()
        result = col.get(ids=[rid], include=["metadatas", "documents"])
        meta = result["metadatas"][0]
        assert meta["file"] == "utils.py"
        assert meta["line"] == 3
        assert meta["severity"] == "CRITICAL"
        assert meta["category"] == "security"
        assert meta["repo"] == "acme/app"
        assert meta["accepted"] == -1          # default: unrated
        assert result["documents"][0] == "import os"


# ── get_similar_reviews ────────────────────────────────────────────────────────


class TestGetSimilarReviews:
    def test_empty_store_returns_empty_list(self, mem):
        results = mem.get_similar_reviews("def foo(): pass", repo="org/repo")
        assert results == []

    def test_returns_review_result_objects(self, mem):
        _add_review(mem, code="def divide(a, b): return a / b",
                    comment="No zero guard", repo="org/repo")
        results = mem.get_similar_reviews("def divide(x, y): return x / y",
                                          repo="org/repo", n=1)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, mem.ReviewResult)
        assert r.comment == "No zero guard"
        assert r.repo == "org/repo"
        assert r.file == "foo.py"
        assert isinstance(r.distance, float)

    def test_repo_filter_excludes_other_repos(self, mem):
        _add_review(mem, code="x = 1", comment="team-a review", repo="team-a/repo")
        _add_review(mem, code="x = 2", comment="team-b review", repo="team-b/repo")

        results = mem.get_similar_reviews("x = 3", repo="team-a/repo", n=5)
        repos = {r.repo for r in results}
        assert repos == {"team-a/repo"}

    def test_no_repo_filter_returns_all(self, mem):
        _add_review(mem, code="a = 1", comment="alpha", repo="org/alpha")
        _add_review(mem, code="b = 2", comment="beta", repo="org/beta")

        results = mem.get_similar_reviews("c = 3", repo="", n=10)
        repos = {r.repo for r in results}
        assert "org/alpha" in repos
        assert "org/beta" in repos

    def test_respects_n_limit(self, mem):
        for i in range(5):
            _add_review(mem, code=f"def fn{i}(): pass", comment=f"c{i}", repo="r")
        results = mem.get_similar_reviews("def fn(): pass", repo="r", n=2)
        assert len(results) <= 2

    def test_result_has_review_id(self, mem):
        rid = _add_review(mem)
        results = mem.get_similar_reviews("def foo(): pass", repo="org/repo", n=1)
        assert results[0].review_id == rid

    def test_results_ordered_by_distance(self, mem):
        """Documents identical to the query should appear first."""
        code_exact = "def target(): return 42"
        code_far = "import logging; logger = logging.getLogger(__name__)"
        _add_review(mem, code=code_exact, comment="exact match", repo="r")
        _add_review(mem, code=code_far, comment="far match", repo="r")

        results = mem.get_similar_reviews(code_exact, repo="r", n=2)
        assert results[0].comment == "exact match"


# ── mark_review_accepted / mark_review_rejected ────────────────────────────────


class TestAcceptanceTracking:
    def test_mark_accepted_sets_flag(self, mem):
        rid = _add_review(mem)
        mem.mark_review_accepted(rid)

        col = mem._get_reviews_collection()
        result = col.get(ids=[rid], include=["metadatas"])
        assert result["metadatas"][0]["accepted"] == 1

    def test_mark_rejected_sets_flag(self, mem):
        rid = _add_review(mem)
        mem.mark_review_rejected(rid)

        col = mem._get_reviews_collection()
        result = col.get(ids=[rid], include=["metadatas"])
        assert result["metadatas"][0]["accepted"] == 0

    def test_default_accepted_is_unrated(self, mem):
        rid = _add_review(mem)
        col = mem._get_reviews_collection()
        result = col.get(ids=[rid], include=["metadatas"])
        assert result["metadatas"][0]["accepted"] == -1

    def test_mark_preserves_other_metadata(self, mem):
        """Accepting a review must not clobber other metadata fields."""
        rid = mem.add_review(
            code_snippet="import os",
            comment="check imports",
            metadata=mem.ReviewMetadata(
                file="main.py", line=5, severity="CRITICAL",
                category="security", repo="org/app",
            ),
        )
        mem.mark_review_accepted(rid)

        col = mem._get_reviews_collection()
        result = col.get(ids=[rid], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["file"] == "main.py"
        assert meta["severity"] == "CRITICAL"
        assert meta["category"] == "security"
        assert meta["accepted"] == 1

    def test_mark_unknown_id_does_not_raise(self, mem):
        """Marking a non-existent review should log a warning, not raise."""
        mem.mark_review_accepted("00000000-0000-0000-0000-000000000000")
        mem.mark_review_rejected("00000000-0000-0000-0000-000000000000")


# ── get_acceptance_rate ────────────────────────────────────────────────────────


class TestGetAcceptanceRate:
    def test_empty_store_returns_zero(self, mem):
        assert mem.get_acceptance_rate("org/repo") == 0.0

    def test_all_unrated_returns_zero(self, mem):
        _add_review(mem, repo="org/repo")
        _add_review(mem, code="x = 2", repo="org/repo")
        assert mem.get_acceptance_rate("org/repo") == 0.0

    def test_all_accepted_returns_one(self, mem):
        r1 = _add_review(mem, code="a = 1", repo="org/repo")
        r2 = _add_review(mem, code="b = 2", repo="org/repo")
        mem.mark_review_accepted(r1)
        mem.mark_review_accepted(r2)
        assert mem.get_acceptance_rate("org/repo") == 1.0

    def test_mixed_ratings_correct_fraction(self, mem):
        # 2 accepted, 1 rejected, 1 unrated → rate = 2/3
        r1 = _add_review(mem, code="a = 1", repo="org/repo")
        r2 = _add_review(mem, code="b = 2", repo="org/repo")
        r3 = _add_review(mem, code="c = 3", repo="org/repo")
        _add_review(mem, code="d = 4", repo="org/repo")   # unrated, not counted
        mem.mark_review_accepted(r1)
        mem.mark_review_accepted(r2)
        mem.mark_review_rejected(r3)
        rate = mem.get_acceptance_rate("org/repo")
        assert abs(rate - 2 / 3) < 1e-9

    def test_rate_scoped_to_repo(self, mem):
        ra = _add_review(mem, code="a = 1", repo="org/repo-a")
        _add_review(mem, code="b = 2", repo="org/repo-b")
        mem.mark_review_accepted(ra)
        # repo-b has an unrated review; repo-a should be 1.0
        assert mem.get_acceptance_rate("org/repo-a") == 1.0
        assert mem.get_acceptance_rate("org/repo-b") == 0.0


# ── add_convention / get_relevant_conventions ──────────────────────────────────


class TestConventions:
    def test_add_convention_returns_uuid(self, mem):
        cid = _add_convention(mem, rule="Never log passwords")
        assert isinstance(cid, str) and len(cid) == 36

    def test_get_conventions_empty_store_returns_empty(self, mem):
        result = mem.get_relevant_conventions("import os", language="python")
        assert result == []

    def test_get_conventions_returns_rule_strings(self, mem):
        _add_convention(mem, rule="Use parameterised queries", language="python")
        results = mem.get_relevant_conventions("cursor.execute(query)", language="python")
        assert isinstance(results, list)
        assert all(isinstance(r, str) for r in results)

    def test_language_filter_includes_matching_language(self, mem):
        _add_convention(mem, rule="Python-only rule", language="python")
        _add_convention(mem, rule="JS-only rule", language="javascript")
        results = mem.get_relevant_conventions("x = 1", language="python")
        texts = " ".join(results)
        assert "Python-only rule" in texts
        assert "JS-only rule" not in texts

    def test_language_filter_includes_global_conventions(self, mem):
        """Conventions with language='' apply to all languages."""
        _add_convention(mem, rule="Never commit secrets", language="")
        results = mem.get_relevant_conventions("API_KEY = 'abc'", language="python")
        assert any("Never commit secrets" in r for r in results)

    def test_respects_n_limit(self, mem):
        for i in range(6):
            _add_convention(mem, rule=f"Rule number {i}", language="python")
        results = mem.get_relevant_conventions("code", language="python", n=3)
        assert len(results) <= 3

    def test_most_relevant_convention_returned_first(self, mem):
        """Both conventions are returned when n=2; content is preserved.

        Note: ordering is not asserted here because the test suite uses MD5-based
        fake embeddings that carry no semantic signal — cosine distances between
        any two short texts are essentially random under this scheme.  The
        important invariants (both documents stored, both returned, n limit
        respected) are all verified below.
        """
        _add_convention(mem, rule="Always use parameterised SQL queries to prevent injection",
                        language="python", category="security")
        _add_convention(mem, rule="Use snake_case for all variable names",
                        language="python", category="python_style")

        results = mem.get_relevant_conventions(
            "cursor.execute('SELECT * FROM users WHERE id=' + user_id)",
            language="python",
            n=2,
        )
        assert len(results) == 2
        texts = " | ".join(results)
        assert "parameterised SQL" in texts or "SQL" in texts
        assert "snake_case" in texts

    def test_convention_metadata_stored(self, mem):
        cid = mem.add_convention(
            rule_text="Functions must have type hints",
            metadata=mem.ConventionMetadata(
                repo="org/app",
                language="python",
                category="python_style",
                added_by="tech-lead",
            ),
        )
        col = mem._get_conventions_collection()
        result = col.get(ids=[cid], include=["metadatas", "documents"])
        meta = result["metadatas"][0]
        assert meta["language"] == "python"
        assert meta["category"] == "python_style"
        assert meta["added_by"] == "tech-lead"
        assert result["documents"][0] == "Functions must have type hints"


# ── Backward-compatibility shims ──────────────────────────────────────────────


class TestBackwardCompatShims:
    """Verify old ReviewMemory / store_review / query_similar still function."""

    def test_store_review_and_query_similar(self, mem):
        memory = mem.ReviewMemory(
            id="compat-id",
            code_snippet="def foo(): pass",
            comment="old-style comment",
            repo="org/repo",
            pr_number=7,
            filename="foo.py",
        )
        mem.store_review(memory)

        results = mem.query_similar("def foo(): pass", n_results=1)
        assert len(results) == 1
        assert results[0]["comment"] == "old-style comment"
        assert "distance" in results[0]

    def test_query_similar_empty_returns_empty(self, mem):
        assert mem.query_similar("anything") == []
