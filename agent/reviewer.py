"""LangChain review orchestrator — Phase 4.

Structured-output pipeline using Gemini (via ChatGoogleGenerativeAI) and
LangChain's ``with_structured_output()`` so every LLM response is validated
as a Pydantic model before it reaches any downstream code.

Public API
----------
run_review(ctx)
    Orchestrate a full PR review end-to-end (diff → per-file findings →
    GitHub comments → Slack notification).

Internal helpers (also importable for tests)
--------------------------------------------
_review_file(llm, repo, pr_number, pr_title, cf)  → FileReview
_summarize(llm, file_reviews)                      → PRSummary
_invoke_with_retry(chain, inputs, max_retries)     → any
_count_tokens(text, model)                         → int
_format_file_comment(file_review)                  → str
_file_review_to_text(file_review)                  → str
_extract_severity(file_reviews)                    → str
_fallback_summary(file_reviews)                    → str
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agent.prompts import review_prompt, summary_prompt
from config import settings
from integrations.github_client import (
    get_pr_diff,
    post_review_comment,
    post_summary_comment,
)
from integrations.slack_client import notify_slack, notify_slack_rich
from memory.vector_store import (
    ReviewMetadata,
    ReviewResult,
    add_review,
    get_relevant_conventions,
    get_similar_reviews,
)
from parser.ast_parser import extract_functions
from parser.diff_extractor import ChangedFile, extract_changed_files

if TYPE_CHECKING:
    from webhook.handler import PRContext

logger = logging.getLogger(__name__)

_MAX_PATCH_CHARS   = 4_000
_MAX_SUMMARY_CHARS = 12_000
_MAX_RETRIES       = 3


# ── Structured output models ───────────────────────────────────────────────────


class LineComment(BaseModel):
    """A single review finding tied to a specific line in the diff."""

    line: int = Field(
        ..., ge=1,
        description="Line number in the NEW file ('+' side of the unified diff).",
    )
    severity: Literal["critical", "high", "medium", "low", "info"] = Field(
        ...,
        description="Impact severity: critical > high > medium > low > info.",
    )
    category: Literal[
        "security", "bug", "logic", "style", "performance", "test_coverage"
    ] = Field(..., description="Classification of the issue type.")
    issue: str = Field(
        ..., min_length=5,
        description="Clear description of what is wrong and why it matters.",
    )
    suggestion: str = Field(
        ..., min_length=5,
        description="Actionable fix recommendation.",
    )
    code_fix: str | None = Field(
        None,
        description="Optional short code snippet showing the corrected code.",
    )


class FileReview(BaseModel):
    """Structured review output for a single changed file."""

    filename: str = Field(..., description="Relative path of the file reviewed.")
    summary: str = Field(
        ...,
        description="1-2 sentence overview of the file's review result.",
    )
    comments: list[LineComment] = Field(
        default_factory=list,
        description="Line-level findings; empty list when no issues found.",
    )
    has_issues: bool = Field(
        ...,
        description="True when at least one actionable issue was found.",
    )


class PRSummary(BaseModel):
    """Overall PR verdict aggregated from all per-file reviews."""

    overall_verdict: Literal["approved", "needs_changes", "critical_issues"] = Field(
        ...,
        description="Top-level review decision for the pull request.",
    )
    summary: str = Field(
        ...,
        description="Markdown body suitable for a PR top-level comment.",
    )


# ── Token counting ─────────────────────────────────────────────────────────────


def _count_tokens(text: str, model: str = "gemini-1.5-flash") -> int:
    """Return the approximate token count for *text*.

    Uses the ~4 characters-per-token heuristic, which is model-agnostic and
    requires no additional dependencies.  The *model* parameter is accepted for
    API compatibility but is not used.

    Args:
        text:  The string to measure.
        model: Ignored — kept for call-site compatibility.

    Returns:
        Non-negative integer token estimate.
    """
    return max(1, len(text) // 4) if text else 0


# ── Retry wrapper ──────────────────────────────────────────────────────────────


def _invoke_with_retry(chain, inputs: dict, max_retries: int = _MAX_RETRIES):
    """Invoke a LangChain Runnable with exponential back-off retry.

    On each failure (any exception) the helper waits ``2**attempt`` seconds
    before the next attempt.  If *max_retries* attempts all fail the last
    exception is re-raised.

    Args:
        chain:       Any LangChain Runnable (e.g. ``prompt | structured_llm``).
        inputs:      Template variable dict forwarded verbatim to
                     ``chain.invoke()``.
        max_retries: Maximum number of invocation attempts (default 3).

    Returns:
        Whatever ``chain.invoke()`` returns on success.

    Raises:
        The exception raised by the last failed attempt.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                wait = 2 ** attempt  # 2 s, 4 s for attempts 1 and 2
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                    attempt, max_retries, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "LLM call failed after %d attempts: %s", max_retries, exc,
                )
    raise last_exc  # type: ignore[misc]


# ── LLM factory ───────────────────────────────────────────────────────────────


def _build_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Construct a ChatGoogleGenerativeAI instance from application settings."""
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=temperature,
        google_api_key=settings.GOOGLE_API_KEY,
    )


# ── Per-file review ────────────────────────────────────────────────────────────


def _review_file(
    llm: ChatGoogleGenerativeAI,
    repo_full_name: str,
    pr_number: int,
    pr_title: str,
    cf: ChangedFile,
) -> FileReview:
    """Run a structured review for a single ``ChangedFile``.

    Enriches the prompt with:
      - Similar past review findings from the vector store.
      - Relevant team conventions retrieved by semantic search.
      - AST-derived function summary (complexity, docstring presence, etc.).

    The LLM is called via ``with_structured_output(FileReview)`` so the return
    value is always a validated ``FileReview`` Pydantic model.

    Args:
        llm:            Shared ``ChatOpenAI`` instance.
        repo_full_name: ``"owner/repo"`` string.
        pr_number:      Pull request number.
        pr_title:       PR title shown in the prompt for context.
        cf:             ``ChangedFile`` produced by ``diff_extractor``.

    Returns:
        ``FileReview`` with per-line comments and a file-level summary.
    """
    # ── Memory: past similar reviews ─────────────────────────────────────
    past_reviews: list[ReviewResult] = get_similar_reviews(
        cf.patch, repo=repo_full_name, n=3
    )

    # ── Memory: relevant team conventions ────────────────────────────────
    conventions: list[str] = get_relevant_conventions(
        cf.patch, language=cf.language, n=3
    )

    # ── Build past_conventions prompt block ──────────────────────────────
    context_parts: list[str] = []
    if past_reviews:
        context_parts.append("### Similar past review findings:")
        _STATUS = {1: "✓ accepted", 0: "✗ rejected", -1: "unrated"}
        for r in past_reviews:
            status = _STATUS.get(r.accepted, "unrated")
            context_parts.append(
                f"[{r.repo} / {r.file}] ({status})\n{r.comment}"
            )

    if conventions:
        context_parts.append("### Team conventions that apply:")
        for rule in conventions:
            context_parts.append(f"• {rule}")

    past_text = (
        "\n\n".join(context_parts)
        if context_parts
        else "No prior context available for this codebase yet."
    )

    # ── AST function summary ──────────────────────────────────────────────
    functions = extract_functions("\n".join(cf.added_lines), cf.language)
    fn_summary = (
        "\n".join(
            f"  • {fn.name}() — lines {fn.start_line}–{fn.end_line}"
            f"  [complexity={fn.complexity}"
            f"{', has_docstring' if fn.has_docstring else ''}"
            f"{', method' if fn.is_method else ''}]"
            for fn in functions
        )
        if functions
        else "  (no named functions detected in added lines)"
    )

    patch_text = cf.patch[:_MAX_PATCH_CHARS]

    # Log estimated token budget before calling the API
    logger.debug(
        "Reviewing %s — patch ~%d tokens",
        cf.filename,
        _count_tokens(patch_text),
    )

    inputs = {
        "past_conventions": past_text,
        "repo":             repo_full_name,
        "pr_number":        pr_number,
        "pr_title":         pr_title,
        "filename":         cf.filename,
        "language":         cf.language,
        "patch":            patch_text,
        "functions_summary": fn_summary,
    }

    structured_llm = llm.with_structured_output(FileReview)
    chain = review_prompt | structured_llm
    return _invoke_with_retry(chain, inputs)


# ── PR-level summary ──────────────────────────────────────────────────────────


def _summarize(llm: ChatGoogleGenerativeAI, file_reviews: list[FileReview]) -> PRSummary:
    """Aggregate per-file findings into an overall ``PRSummary``.

    Args:
        llm:          Shared ``ChatOpenAI`` instance.
        file_reviews: List of ``FileReview`` objects from ``_review_file``.

    Returns:
        ``PRSummary`` with an overall verdict and Markdown summary body.
    """
    findings_text = "\n\n---\n\n".join(
        _file_review_to_text(fr) for fr in file_reviews
    )[:_MAX_SUMMARY_CHARS]

    structured_llm = llm.with_structured_output(PRSummary)
    chain = summary_prompt | structured_llm
    return _invoke_with_retry(chain, {"findings": findings_text})


# ── Formatting helpers ────────────────────────────────────────────────────────


def _format_file_comment(fr: FileReview) -> str:
    """Render a ``FileReview`` as a Markdown GitHub comment body.

    Produces a human-readable comment with severity icons, grouped by line,
    and a standard disclaimer footer.
    """
    _ICONS = {
        "critical": "🚨", "high": "❗", "medium": "⚠️", "low": "💡", "info": "ℹ️",
    }
    lines: list[str] = [
        f"## 🤖 CodeReviewBot — `{fr.filename}`",
        "",
        fr.summary,
        "",
    ]
    if not fr.comments:
        lines.append("✅ No issues found in this file.")
    else:
        for c in sorted(fr.comments, key=lambda x: x.line):
            icon = _ICONS.get(c.severity, "•")
            lines += [
                f"**{icon} Line {c.line}** `[{c.severity.upper()}]` `{c.category}`",
                f"> {c.issue}",
                f"**Suggestion:** {c.suggestion}",
            ]
            if c.code_fix:
                lines.append(f"```\n{c.code_fix}\n```")
            lines.append("")

    lines += [
        "---",
        "*Generated by CodeReviewBot. Not a substitute for human review.*",
    ]
    return "\n".join(lines)


def _file_review_to_text(fr: FileReview) -> str:
    """Convert a ``FileReview`` to a plain-text block for the summary prompt."""
    parts: list[str] = [f"### {fr.filename}", fr.summary]
    for c in fr.comments:
        fix_part = f"\n  Fix: `{c.code_fix}`" if c.code_fix else ""
        parts.append(
            f"- [{c.severity.upper()}] line {c.line} ({c.category}): {c.issue}"
            f"\n  Suggestion: {c.suggestion}{fix_part}"
        )
    return "\n".join(parts)


def _verdict_emoji(verdict: str) -> str:
    return {"approved": "✅", "needs_changes": "⚠️", "critical_issues": "🚨"}.get(
        verdict, "🤖"
    )


def _extract_severity(file_reviews: list[FileReview]) -> str:
    """Map the worst structured severity across *file_reviews* to a legacy
    severity string used by the vector store (CRITICAL / WARNING / SUGGESTION).
    """
    severities = {c.severity for fr in file_reviews for c in fr.comments}
    if "critical" in severities:
        return "CRITICAL"
    if "high" in severities:
        return "WARNING"
    return "SUGGESTION"


def _fallback_summary(file_reviews: list[FileReview]) -> str:
    """Build a plain-text PR summary without calling the LLM.

    Used when the ``_summarize`` LLM call fails after all retries.
    """
    total_issues = sum(len(fr.comments) for fr in file_reviews)
    verdict = "⚠️ NEEDS CHANGES" if total_issues else "✅ APPROVED"
    lines: list[str] = [
        f"## 🤖 CodeReviewBot Summary\n\n**{verdict}**\n",
    ]
    for fr in file_reviews:
        issue_count = len(fr.comments)
        tag = f"{issue_count} issue(s)" if issue_count else "no issues"
        lines.append(f"- **{fr.filename}** — {fr.summary} ({tag})")
    lines.append("\n---\n*Generated by CodeReviewBot.*")
    return "\n".join(lines)


# ── Public entry point ────────────────────────────────────────────────────────


def run_review(ctx: "PRContext") -> None:
    """Orchestrate a full PR review from diff fetch to posted comments.

    Pipeline:
      1. Fetch the diff via PyGithub (``get_pr_diff``).
      2. Parse into ``ChangedFile`` objects (``extract_changed_files``).
      3. For each file: run ``_review_file`` (structured LLM call + memory).
      4. Post per-file Markdown comments to GitHub.
      5. Persist each finding in the vector store for future context.
      6. Run ``_summarize`` for an overall PR verdict.
      7. Post the summary comment to GitHub.
      8. Send a Slack notification.

    Never raises — all exceptions are caught so a failing review never
    crashes the Uvicorn worker.  Critical failures trigger a Slack alert.
    """
    log   = logger.getChild("run_review")
    extra = ctx.log_extra()
    log.info("Review pipeline started", extra=extra)

    try:
        raw_files    = get_pr_diff(
            repo_full_name=ctx.repo_full_name,
            base_sha=ctx.base_sha,
            head_sha=ctx.head_sha,
        )
        changed_files = extract_changed_files(raw_files)

        if not changed_files:
            log.info("No reviewable files in diff — skipping", extra=extra)
            post_summary_comment(
                ctx.repo_full_name,
                ctx.pr_number,
                "🤖 **CodeReviewBot**: No reviewable source files found in this PR.",
            )
            return

        log.info(
            "Reviewing %d file(s)", len(changed_files),
            extra=extra | {"file_count": len(changed_files)},
        )

        llm: ChatGoogleGenerativeAI = _build_llm()
        file_reviews: list[FileReview] = []

        for cf in changed_files:
            log.debug("Reviewing file: %s", cf.filename, extra=extra)
            try:
                fr = _review_file(
                    llm, ctx.repo_full_name, ctx.pr_number, ctx.pr_title, cf
                )
            except Exception:
                log.exception(
                    "LLM review failed for %s — skipping", cf.filename, extra=extra
                )
                continue

            file_reviews.append(fr)

            # Post per-file comment to GitHub
            post_review_comment(
                repo_full_name=ctx.repo_full_name,
                pr_number=ctx.pr_number,
                commit_sha=ctx.head_sha,
                filename=cf.filename,
                body=_format_file_comment(fr),
            )

            # Persist finding for future context
            add_review(
                code_snippet=cf.patch[:2_000],
                comment=fr.summary[:1_000],
                metadata=ReviewMetadata(
                    file=cf.filename,
                    severity=_extract_severity([fr]),
                    category=(
                        fr.comments[0].category if fr.comments else "general"
                    ),
                    repo=ctx.repo_full_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
            )

        if not file_reviews:
            log.warning(
                "All per-file reviews failed — no summary to post", extra=extra
            )
            return

        # ── Aggregate summary ─────────────────────────────────────────────
        try:
            pr_summary = _summarize(llm, file_reviews)
            verdict    = pr_summary.overall_verdict
            summary_body = pr_summary.summary
        except Exception:
            log.exception("Summary generation failed — using fallback", extra=extra)
            verdict      = (
                "needs_changes"
                if any(fr.has_issues for fr in file_reviews)
                else "approved"
            )
            summary_body = _fallback_summary(file_reviews)

        post_summary_comment(ctx.repo_full_name, ctx.pr_number, summary_body)

        notify_slack_rich(
            repo=ctx.repo_full_name,
            pr_number=ctx.pr_number,
            pr_url=ctx.pr_url,
            verdict=f"{_verdict_emoji(verdict)} {verdict.upper().replace('_', ' ')}",
            files_reviewed=len(file_reviews),
        )

        log.info(
            "Review pipeline complete",
            extra=extra | {"files_reviewed": len(file_reviews), "verdict": verdict},
        )

    except Exception:
        log.exception("Review pipeline crashed", extra=extra)
        try:
            notify_slack(
                f":x: CodeReviewBot failed reviewing {ctx.repo_full_name} "
                f"PR #{ctx.pr_number} (review_id={ctx.review_id}). "
                f"Check server logs."
            )
        except Exception:  # noqa: BLE001
            pass
