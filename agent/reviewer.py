"""LangChain review orchestrator.

run_review(ctx) is the single public entry point.  It receives a PRContext
built by the webhook handler and drives the full review pipeline:

    repo.compare(base, head)
         │
         ▼
    diff_extractor.extract_changed_files()
         │
         ▼
    ast_parser.extract_functions()
         │
         ├── vector_store.get_similar_reviews()    ← past findings for this repo
         ├── vector_store.get_relevant_conventions() ← team coding rules
         │
         ▼
    review_prompt | ChatOpenAI        ← per-file findings
         │
         ├── github_client.post_review_comment()
         └── vector_store.add_review()             ← persist for future PRs
         │
         ▼
    summary_prompt | ChatOpenAI
         │
         ├── github_client.post_summary_comment()
         └── slack_client.notify_slack_rich()
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI

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
from parser.diff_extractor import extract_changed_files

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from webhook.handler import PRContext

logger = logging.getLogger(__name__)

_MAX_PATCH_CHARS = 4_000
_MAX_SUMMARY_CHARS = 12_000

_SEVERITY_RE = re.compile(r"\[(CRITICAL|WARNING|SUGGESTION)\]", re.IGNORECASE)


# ── Internal types ─────────────────────────────────────────────────────────────


@dataclass
class FileFinding:
    filename: str
    language: str
    comment: str
    patch: str


# ── LLM factory ───────────────────────────────────────────────────────────────


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.2,
        api_key=settings.OPENAI_API_KEY,
    )


# ── Per-file review ────────────────────────────────────────────────────────────


def _review_file(
    llm: ChatOpenAI,
    repo_full_name: str,
    pr_number: int,
    pr_title: str,
    cf,
) -> str:
    """Run the review chain for a single ChangedFile.

    Queries both past similar reviews and relevant team conventions to build
    rich context for the LLM prompt.

    Args:
        llm:            Shared ChatOpenAI instance.
        repo_full_name: "owner/repo".
        pr_number:      Pull request number.
        pr_title:       PR title shown in the prompt.
        cf:             ChangedFile dataclass from diff_extractor.

    Returns:
        Raw LLM response string (Markdown-formatted findings).
    """
    # ── Memory: past reviews for this repo ───────────────────────────────
    past_reviews: list[ReviewResult] = get_similar_reviews(
        cf.patch, repo=repo_full_name, n=3
    )

    # ── Memory: relevant team conventions ────────────────────────────────
    conventions: list[str] = get_relevant_conventions(
        cf.patch, language=cf.language, n=3
    )

    # ── Build past_conventions block for the prompt ───────────────────────
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

    # ── AST summary ───────────────────────────────────────────────────────
    functions = extract_functions("\n".join(cf.added_lines), cf.language)
    if functions:
        fn_summary = "\n".join(
            f"  • {fn.name}() — lines {fn.start_line}–{fn.end_line}"
            f"  [complexity={fn.complexity}"
            f"{', has_docstring' if fn.has_docstring else ''}"
            f"{', method' if fn.is_method else ''}]"
            for fn in functions
        )
    else:
        fn_summary = "  (no named functions detected in added lines)"

    chain = review_prompt | llm
    response = chain.invoke({
        "past_conventions": past_text,
        "repo": repo_full_name,
        "pr_number": pr_number,
        "pr_title": pr_title,
        "filename": cf.filename,
        "language": cf.language,
        "patch": cf.patch[:_MAX_PATCH_CHARS],
        "functions_summary": fn_summary,
    })
    return response.content


# ── Public entry point ────────────────────────────────────────────────────────


def run_review(ctx: "PRContext") -> None:
    """Orchestrate a full PR review from diff fetch to posted comments.

    Never raises — all exceptions are caught and converted to a Slack alert
    so a failing review doesn't crash the Uvicorn worker.
    """
    log = logger.getChild("run_review")
    extra = ctx.log_extra()

    log.info("Review pipeline started", extra=extra)

    try:
        raw_files = get_pr_diff(
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

        log.info("Reviewing %d file(s)", len(changed_files),
                 extra=extra | {"file_count": len(changed_files)})

        llm = _build_llm()
        findings: list[FileFinding] = []

        for cf in changed_files:
            log.debug("Reviewing file: %s", cf.filename, extra=extra)
            try:
                comment = _review_file(llm, ctx.repo_full_name, ctx.pr_number, ctx.pr_title, cf)
            except Exception:
                log.exception("LLM review failed for %s — skipping", cf.filename, extra=extra)
                continue

            findings.append(FileFinding(
                filename=cf.filename, language=cf.language,
                comment=comment, patch=cf.patch,
            ))

            post_review_comment(
                repo_full_name=ctx.repo_full_name,
                pr_number=ctx.pr_number,
                commit_sha=ctx.head_sha,
                filename=cf.filename,
                body=_format_file_comment(cf.filename, comment),
            )

            # Persist this review for future context
            add_review(
                code_snippet=cf.patch[:2_000],
                comment=comment[:1_000],
                metadata=ReviewMetadata(
                    file=cf.filename,
                    severity=_extract_severity(comment),
                    category="general",
                    repo=ctx.repo_full_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
            )

        if not findings:
            log.warning("All per-file reviews failed — no summary to post", extra=extra)
            return

        combined = "\n\n---\n\n".join(
            f"### {f.filename}\n{f.comment}" for f in findings
        )[:_MAX_SUMMARY_CHARS]

        summary_chain = summary_prompt | llm
        summary_resp = summary_chain.invoke({"findings": combined})
        verdict_body = summary_resp.content

        post_summary_comment(ctx.repo_full_name, ctx.pr_number, verdict_body)

        verdict_line = _extract_verdict(verdict_body)
        notify_slack_rich(
            repo=ctx.repo_full_name,
            pr_number=ctx.pr_number,
            pr_url=ctx.pr_url,
            verdict=verdict_line,
            files_reviewed=len(findings),
        )

        log.info("Review pipeline complete",
                 extra=extra | {"files_reviewed": len(findings), "verdict": verdict_line})

    except Exception:
        log.exception("Review pipeline crashed", extra=extra)
        try:
            notify_slack(
                f":x: CodeReviewBot failed reviewing {ctx.repo_full_name} "
                f"PR #{ctx.pr_number} (review_id={ctx.review_id}). "
                f"Check server logs."
            )
        except Exception:
            pass


# ── Helpers ────────────────────────────────────────────────────────────────────


def _format_file_comment(filename: str, comment: str) -> str:
    return (
        f"## 🤖 CodeReviewBot — `{filename}`\n\n"
        f"{comment}\n\n"
        f"---\n*Generated by CodeReviewBot. Not a substitute for human review.*"
    )


def _extract_severity(text: str) -> str:
    """Parse the first [SEVERITY] tag from LLM output. Defaults to WARNING."""
    m = _SEVERITY_RE.search(text)
    return m.group(1).upper() if m else "WARNING"


def _extract_verdict(summary: str) -> str:
    """Pull the overall verdict line from the summary comment."""
    upper = summary.upper()
    if "CRITICAL" in upper:
        return "CRITICAL ISSUES"
    if "NEEDS CHANGES" in upper or "NEEDS_CHANGES" in upper:
        return "NEEDS CHANGES"
    if "APPROVED" in upper:
        return "APPROVED"
    return "REVIEWED"
