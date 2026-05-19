"""LangChain review orchestrator.

run_review(ctx) is the single public entry point.  It receives a PRContext
built by the webhook handler and drives the full review pipeline:

    repo.compare(base, head)          ← fetch real diff via GitHub API
         │
         ▼
    diff_extractor.extract_changed_files()
         │
         ▼
    ast_parser.extract_functions()    ← per file, identify changed functions
         │
         ├── vector_store.query_similar()   ← retrieve past team conventions
         │
         ▼
    review_prompt | ChatOpenAI        ← generate per-file findings
         │
         ├── github_client.post_review_comment()
         └── vector_store.store_review()
         │
         ▼
    summary_prompt | ChatOpenAI       ← roll up all findings
         │
         ├── github_client.post_summary_comment()
         └── slack_client.notify_slack_rich()
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from agent.prompts import review_prompt, summary_prompt
from config import settings
from integrations.github_client import (
    get_pr_diff,
    post_review_comment,
    post_summary_comment,
)
from integrations.slack_client import notify_slack, notify_slack_rich
from memory.vector_store import ReviewMemory, query_similar, store_review
from parser.ast_parser import extract_functions
from parser.diff_extractor import extract_changed_files

# Import PRContext at TYPE_CHECKING time only to avoid a circular import.
# At runtime we receive it as a plain object — no isinstance check needed.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from webhook.handler import PRContext

logger = logging.getLogger(__name__)

# Maximum patch length sent to the LLM — guards against token overflow.
_MAX_PATCH_CHARS = 4_000
# Maximum combined review findings sent to the summary prompt.
_MAX_SUMMARY_CHARS = 12_000


# ── Internal types ─────────────────────────────────────────────────────────────


@dataclass
class FileFinding:
    filename: str
    language: str
    comment: str
    patch: str


# ── LLM factory ───────────────────────────────────────────────────────────────


def _build_llm() -> ChatOpenAI:
    """Return a configured ChatOpenAI instance."""
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
    cf,                    # parser.diff_extractor.ChangedFile
) -> str:
    """Run the review chain for a single ChangedFile.

    Accepts individual PR-identity values rather than a full PRContext so
    the function stays unit-testable without constructing the whole dataclass.

    Args:
        llm:            Shared ChatOpenAI instance.
        repo_full_name: "owner/repo" string.
        pr_number:      Pull request number.
        pr_title:       Pull request title shown in the prompt.
        cf:             ChangedFile dataclass from diff_extractor.

    Returns:
        Raw LLM response string (Markdown-formatted findings).
    """
    # Retrieve up to 3 semantically similar past comments.
    past_reviews = query_similar(cf.patch, n_results=3)
    if past_reviews:
        past_text = "\n\n".join(
            f"[{p['repo']} PR#{p['pr_number']} / {p.get('filename', '?')}]\n{p['comment']}"
            for p in past_reviews
        )
    else:
        past_text = "No similar past reviews found for this codebase yet."

    # Identify changed functions via AST for richer prompt context.
    functions = extract_functions("\n".join(cf.added_lines), cf.language)
    if functions:
        fn_summary = "\n".join(
            f"  • {fn.name}() — lines {fn.start_line}–{fn.end_line}"
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

    Called by the webhook handler as a background task.  Never raises —
    all exceptions are caught, logged, and converted to a Slack alert so
    the team knows a review failed without crashing the worker process.

    Pipeline:
        1. Fetch diff via repo.compare(base_sha, head_sha).
        2. Parse diff into ChangedFile objects.
        3. For each file: query memory → LLM review → post comment → store.
        4. Roll up all findings into a summary → post + Slack.

    Args:
        ctx: PRContext dataclass built by the webhook handler.
    """
    log = logger.getChild("run_review")
    extra = ctx.log_extra()

    log.info("Review pipeline started", extra=extra)

    try:
        # ── Step 1: Fetch full diff via compare() ─────────────────────────
        log.debug(
            "Fetching diff %s..%s",
            ctx.base_sha[:12], ctx.head_sha[:12],
            extra=extra,
        )
        raw_files = get_pr_diff(
            repo_full_name=ctx.repo_full_name,
            base_sha=ctx.base_sha,
            head_sha=ctx.head_sha,
        )

        # ── Step 2: Parse into ChangedFile dataclasses ────────────────────
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
            "Reviewing %d file(s)",
            len(changed_files),
            extra=extra | {"file_count": len(changed_files)},
        )

        # ── Step 3: Per-file review ───────────────────────────────────────
        llm = _build_llm()
        findings: list[FileFinding] = []

        for cf in changed_files:
            log.debug("Reviewing file: %s", cf.filename, extra=extra)

            try:
                comment = _review_file(llm, ctx.repo_full_name, ctx.pr_number, ctx.pr_title, cf)
            except Exception:
                log.exception(
                    "LLM review failed for %s — skipping file",
                    cf.filename,
                    extra=extra,
                )
                continue

            findings.append(FileFinding(
                filename=cf.filename,
                language=cf.language,
                comment=comment,
                patch=cf.patch,
            ))

            # Post per-file review comment
            post_review_comment(
                repo_full_name=ctx.repo_full_name,
                pr_number=ctx.pr_number,
                commit_sha=ctx.head_sha,
                filename=cf.filename,
                body=_format_file_comment(cf.filename, comment),
            )

            # Persist to vector store for future PRs
            store_review(ReviewMemory(
                id=str(uuid.uuid4()),
                code_snippet=cf.patch[:2_000],
                comment=comment[:1_000],
                repo=ctx.repo_full_name,
                pr_number=ctx.pr_number,
                filename=cf.filename,
            ))

        # ── Step 4: Summary comment ───────────────────────────────────────
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

        # ── Step 5: Slack notification ────────────────────────────────────
        verdict_line = _extract_verdict(verdict_body)
        notify_slack_rich(
            repo=ctx.repo_full_name,
            pr_number=ctx.pr_number,
            pr_url=ctx.pr_url,
            verdict=verdict_line,
            files_reviewed=len(findings),
        )

        log.info(
            "Review pipeline complete",
            extra=extra | {"files_reviewed": len(findings), "verdict": verdict_line},
        )

    except Exception:
        log.exception("Review pipeline crashed", extra=extra)
        # Best-effort Slack alert — don't let this raise either.
        try:
            notify_slack(
                f":x: CodeReviewBot failed to review {ctx.repo_full_name} "
                f"PR #{ctx.pr_number} (review_id={ctx.review_id}). "
                f"Check server logs for details."
            )
        except Exception:
            pass


# ── Helpers ────────────────────────────────────────────────────────────────────


def _format_file_comment(filename: str, comment: str) -> str:
    """Wrap per-file LLM output in a consistent Markdown header."""
    return (
        f"## 🤖 CodeReviewBot — `{filename}`\n\n"
        f"{comment}\n\n"
        f"---\n*Generated by CodeReviewBot. Not a substitute for human review.*"
    )


def _extract_verdict(summary: str) -> str:
    """Pull the APPROVED / NEEDS CHANGES / CRITICAL ISSUES line from the summary.

    Falls back to a generic string if the LLM didn't follow the template.
    """
    for line in summary.splitlines():
        line = line.strip().upper()
        if "APPROVED" in line:
            return "APPROVED"
        if "CRITICAL" in line:
            return "CRITICAL ISSUES"
        if "NEEDS CHANGES" in line or "NEEDS_CHANGES" in line:
            return "NEEDS CHANGES"
    return "REVIEWED"
