"""Slack notification client for CodeReviewBot.

All functions skip silently when ``SLACK_WEBHOOK_URL`` is not configured so
local development never requires a Slack workspace.

Functions
---------
send_review_summary   Block Kit message with severity breakdown table and
                      colour-coded attachment (red / yellow / green).
notify_slack          Plain-text webhook message (simple alerts).
notify_slack_rich     Structured Block Kit message for completed reviews.
"""

from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger(__name__)

# Attachment colours mapped to severity profile.
_COLOUR_CRITICAL = "#FF4444"   # red
_COLOUR_HIGH     = "#FFB800"   # amber
_COLOUR_OK       = "#36A64F"   # green


# ── New Phase-5 function ──────────────────────────────────────────────────────


def send_review_summary(
    pr_url:    str,
    pr_title:  str,
    critical:  int,
    high:      int,
    medium:    int,
    low:       int,
    repo:      str,
    info:      int = 0,
) -> None:
    """Send a structured Block Kit review summary to the configured Slack webhook.

    Attachment colour logic:
        🔴 Red   (``#FF4444``) — one or more **critical** findings.
        🟡 Amber (``#FFB800``) — no critical but one or more **high** findings.
        🟢 Green (``#36A64F``) — no critical, no high (clean or informational only).

    The message includes:
        * Header with PR link and repository name.
        * Severity breakdown fields (critical / high / medium / low / info).
        * An overall verdict line.
        * A "View PR" button action.

    Args:
        pr_url:   Full GitHub URL of the pull request.
        pr_title: Title of the pull request.
        critical: Number of critical-severity findings.
        high:     Number of high-severity findings.
        medium:   Number of medium-severity findings.
        low:      Number of low-severity findings.
        repo:     ``"owner/repo"`` string.
        info:     Number of info-level findings (default 0).
    """
    if not settings.SLACK_WEBHOOK_URL:
        logger.debug("SLACK_WEBHOOK_URL not set — skipping send_review_summary")
        return

    # ── Colour and verdict ────────────────────────────────────────────────
    if critical > 0:
        colour  = _COLOUR_CRITICAL
        verdict = "🚨 *CRITICAL ISSUES* — immediate action required"
    elif high > 0:
        colour  = _COLOUR_HIGH
        verdict = "⚠️ *NEEDS CHANGES* — high-priority issues found"
    else:
        colour  = _COLOUR_OK
        verdict = "✅ *APPROVED* — no critical or high issues"

    total = critical + high + medium + low + info

    # ── Block Kit payload ─────────────────────────────────────────────────
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🤖 CodeReviewBot — Review Complete",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Repository:* `{repo}`\n"
                    f"*PR:* <{pr_url}|{pr_title}>"
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": verdict},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"🚨 *Critical:*\n{critical}"},
                {"type": "mrkdwn", "text": f"❗ *High:*\n{high}"},
                {"type": "mrkdwn", "text": f"⚠️ *Medium:*\n{medium}"},
                {"type": "mrkdwn", "text": f"💡 *Low:*\n{low}"},
                {"type": "mrkdwn", "text": f"ℹ️ *Info:*\n{info}"},
                {"type": "mrkdwn", "text": f"📊 *Total findings:*\n{total}"},
            ],
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View PR", "emoji": True},
                    "url": pr_url,
                    "style": "primary" if critical == 0 and high == 0 else "danger",
                }
            ],
        },
    ]

    # Colour-coded legacy attachment (Slack still renders these in sidebar).
    attachments = [
        {
            "color": colour,
            "fallback": f"CodeReviewBot reviewed {repo}: {critical} critical, "
                        f"{high} high, {medium} medium, {low} low findings.",
        }
    ]

    payload: dict = {"blocks": blocks, "attachments": attachments}

    _post_webhook(payload, label="send_review_summary")


# ── Existing helpers (kept for backward compatibility) ────────────────────────


def notify_slack(message: str) -> None:
    """Send a plain-text message to the configured Slack webhook URL.

    Silently skips if ``SLACK_WEBHOOK_URL`` is not configured.

    Args:
        message: The text payload to send.
    """
    if not settings.SLACK_WEBHOOK_URL:
        logger.debug("SLACK_WEBHOOK_URL not set — skipping notification")
        return
    _post_webhook({"text": message}, label="notify_slack")


def notify_slack_rich(
    repo:           str,
    pr_number:      int,
    pr_url:         str,
    verdict:        str,
    files_reviewed: int,
) -> None:
    """Send a structured Block Kit message for a completed review.

    Args:
        repo:           ``"owner/repo"`` string.
        pr_number:      Pull request number.
        pr_url:         Full URL to the PR.
        verdict:        Human-readable verdict string (e.g. ``"✅ APPROVED"``).
        files_reviewed: Number of files reviewed.
    """
    if not settings.SLACK_WEBHOOK_URL:
        return

    _EMOJI_MAP = {
        "APPROVED":       ":white_check_mark:",
        "NEEDS CHANGES":  ":warning:",
        "CRITICAL ISSUES": ":rotating_light:",
    }
    emoji = _EMOJI_MAP.get(verdict.upper(), ":information_source:")

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{emoji} *CodeReviewBot* finished reviewing "
                    f"<{pr_url}|{repo} PR #{pr_number}>"
                ),
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Verdict:*\n{verdict}"},
                {"type": "mrkdwn", "text": f"*Files reviewed:*\n{files_reviewed}"},
            ],
        },
    ]
    _post_webhook({"blocks": blocks}, label="notify_slack_rich")


# ── Internal HTTP helper ──────────────────────────────────────────────────────


def _post_webhook(payload: dict, label: str = "slack") -> None:
    """POST *payload* as JSON to the configured Slack webhook URL.

    Logs a warning on any HTTP or network error — never raises so that
    Slack failures never crash the review pipeline.

    Args:
        payload: Dict to serialise as the POST body.
        label:   Caller name used in log messages.
    """
    try:
        import httpx
        response = httpx.post(
            settings.SLACK_WEBHOOK_URL,
            json=payload,
            timeout=5.0,
        )
        response.raise_for_status()
        logger.debug("Slack %s notification sent (status %d)", label, response.status_code)
    except Exception as exc:
        logger.warning("Slack %s notification failed: %s", label, exc)
