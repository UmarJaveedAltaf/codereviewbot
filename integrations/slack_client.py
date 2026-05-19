from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger(__name__)


def notify_slack(message: str) -> None:
    """Send a plain-text message to the configured Slack webhook URL.

    Silently skips if SLACK_WEBHOOK_URL is not configured.

    Args:
        message: The text payload to send.
    """
    if not settings.SLACK_WEBHOOK_URL:
        logger.debug("SLACK_WEBHOOK_URL not set — skipping notification")
        return

    try:
        import httpx
        response = httpx.post(
            settings.SLACK_WEBHOOK_URL,
            json={"text": message},
            timeout=5.0,
        )
        response.raise_for_status()
        logger.debug("Slack notification sent")
    except Exception as exc:
        logger.warning("Slack notification failed: %s", exc)


def notify_slack_rich(repo: str, pr_number: int, pr_url: str, verdict: str, files_reviewed: int) -> None:
    """Send a structured Block Kit message to Slack for richer formatting.

    Args:
        repo: "owner/repo" string.
        pr_number: Pull request number.
        pr_url: Full URL to the PR.
        verdict: APPROVED / NEEDS CHANGES / CRITICAL ISSUES
        files_reviewed: Count of files reviewed.
    """
    if not settings.SLACK_WEBHOOK_URL:
        return

    emoji = {"APPROVED": ":white_check_mark:", "NEEDS CHANGES": ":warning:", "CRITICAL ISSUES": ":rotating_light:"}.get(
        verdict, ":information_source:"
    )

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"{emoji} *CodeReviewBot* finished reviewing <{pr_url}|{repo} PR #{pr_number}>"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Verdict:*\n{verdict}"},
            {"type": "mrkdwn", "text": f"*Files reviewed:*\n{files_reviewed}"},
        ]},
    ]

    try:
        import httpx
        response = httpx.post(settings.SLACK_WEBHOOK_URL, json={"blocks": blocks}, timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Slack rich notification failed: %s", exc)
