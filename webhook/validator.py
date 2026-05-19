"""HMAC-SHA256 signature validation for GitHub webhooks.

GitHub signs every payload with the webhook secret and puts the digest in
the X-Hub-Signature-256 header (format: "sha256=<hex>").  We must validate
this *before* touching the payload to prevent spoofed events.

Reference:
    https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries
"""

import hashlib
import hmac
import logging

from fastapi import HTTPException, Request, status

from config import settings

logger = logging.getLogger(__name__)


async def verify_github_signature(request: Request) -> bytes:
    """FastAPI dependency: validate X-Hub-Signature-256 and return the raw body.

    Reads the request body once (FastAPI caches it on the request object so
    subsequent reads inside the route handler are free).  Computes the
    expected HMAC-SHA256 digest against GITHUB_WEBHOOK_SECRET and compares
    using a constant-time equality check to prevent timing attacks.

    Args:
        request: The incoming FastAPI/Starlette request object.

    Raises:
        HTTPException 401: Signature header is absent, malformed, or wrong.
        HTTPException 500: Webhook secret is not configured.

    Returns:
        The raw (bytes) request body, ready to be JSON-decoded by the caller.
    """
    if not settings.GITHUB_WEBHOOK_SECRET:
        logger.error("GITHUB_WEBHOOK_SECRET is not set — cannot validate webhooks")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook secret not configured",
        )

    signature_header: str = request.headers.get("X-Hub-Signature-256", "")
    if not signature_header:
        logger.warning(
            "Webhook received without X-Hub-Signature-256 header",
            extra={"path": str(request.url)},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Hub-Signature-256 header is missing",
        )

    if not signature_header.startswith("sha256="):
        logger.warning("Malformed signature header: %r", signature_header[:40])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Signature header must start with 'sha256='",
        )

    body: bytes = await request.body()
    if not body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty request body",
        )

    expected_digest = hmac.new(
        key=settings.GITHUB_WEBHOOK_SECRET.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    expected_header = f"sha256={expected_digest}"

    if not hmac.compare_digest(expected_header, signature_header):
        # Log at WARNING, not ERROR — this is expected noise from scanners.
        logger.warning(
            "Webhook signature mismatch — possible spoofed request",
            extra={"remote": request.client.host if request.client else "unknown"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Signature verification failed",
        )

    return body
