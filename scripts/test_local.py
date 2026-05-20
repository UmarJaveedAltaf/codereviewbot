"""Local development script: send a mock PR webhook to the running server.

Usage:
    python scripts/test_local.py [--url http://localhost:8000]

Requires the server to be running (`make run`) and optionally an ngrok tunnel.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

import httpx
import httpx
from dotenv import load_dotenv


load_dotenv()

PAYLOAD_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "mock_pr_payload.json"


def sign_payload(payload: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), msg=payload, digestmod=hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a mock PR webhook for local testing")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the FastAPI server")
    args = parser.parse_args()

    secret = os.getenv("GITHUB_WEBHOOK_SECRET", "test-secret")
    payload = PAYLOAD_PATH.read_bytes()
    signature = sign_payload(payload, secret)

    headers = {
        "Content-Type": "application/json",
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": signature,
    }

    target = args.url.rstrip("/") + "/webhook/github"
    print(f"Sending mock webhook to {target} ...")

    try:
        response = httpx.post(target, content=payload, headers=headers, timeout=10.0)
        print(f"Status: {response.status_code}")
        print(response.json())
    except httpx.ConnectError:
        print(f"ERROR: Could not connect to {target}. Is the server running?")
        sys.exit(1)


if __name__ == "__main__":
    main()
