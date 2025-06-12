#!/usr/bin/env python3
"""Decode and inspect JWT tokens for debugging."""

import base64
from datetime import UTC, datetime
import json
import sys
from typing import Any

# Constants
JWT_PARTS_COUNT = 3
SIGNATURE_PREVIEW_LENGTH = 20


def decode_jwt(token: str) -> dict[str, Any]:
    """Decode JWT token without verification."""
    parts = token.split(".")
    if len(parts) != JWT_PARTS_COUNT:
        return {"error": "Invalid JWT format - expected 3 parts"}

    # Decode header
    header = json.loads(base64.urlsafe_b64decode(parts[0] + "=="))

    # Decode payload
    payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))

    # Convert timestamps to readable format
    if "exp" in payload:
        payload["exp_readable"] = datetime.fromtimestamp(
            payload["exp"], tz=UTC
        ).isoformat()
    if "iat" in payload:
        payload["iat_readable"] = datetime.fromtimestamp(
            payload["iat"], tz=UTC
        ).isoformat()
    if "auth_time" in payload:
        payload["auth_time_readable"] = datetime.fromtimestamp(
            payload["auth_time"], tz=UTC
        ).isoformat()

    return {
        "header": header,
        "payload": payload,
        "signature": (
            parts[2][:SIGNATURE_PREVIEW_LENGTH] + "..."
            if len(parts[2]) > SIGNATURE_PREVIEW_LENGTH
            else parts[2]
        ),
    }


if __name__ == "__main__":
    # Use ternary operator for token assignment
    token = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()

    if not token:
        print("Usage: python decode_jwt.py <token>")
        print("   or: echo $TOKEN | python decode_jwt.py")
        sys.exit(1)

    try:
        decoded = decode_jwt(token)
        print(json.dumps(decoded, indent=2))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error decoding token: {e}")
        sys.exit(1)
