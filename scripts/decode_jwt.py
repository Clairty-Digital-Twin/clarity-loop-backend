#!/usr/bin/env python3
"""Decode and inspect JWT tokens for debugging."""

import base64
from datetime import datetime
import json
import sys


def decode_jwt(token: str) -> dict:
    """Decode JWT token without verification."""
    parts = token.split('.')
    if len(parts) != 3:
        return {"error": "Invalid JWT format - expected 3 parts"}

    # Decode header
    header = json.loads(base64.urlsafe_b64decode(parts[0] + '=='))

    # Decode payload
    payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=='))

    # Convert timestamps to readable format
    if 'exp' in payload:
        payload['exp_readable'] = datetime.fromtimestamp(payload['exp']).isoformat()
    if 'iat' in payload:
        payload['iat_readable'] = datetime.fromtimestamp(payload['iat']).isoformat()
    if 'auth_time' in payload:
        payload['auth_time_readable'] = datetime.fromtimestamp(payload['auth_time']).isoformat()

    return {
        "header": header,
        "payload": payload,
        "signature": parts[2][:20] + "..." if len(parts[2]) > 20 else parts[2]
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        # Read from stdin
        token = sys.stdin.read().strip()

    if not token:
        print("Usage: python decode_jwt.py <token>")
        print("   or: echo $TOKEN | python decode_jwt.py")
        sys.exit(1)

    try:
        decoded = decode_jwt(token)
        print(json.dumps(decoded, indent=2))
    except Exception as e:
        print(f"Error decoding token: {e}")
        sys.exit(1)
