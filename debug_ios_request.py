#!/usr/bin/env python3
"""Debug server to capture exact iOS request."""

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()


@app.post("/api/v1/auth/login")
async def debug_login(request: Request):
    """Capture and display the exact request from iOS."""
    print("=" * 60)
    print("iOS REQUEST RECEIVED")
    print("=" * 60)

    # Headers
    print("HEADERS:")
    for key, value in request.headers.items():
        print(f"  {key}: {value}")

    # Raw body
    body_bytes = await request.body()
    print(f"\nRAW BODY BYTES: {body_bytes}")
    print(f"BODY LENGTH: {len(body_bytes)} bytes")

    # Try to decode as string
    try:
        body_str = body_bytes.decode("utf-8")
        print(f"\nBODY AS STRING:\n{body_str}")

        # Show character at position 55 (where error occurred)
        if len(body_str) > 55:
            print(
                f"\nCHARACTER AT POSITION 55: '{body_str[55]}' (ASCII: {ord(body_str[55])})"
            )
            print(f"CONTEXT: ...{body_str[50:60]}...")

        # Try to parse as JSON
        try:
            body_json = json.loads(body_str)
            print(f"\nPARSED JSON:\n{json.dumps(body_json, indent=2)}")

            # Return success response to keep iOS app happy
            return JSONResponse(
                {
                    "access_token": "debug_token",
                    "refresh_token": "debug_refresh",
                    "token_type": "bearer",
                    "expires_in": 3600,
                    "scope": "full_access",
                }
            )

        except json.JSONDecodeError as e:
            print(f"\nJSON DECODE ERROR: {e}")
            print(f"Error at position: {e.pos}")
            return JSONResponse(
                status_code=422, content={"error": str(e), "position": e.pos}
            )

    except UnicodeDecodeError as e:
        print(f"\nUNICODE DECODE ERROR: {e}")
        return JSONResponse(
            status_code=400, content={"error": "Invalid UTF-8 encoding"}
        )


if __name__ == "__main__":
    print("Starting debug server on http://localhost:8001")
    print("Configure iOS app to use: http://YOUR_LOCAL_IP:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
