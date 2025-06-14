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
    # Headers
    for _key, _value in request.headers.items():
        pass

    # Raw body
    body_bytes = await request.body()

    # Try to decode as string
    try:
        body_str = body_bytes.decode("utf-8")

        # Show character at position 55 (where error occurred)
        if len(body_str) > 55:
            pass

        # Try to parse as JSON
        try:
            json.loads(body_str)  # Validate JSON format

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
            return JSONResponse(
                status_code=422, content={"error": str(e), "position": e.pos}
            )

    except UnicodeDecodeError:
        return JSONResponse(
            status_code=400, content={"error": "Invalid UTF-8 encoding"}
        )


if __name__ == "__main__":
    # Use environment variable or default to localhost for security
    import os

    host = os.getenv("DEBUG_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=8001)
