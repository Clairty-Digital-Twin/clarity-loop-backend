#!/usr/bin/env python3
"""Enhanced debug script to test iOS login issues."""

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
import uvicorn

app = FastAPI()


class UserLoginRequest(BaseModel):
    """Login request model."""

    email: EmailStr
    password: str


@app.post("/api/v1/auth/login")
async def debug_login(request: Request):
    """Debug endpoint that logs everything about the request."""
    # 1. Log all headers
    for _key, _value in request.headers.items():
        pass

    # 2. Get raw body
    try:
        body_bytes = await request.body()

        # 3. Try to decode as UTF-8
        try:
            body_str = body_bytes.decode("utf-8")

            # 4. Show detailed character analysis
            for i, _char in enumerate(body_str[:100]):  # First 100 chars
                if i == 55:  # The problematic position
                    pass

            # 5. Try to parse as JSON
            try:
                body_json = json.loads(body_str)

                # 6. Try Pydantic validation
                try:
                    login_data = UserLoginRequest(**body_json)

                    # Return success response
                    return JSONResponse(
                        {
                            "access_token": "debug_token_12345",
                            "refresh_token": "debug_refresh_67890",
                            "token_type": "bearer",
                            "expires_in": 3600,
                            "scope": "full_access",
                            "debug_info": {
                                "request_received": True,
                                "json_parsed": True,
                                "pydantic_validated": True,
                                "email_received": login_data.email,
                            },
                        }
                    )

                except Exception as e:
                    return JSONResponse(
                        status_code=422,
                        content={
                            "error": "Pydantic validation failed",
                            "details": str(e),
                            "debug_info": {
                                "body_received": body_str,
                                "json_parsed": True,
                                "pydantic_validated": False,
                            },
                        },
                    )

            except json.JSONDecodeError as e:
                if e.pos < len(body_str):
                    pass

                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "JSON parsing failed",
                        "details": str(e),
                        "position": e.pos,
                        "debug_info": {
                            "body_length": len(body_str),
                            "body_preview": body_str[:100],
                            "error_context": (
                                body_str[max(0, e.pos - 20) : e.pos + 20]
                                if e.pos < len(body_str)
                                else None
                            ),
                        },
                    },
                )

        except UnicodeDecodeError as e:
            hex_preview = body_bytes.hex()[:200]

            return JSONResponse(
                status_code=400,
                content={
                    "error": "UTF-8 decoding failed",
                    "details": str(e),
                    "debug_info": {
                        "body_bytes_length": len(body_bytes),
                        "hex_preview": hex_preview,
                    },
                },
            )

    except Exception as e:
        import traceback

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": "Unexpected error",
                "details": str(e),
                "type": type(e).__name__,
            },
        )


@app.post("/test/echo")
async def echo_request(request: Request):
    """Simple echo endpoint for testing."""
    body = await request.body()
    return {
        "method": request.method,
        "path": request.url.path,
        "headers": dict(request.headers),
        "body_length": len(body),
        "body_utf8": body.decode("utf-8", errors="replace"),
        "body_hex": body.hex()[:100] + "..." if len(body) > 50 else body.hex(),
    }


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
