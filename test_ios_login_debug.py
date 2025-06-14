#!/usr/bin/env python3
"""Enhanced debug script to test iOS login issues."""

import asyncio
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, Request
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
    print("\n" + "=" * 80)
    print(f"[{datetime.now()}] iOS LOGIN REQUEST RECEIVED")
    print("=" * 80)

    # 1. Log all headers
    print("\n📋 HEADERS:")
    for key, value in request.headers.items():
        print(f"  {key}: {value}")

    # 2. Get raw body
    try:
        body_bytes = await request.body()
        print("\n📦 RAW BODY:")
        print(f"  Length: {len(body_bytes)} bytes")
        print(f"  Raw bytes: {body_bytes}")

        # 3. Try to decode as UTF-8
        try:
            body_str = body_bytes.decode("utf-8")
            print("\n📝 BODY AS STRING:")
            print(f"  {body_str}")

            # 4. Show detailed character analysis
            print("\n🔍 CHARACTER ANALYSIS:")
            for i, char in enumerate(body_str[:100]):  # First 100 chars
                print(
                    f"  Position {i}: '{char}' (ASCII: {ord(char)}, hex: {hex(ord(char))})"
                )
                if i == 55:  # The problematic position
                    print("  ⚠️  POSITION 55 DETECTED!")

            # 5. Try to parse as JSON
            print("\n🎯 JSON PARSING:")
            try:
                body_json = json.loads(body_str)
                print("  ✅ Successfully parsed JSON:")
                print(f"  {json.dumps(body_json, indent=2)}")

                # 6. Try Pydantic validation
                print("\n✨ PYDANTIC VALIDATION:")
                try:
                    login_data = UserLoginRequest(**body_json)
                    print("  ✅ Pydantic validation successful!")
                    print(f"  Email: {login_data.email}")
                    print(f"  Password: {'*' * len(login_data.password)}")

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
                    print(f"  ❌ Pydantic validation failed: {e}")
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
                print(f"  ❌ JSON parsing failed: {e}")
                print(f"  Error position: {e.pos}")
                if e.pos < len(body_str):
                    print(
                        f"  Character at error position: '{body_str[e.pos]}' (ASCII: {ord(body_str[e.pos])})"
                    )
                    print(f"  Context: ...{body_str[max(0, e.pos - 20):e.pos + 20]}...")

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
            print(f"  ❌ UTF-8 decoding failed: {e}")
            hex_preview = body_bytes.hex()[:200]
            print(f"  Hex preview: {hex_preview}")

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
        print(f"\n❌ UNEXPECTED ERROR: {e}")
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
    print("\n🚀 Starting iOS Debug Server")
    print("📍 Listening on: http://0.0.0.0:8001")
    print("🔧 Configure iOS app to use: http://YOUR_LOCAL_IP:8001")
    print("\n📌 Available endpoints:")
    print("  - POST /api/v1/auth/login - Debug login endpoint")
    print("  - POST /test/echo - Echo any request\n")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
