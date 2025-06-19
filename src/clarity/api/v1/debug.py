"""Debug endpoints for request inspection."""

# removed - breaks FastAPI

import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/debug",
    tags=["debug"],
)


@router.post("/capture-raw-request")
async def capture_raw_request(request: Request) -> dict[str, Any]:
    """Capture and analyze raw request body to debug JSON parsing issues."""
    try:
        # Get raw body bytes
        body_bytes = await request.body()

        # Prepare response with proper typing
        response: dict[str, Any] = {
            "headers": dict(request.headers),
            "method": request.method,
            "url": str(request.url),
            "body_info": {
                "length_bytes": len(body_bytes),
                "raw_bytes": body_bytes.hex(),  # Hex representation
            },
        }

        # Try to decode as UTF-8
        try:
            body_str = body_bytes.decode("utf-8")
            response["body_info"]["decoded_string"] = body_str
            response["body_info"]["decoded_success"] = True

            # Analyze character at position 55 if it exists
            debug_position = 55  # Position to analyze for debugging
            if len(body_str) > debug_position:
                char_at_55 = body_str[debug_position]
                response["body_info"]["position_55"] = {
                    "character": char_at_55,
                    "ascii_code": ord(char_at_55),
                    "hex": hex(ord(char_at_55)),
                    "context": body_str[max(0, 50) : min(len(body_str), 60)],
                }

            # Try to parse as JSON
            try:
                body_json = json.loads(body_str)
                response["body_info"]["json_valid"] = True
                response["body_info"]["json_data"] = body_json

                # Log for debugging
                logger.warning("✅ DEBUG: Valid JSON received")
                logger.warning("JSON: %s", json.dumps(body_json, indent=2))

            except json.JSONDecodeError as e:
                response["body_info"]["json_valid"] = False
                response["body_info"]["json_error"] = {
                    "message": str(e),
                    "position": e.pos,
                    "line": e.lineno,
                    "column": e.colno,
                }

                # Show context around error position
                if e.pos and e.pos < len(body_str):
                    start = max(0, e.pos - 10)
                    end = min(len(body_str), e.pos + 10)
                    response["body_info"]["json_error"]["context"] = {
                        "before": body_str[start : e.pos],
                        "at": body_str[e.pos] if e.pos < len(body_str) else "EOF",
                        "after": (
                            body_str[e.pos + 1 : end]
                            if e.pos < len(body_str) - 1
                            else ""
                        ),
                    }

                logger.exception(
                    "❌ DEBUG: JSON decode error at position %s: %s", e.pos, e
                )

        except UnicodeDecodeError as e:
            response["body_info"]["decoded_success"] = False
            response["body_info"]["decode_error"] = str(e)
            logger.exception("❌ DEBUG: UTF-8 decode error: %s", e)

        return response

    except Exception as e:
        logger.exception("Debug endpoint error")
        return {"error": str(e), "type": type(e).__name__}


@router.post("/echo-login")
async def echo_login_request(request: Request) -> JSONResponse:
    """Echo back the exact login request for debugging."""
    body_bytes = await request.body()

    # Log everything
    logger.warning("=" * 60)
    logger.warning("ECHO LOGIN REQUEST")
    logger.warning("=" * 60)
    logger.warning("Headers: %s", dict(request.headers))
    logger.warning("Body bytes: %r", body_bytes)
    logger.warning("Body hex: %s", body_bytes.hex())

    try:
        body_str = body_bytes.decode("utf-8")
        logger.warning("Body string: %s", body_str)

        # Return mock auth response to keep client happy
        return JSONResponse(
            {
                "access_token": "debug_token",
                "refresh_token": "debug_refresh",
                "token_type": "bearer",
                "expires_in": 3600,
                "scope": "full_access",
            }
        )
    except Exception as e:
        logger.exception("Echo error: %s", e)
        return JSONResponse(status_code=400, content={"error": str(e)})
