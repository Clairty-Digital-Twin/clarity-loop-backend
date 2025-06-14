"""Request logging middleware for debugging."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests for debugging."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Log request details before processing."""
        # Log basic request info
        logger.warning(f"🔍 REQUEST: {request.method} {request.url.path}")
        logger.warning(f"  Headers: {dict(request.headers)}")

        # For POST/PUT/PATCH requests, try to log the body
        if request.method in {"POST", "PUT", "PATCH"}:
            try:
                # Note: After reading body here, we need to make it available to the endpoint
                body_bytes = await request.body()
                logger.warning(f"  Body length: {len(body_bytes)} bytes")

                # Try to decode as UTF-8
                try:
                    body_str = body_bytes.decode("utf-8")
                    logger.warning(f"  Body preview: {body_str[:200]}...")

                    # Try to parse as JSON
                    try:
                        body_json = json.loads(body_str)
                        logger.warning(
                            f"  Parsed JSON: {json.dumps(body_json, indent=2)}"
                        )
                    except json.JSONDecodeError:
                        logger.warning("  Body is not valid JSON")
                except UnicodeDecodeError:
                    logger.warning(
                        f"  Body is not UTF-8, hex preview: {body_bytes.hex()[:100]}..."
                    )

                # Store body for the endpoint to use
                request._body = body_bytes

            except Exception as e:
                logger.exception(f"  Failed to read request body: {e}")

        # Process the request
        response = await call_next(request)

        # Log response status
        logger.warning(f"  Response: {response.status_code}")

        return response
