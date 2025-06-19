"""Custom OpenAPI schema generation."""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI) -> dict[str, Any]:
    """Generate custom OpenAPI schema with security schemes and enhancements."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
        contact=app.contact,
        license_info=app.license_info,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token authentication. Get token from /api/v1/auth/login",
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key authentication for service-to-service communication",
        },
    }

    # Add global security (can be overridden per endpoint)
    openapi_schema["security"] = [{"BearerAuth": []}]

    # Add common response schemas
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}

    # Add error response schema
    openapi_schema["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "error": {
                "type": "string",
                "description": "Error message",
                "example": "Invalid credentials",
            },
            "code": {
                "type": "string",
                "description": "Error code",
                "example": "AUTH_001",
            },
            "details": {
                "type": "object",
                "description": "Additional error details",
                "additionalProperties": True,
            },
        },
        "required": ["error"],
    }

    # Add paginated response schema
    openapi_schema["components"]["schemas"]["PaginationMeta"] = {
        "type": "object",
        "properties": {
            "total": {
                "type": "integer",
                "description": "Total number of items",
                "example": 100,
            },
            "page": {
                "type": "integer",
                "description": "Current page number",
                "example": 1,
            },
            "per_page": {
                "type": "integer",
                "description": "Items per page",
                "example": 20,
            },
            "total_pages": {
                "type": "integer",
                "description": "Total number of pages",
                "example": 5,
            },
        },
        "required": ["total", "page", "per_page", "total_pages"],
    }

    # Process all paths to add common error responses and fix issues
    for path, methods in openapi_schema.get("paths", {}).items():
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue

            # Add operationId if missing
            if "operationId" not in operation:
                operation["operationId"] = (
                    f"{method}_{path.replace('/', '_').strip('_')}"
                )

            # Add common error responses
            if "responses" not in operation:
                operation["responses"] = {}

            # Add 401 for authenticated endpoints
            if path not in {"/", "/health", "/metrics"} and not path.startswith(
                "/api/v1/auth/"
            ):
                if "401" not in operation["responses"]:
                    operation["responses"]["401"] = {
                        "description": "Unauthorized - Invalid or missing authentication",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                },
                                "example": {
                                    "error": "Invalid or expired token",
                                    "code": "AUTH_001",
                                },
                            }
                        },
                    }

                if "403" not in operation["responses"]:
                    operation["responses"]["403"] = {
                        "description": "Forbidden - Insufficient permissions",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                },
                                "example": {
                                    "error": "Insufficient permissions for this operation",
                                    "code": "AUTH_002",
                                },
                            }
                        },
                    }

            # Add 500 for all endpoints
            if "500" not in operation["responses"]:
                operation["responses"]["500"] = {
                    "description": "Internal Server Error",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            "example": {
                                "error": "An unexpected error occurred",
                                "code": "INTERNAL_001",
                            },
                        }
                    },
                }

            # Set security for endpoints
            if path in {
                "/",
                "/health",
                "/metrics",
                "/docs",
                "/redoc",
                "/openapi.json",
            } or path.startswith("/api/v1/auth/"):
                # Public endpoints
                operation["security"] = []
            elif "security" not in operation:
                # Protected endpoints use default security
                operation["security"] = [{"BearerAuth": []}]

    # Cache the schema
    app.openapi_schema = openapi_schema
    return openapi_schema
