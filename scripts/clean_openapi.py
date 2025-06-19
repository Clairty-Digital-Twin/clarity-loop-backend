#!/usr/bin/env python3
"""Clean and enhance OpenAPI spec according to best practices."""

from collections import OrderedDict
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def clean_openapi_spec() -> None:
    """Clean and enhance the OpenAPI spec."""
    # Load the generated spec
    with Path("openapi.json").open(encoding="utf-8") as f:
        spec = json.load(f)

    # 1. Add security schemes
    if "components" not in spec:
        spec["components"] = {}

    spec["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token authentication",
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key authentication",
        },
    }

    # 2. Add global security requirement (can be overridden per endpoint)
    spec["security"] = [{"BearerAuth": []}]

    # 3. Clean up paths and tags
    cleaned_paths = OrderedDict()

    for original_path, methods in spec["paths"].items():
        path = original_path
        # Skip if path is malformed
        if not path:
            continue

        # Fix double test paths
        if "/test/test/" in path:
            path = path.replace("/test/test/", "/test/")

        # Remove trailing slashes except for root paths
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        cleaned_methods = {}
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue

            # Remove duplicate "API v1" tag
            if "tags" in operation:
                operation["tags"] = [
                    tag for tag in operation["tags"] if tag != "API v1"
                ]
                # Ensure at least one tag
                if not operation["tags"]:
                    operation["tags"] = ["default"]

            # Add security to authenticated endpoints
            if path not in {"/", "/health", "/metrics"} and not path.startswith(
                "/api/v1/auth/"
            ):
                if "security" not in operation:
                    operation["security"] = [{"BearerAuth": []}]
            else:
                # Public endpoints
                operation["security"] = []

            # Add error responses
            if "responses" not in operation:
                operation["responses"] = {}

            # Add common error responses
            if "401" not in operation["responses"]:
                operation["responses"]["401"] = {
                    "description": "Unauthorized - Invalid or missing authentication",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    },
                }

            if "403" not in operation["responses"]:
                operation["responses"]["403"] = {
                    "description": "Forbidden - Insufficient permissions",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    },
                }

            if "500" not in operation["responses"]:
                operation["responses"]["500"] = {
                    "description": "Internal Server Error",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    },
                }

            cleaned_methods[method] = operation

        if cleaned_methods:
            cleaned_paths[path] = cleaned_methods

    spec["paths"] = cleaned_paths

    # 4. Add error response schema
    if "schemas" not in spec["components"]:
        spec["components"]["schemas"] = {}

    spec["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "error": {"type": "string", "description": "Error message"},
            "code": {"type": "string", "description": "Error code"},
            "details": {
                "type": "object",
                "description": "Additional error details",
                "additionalProperties": True,
            },
        },
        "required": ["error"],
    }

    # 5. Enhance info section
    spec["info"]["x-logo"] = {
        "url": "https://clarity.novamindnyc.com/logo.png",
        "altText": "CLARITY Logo",
    }

    spec["info"]["contact"] = {
        "name": "CLARITY Support",
        "email": "support@clarity.novamindnyc.com",
        "url": "https://clarity.novamindnyc.com",
    }

    spec["info"]["license"] = {
        "name": "Proprietary",
        "url": "https://clarity.novamindnyc.com/license",
    }

    # 6. Add servers
    spec["servers"] = [
        {
            "url": "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com",
            "description": "Production server (AWS ALB)",
        },
        {"url": "http://localhost:8000", "description": "Local development server"},
    ]

    # 7. Add tags with descriptions
    spec["tags"] = [
        {
            "name": "authentication",
            "description": "User authentication and authorization endpoints",
        },
        {"name": "health-data", "description": "Health data management and retrieval"},
        {"name": "healthkit", "description": "Apple HealthKit data integration"},
        {
            "name": "pat-analysis",
            "description": "Physical Activity Test (PAT) analysis endpoints",
        },
        {"name": "ai-insights", "description": "AI-powered health insights generation"},
        {"name": "metrics", "description": "Health metrics and statistics"},
        {"name": "websocket", "description": "WebSocket real-time communication"},
        {"name": "debug", "description": "Debug endpoints (development only)"},
        {"name": "test", "description": "Test endpoints for API validation"},
    ]

    # Write cleaned spec
    with Path("openapi-cleaned.json").open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print("✅ OpenAPI spec cleaned and saved to openapi-cleaned.json")

    # Also create YAML version
    if yaml is not None:
        with Path("openapi-cleaned.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        print("✅ YAML version saved to openapi-cleaned.yaml")
    else:
        print("⚠️  PyYAML not installed, skipping YAML generation")


if __name__ == "__main__":
    clean_openapi_spec()
