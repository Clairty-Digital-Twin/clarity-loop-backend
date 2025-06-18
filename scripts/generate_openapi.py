#!/usr/bin/env python3
"""Generate OpenAPI spec from FastAPI app."""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set required env vars to avoid AWS initialization
os.environ["SKIP_AWS_INIT"] = "true"
os.environ["ENABLE_AUTH"] = "false"

from src.clarity.main import app

# Generate OpenAPI spec
openapi_spec = app.openapi()

# Write to file
with open("openapi.json", "w") as f:
    json.dump(openapi_spec, f, indent=2)

print("OpenAPI spec generated: openapi.json")
