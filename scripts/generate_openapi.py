#!/usr/bin/env python3
"""Generate OpenAPI spec from FastAPI app."""

import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set required env vars to avoid AWS initialization
os.environ["SKIP_AWS_INIT"] = "true"
os.environ["ENABLE_AUTH"] = "false"

from src.clarity.main import app

# Generate OpenAPI spec
openapi_spec = app.openapi()

# Write to file
with Path("openapi.json").open("w", encoding="utf-8") as f:
    json.dump(openapi_spec, f, indent=2)

print("OpenAPI spec generated: openapi.json")
