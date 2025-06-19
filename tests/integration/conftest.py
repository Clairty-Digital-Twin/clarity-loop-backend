"""Configure integration tests to use mock services."""

import os

# Set this BEFORE any imports
os.environ["SKIP_EXTERNAL_SERVICES"] = "true"
os.environ["ENABLE_SELF_SIGNUP"] = "true"
