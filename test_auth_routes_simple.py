#!/usr/bin/env python3
"""Debug test to check auth route registration - simplified version."""

import sys
import os

# Suppress warnings
os.environ["ENABLE_SELF_SIGNUP"] = "true"
os.environ["SKIP_EXTERNAL_SERVICES"] = "true"

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Test 1: Direct import and check
print("=== Test 1: Direct router import ===")
from clarity.api.v1.auth import router as auth_router
print(f"Auth router imported: {auth_router}")
print(f"Router routes: {[r.path for r in auth_router.routes]}")

# Test 2: Create app with just auth router
print("\n=== Test 2: App with just auth router ===")
app1 = FastAPI()
app1.include_router(auth_router, prefix="/api/v1/auth")
client1 = TestClient(app1)
print(f"App routes: {[r.path for r in app1.routes]}")

# Test auth health endpoint
response = client1.get("/api/v1/auth/health")
print(f"Health check response: {response.status_code}")

# Test 3: Import and use api_router
print("\n=== Test 3: Full API router ===")
from clarity.api.v1.router import api_router
app2 = FastAPI()
app2.include_router(api_router, prefix="/api/v1")
client2 = TestClient(app2)
print(f"API router routes: {[r.path for r in api_router.routes]}")

# Test auth health via full router
response = client2.get("/api/v1/auth/health")
print(f"Health check response via API router: {response.status_code}")

# Test 4: Use create_app but without lifespan context
print("\n=== Test 4: Testing create_app (no context) ===")
from clarity.main import create_app
app3 = create_app()
# Don't use context manager to avoid lifespan issues
client3 = TestClient(app3)

# Check what routes are registered
print("All app routes:")
for route in app3.routes:
    if hasattr(route, 'path'):
        print(f"  {route.path} - {route.methods if hasattr(route, 'methods') else 'N/A'}")

print("\nAuth-specific routes:")
auth_routes = [r for r in app3.routes if hasattr(r, 'path') and '/auth' in r.path]
for route in auth_routes:
    print(f"  {route.path} - {route.methods}")

# Test health endpoint
response = client3.get("/api/v1/auth/health")
print(f"\nHealth check response via create_app: {response.status_code}")

# Test a 404
response = client3.get("/api/v1/auth/nonexistent")  
print(f"Nonexistent endpoint: {response.status_code}")

print("\n=== Test 5: Check if routes are actually missing ===")
# List all registered paths
all_paths = [r.path for r in app3.routes if hasattr(r, 'path')]
print(f"Total routes: {len(all_paths)}")
print(f"Auth routes found: {[p for p in all_paths if 'auth' in p]}")

# Try to access auth health directly without prefix
response = client3.get("/auth/health")
print(f"Direct /auth/health: {response.status_code}")

# Clean up
client1.close()
client2.close()
client3.close()