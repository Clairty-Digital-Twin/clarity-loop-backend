#!/usr/bin/env python3
"""Debug test to check auth route registration."""

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

# Test 4: Use create_app
print("\n=== Test 4: Using create_app ===")
from clarity.main import create_app
app3 = create_app()
client3 = TestClient(app3)

# Check what routes are registered
print(f"Main app routes: {[(r.path, r.methods) for r in app3.routes if '/auth' in r.path]}")

# Test health endpoint
response = client3.get("/api/v1/auth/health")
print(f"Health check response via create_app: {response.status_code}")

# Test 5: Check if it's a lifespan issue
print("\n=== Test 5: Testing with/without lifespan ===")
# Create app without lifespan
app4 = FastAPI()
from clarity.api.v1.router import api_router as v1_router
app4.include_router(v1_router, prefix="/api/v1")

# Test with regular client
client4 = TestClient(app4)
response = client4.get("/api/v1/auth/health")
print(f"Without lifespan: {response.status_code}")

# Test with app context
with TestClient(app3) as client5:
    response = client5.get("/api/v1/auth/health") 
    print(f"With TestClient context manager: {response.status_code}")