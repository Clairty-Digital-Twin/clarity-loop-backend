#!/bin/bash

# General Smoke Test Script
# Runs all smoke tests for the Clarity backend
# Usage: ./smoke-test.sh [BASE_URL]
# Example: ./smoke-test.sh http://localhost:8000

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base URL (default to localhost)
BASE_URL="${1:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔥 Running All Smoke Tests against: $BASE_URL"
echo "=============================================="

TOTAL_FAILED=0

# Run auth smoke tests
echo -e "\n${YELLOW}Auth Smoke Tests:${NC}"
if "$SCRIPT_DIR/smoke-test-auth-suite.sh" "$BASE_URL"; then
    echo -e "${GREEN}✓ Auth tests passed${NC}"
else
    echo -e "${RED}✗ Auth tests failed${NC}"
    ((TOTAL_FAILED++))
fi

# Add more smoke test suites here as they're created
# Example:
# echo -e "\n${YELLOW}Health Data Smoke Tests:${NC}"
# if "$SCRIPT_DIR/smoke-test-health.sh" "$BASE_URL"; then
#     echo -e "${GREEN}✓ Health data tests passed${NC}"
# else
#     echo -e "${RED}✗ Health data tests failed${NC}"
#     ((TOTAL_FAILED++))
# fi

echo -e "\n=============================================="

if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All smoke tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ $TOTAL_FAILED test suite(s) failed!${NC}"
    exit 1
fi