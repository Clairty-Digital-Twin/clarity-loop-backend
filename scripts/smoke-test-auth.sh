#!/bin/bash

# Auth Smoke Test Script
# Verifies all auth endpoints return expected status codes
# Usage: ./smoke-test-auth.sh [BASE_URL]
# Example: ./smoke-test-auth.sh http://localhost:8000

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base URL (default to localhost)
BASE_URL="${1:-http://localhost:8000}"
API_URL="${BASE_URL}/api/v1"

# Test counter
PASSED=0
FAILED=0

# Generate unique email for testing
TEST_EMAIL="test-$(date +%s)@example.com"
GOOD_PASSWORD="SecurePassword123!"
WEAK_PASSWORD="nocaps123"

echo "🧪 Running Auth Smoke Tests against: $BASE_URL"
echo "================================================"

# Function to run a test
run_test() {
    local test_name="$1"
    local expected_code="$2"
    local curl_cmd="$3"
    
    echo -n "Testing $test_name... "
    
    # Execute curl and capture response code
    response_code=$(eval "$curl_cmd")
    
    if [ "$response_code" = "$expected_code" ]; then
        echo -e "${GREEN}✓ PASS${NC} (Expected: $expected_code, Got: $response_code)"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC} (Expected: $expected_code, Got: $response_code)"
        ((FAILED++))
        
        # If we got a 500 error, try to get the response body for debugging
        if [ "$response_code" = "500" ]; then
            echo -e "${YELLOW}  Response body:${NC}"
            eval "${curl_cmd//-o \/dev\/null -w '%{http_code}'/-s}" | jq . 2>/dev/null || echo "  (Could not parse response)"
        fi
    fi
}

# Test 1: Health Check
run_test "Health Check" "200" \
    "curl -s -o /dev/null -w '%{http_code}' -X GET '${API_URL}/health'"

# Test 2: Registration with valid password
run_test "Registration - Valid Password" "201" \
    "curl -s -o /dev/null -w '%{http_code}' \
    -X POST '${API_URL}/auth/register' \
    -H 'Content-Type: application/json' \
    -d '{\"email\":\"${TEST_EMAIL}\",\"display_name\":\"Test User\",\"password\":\"${GOOD_PASSWORD}\"}'"

# Test 3: Registration with weak password (no uppercase)
run_test "Registration - Weak Password" "400" \
    "curl -s -o /dev/null -w '%{http_code}' \
    -X POST '${API_URL}/auth/register' \
    -H 'Content-Type: application/json' \
    -d '{\"email\":\"test2-$(date +%s)@example.com\",\"display_name\":\"Test User\",\"password\":\"${WEAK_PASSWORD}\"}'"

# Test 4: Registration with duplicate email
run_test "Registration - Duplicate Email" "409" \
    "curl -s -o /dev/null -w '%{http_code}' \
    -X POST '${API_URL}/auth/register' \
    -H 'Content-Type: application/json' \
    -d '{\"email\":\"${TEST_EMAIL}\",\"display_name\":\"Test User\",\"password\":\"${GOOD_PASSWORD}\"}'"

# Test 5: Login with correct credentials
run_test "Login - Valid Credentials" "200" \
    "curl -s -o /dev/null -w '%{http_code}' \
    -X POST '${API_URL}/auth/login' \
    -H 'Content-Type: application/json' \
    -d '{\"email\":\"${TEST_EMAIL}\",\"password\":\"${GOOD_PASSWORD}\"}'"

# Test 6: Login with wrong password
run_test "Login - Wrong Password" "401" \
    "curl -s -o /dev/null -w '%{http_code}' \
    -X POST '${API_URL}/auth/login' \
    -H 'Content-Type: application/json' \
    -d '{\"email\":\"${TEST_EMAIL}\",\"password\":\"WrongPassword123!\"}'"

# Test 7: Login with non-existent email
run_test "Login - Non-existent Email" "401" \
    "curl -s -o /dev/null -w '%{http_code}' \
    -X POST '${API_URL}/auth/login' \
    -H 'Content-Type: application/json' \
    -d '{\"email\":\"nonexistent@example.com\",\"password\":\"${GOOD_PASSWORD}\"}'"

# Test 8: Invalid JSON format
run_test "Registration - Invalid JSON" "422" \
    "curl -s -o /dev/null -w '%{http_code}' \
    -X POST '${API_URL}/auth/register' \
    -H 'Content-Type: application/json' \
    -d '{\"email\":\"invalid-email-format\",\"password\":\"short\"}'"

echo "================================================"
echo "Test Summary:"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed! Safe to deploy.${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed! Do not deploy.${NC}"
    exit 1
fi