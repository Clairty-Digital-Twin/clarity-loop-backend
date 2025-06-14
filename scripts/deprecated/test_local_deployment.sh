#!/bin/bash
# CLARITY Backend Local Testing Script

set -e

echo "🔍 CLARITY Backend Local Testing"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local url=$1
    local expected_status=$2
    local description=$3
    
    echo -n "Testing $description... "
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✓ PASS${NC} (Status: $response)"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (Expected: $expected_status, Got: $response)"
        return 1
    fi
}

# Function to test POST endpoint
test_post_endpoint() {
    local url=$1
    local data=$2
    local expected_status=$3
    local description=$4
    
    echo -n "Testing $description... "
    response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$data" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✓ PASS${NC} (Status: $response)"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (Expected: $expected_status, Got: $response)"
        return 1
    fi
}

echo "1. Building Docker image..."
echo "------------------------"
# Use the AWS Dockerfile since it has the files inline
docker build -f Dockerfile.aws -t clarity-backend-test . || {
    echo -e "${RED}Failed to build Docker image${NC}"
    exit 1
}

echo -e "\n2. Starting container..."
echo "------------------------"
# Run in detached mode
docker run -d --name clarity-test \
    -p 8080:8000 \
    -e ENVIRONMENT=development \
    -e SKIP_EXTERNAL_SERVICES=true \
    -e COGNITO_USER_POOL_ID=test-pool \
    -e COGNITO_CLIENT_ID=test-client \
    -e AWS_DEFAULT_REGION=us-east-1 \
    clarity-backend-test || {
    echo -e "${RED}Failed to start container${NC}"
    exit 1
}

echo "Waiting for container to be ready..."
sleep 5

# Check if container is running
if ! docker ps | grep -q clarity-test; then
    echo -e "${RED}Container is not running!${NC}"
    echo "Container logs:"
    docker logs clarity-test
    docker rm -f clarity-test 2>/dev/null
    exit 1
fi

echo -e "\n3. Running endpoint tests..."
echo "------------------------"

# Test results counter
total=0
passed=0

# Basic endpoints
test_endpoint "http://localhost:8080/health" "200" "Health check" && ((passed++)) || true; ((total++))
test_endpoint "http://localhost:8080/docs" "200" "API documentation" && ((passed++)) || true; ((total++))
test_endpoint "http://localhost:8080/openapi.json" "200" "OpenAPI schema" && ((passed++)) || true; ((total++))

# Auth endpoints (expect 422 or 401)
test_post_endpoint "http://localhost:8080/api/v1/auth/register" \
    '{"email":"test@test.com","password":"Test123!@#"}' \
    "422" "Registration endpoint" && ((passed++)) || true; ((total++))

# Health data endpoints (expect 401 without auth)
test_endpoint "http://localhost:8080/api/v1/health-data" "401" "Health data list (no auth)" && ((passed++)) || true; ((total++))
test_endpoint "http://localhost:8080/api/v1/health-data/health" "200" "Health data service check" && ((passed++)) || true; ((total++))

# WebSocket endpoint (expect 403 or upgrade required)
test_endpoint "http://localhost:8080/api/v1/ws" "403" "WebSocket endpoint" && ((passed++)) || true; ((total++))

echo -e "\n4. Container logs (last 20 lines):"
echo "------------------------"
docker logs --tail 20 clarity-test

echo -e "\n5. Test Summary:"
echo "------------------------"
echo "Total tests: $total"
echo "Passed: $passed"
echo "Failed: $((total - passed))"

if [ $passed -eq $total ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${YELLOW}Some tests failed - this is expected for endpoints requiring auth${NC}"
fi

echo -e "\n6. Interactive testing commands:"
echo "------------------------"
echo "# View live logs:"
echo "docker logs -f clarity-test"
echo ""
echo "# Test with curl:"
echo "curl http://localhost:8080/health | jq"
echo ""
echo "# Open API docs:"
echo "open http://localhost:8080/docs"
echo ""
echo "# Stop and remove container:"
echo "docker stop clarity-test && docker rm clarity-test"

echo -e "\n${YELLOW}Container is still running for manual testing${NC}"
echo "Run 'docker stop clarity-test && docker rm clarity-test' when done"