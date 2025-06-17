#!/bin/bash
# IAM Permissions Testing Script
# Tests all required permissions for CLARITY platform

set -e

echo "🔒 CLARITY IAM Permissions Test Suite"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0

# Function to test a permission
test_permission() {
    local test_name=$1
    local command=$2
    local expected_result=$3
    
    echo -n "Testing: $test_name... "
    
    if eval "$command" &> /dev/null; then
        if [ "$expected_result" = "allow" ]; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((PASSED++))
        else
            echo -e "${RED}✗ FAIL (Should be denied)${NC}"
            ((FAILED++))
        fi
    else
        if [ "$expected_result" = "deny" ]; then
            echo -e "${GREEN}✓ PASS (Correctly denied)${NC}"
            ((PASSED++))
        else
            echo -e "${RED}✗ FAIL (Should be allowed)${NC}"
            ((FAILED++))
        fi
    fi
}

echo "1. Testing S3 Permissions"
echo "------------------------"
test_permission "Read ML models bucket" \
    "aws s3 ls s3://clarity-ml-models-124355672559/" \
    "allow"

test_permission "Read health data bucket" \
    "aws s3 ls s3://clarity-health-data-storage/" \
    "allow"

test_permission "Write to health data bucket" \
    "echo 'test' | aws s3 cp - s3://clarity-health-data-storage/test-write.txt" \
    "allow"

test_permission "Access unauthorized bucket" \
    "aws s3 ls s3://some-other-bucket/" \
    "deny"

echo ""
echo "2. Testing DynamoDB Permissions"
echo "------------------------------"
test_permission "Query health data table" \
    "aws dynamodb describe-table --table-name clarity-health-data" \
    "allow"

test_permission "Access unauthorized table" \
    "aws dynamodb describe-table --table-name some-other-table" \
    "deny"

echo ""
echo "3. Testing Cognito Permissions"
echo "-----------------------------"
test_permission "List users in pool" \
    "aws cognito-idp list-users --user-pool-id us-east-1_efXaR5EcP --limit 1" \
    "allow"

test_permission "Access unauthorized pool" \
    "aws cognito-idp list-users --user-pool-id us-east-1_unauthorized --limit 1" \
    "deny"

echo ""
echo "4. Testing Secrets Manager"
echo "-------------------------"
test_permission "Read Gemini API key" \
    "aws secretsmanager get-secret-value --secret-id clarity/gemini-api-key --query SecretString" \
    "allow"

test_permission "Access unauthorized secret" \
    "aws secretsmanager get-secret-value --secret-id unauthorized/secret" \
    "deny"

echo ""
echo "5. Testing CloudWatch"
echo "--------------------"
test_permission "Write logs" \
    "aws logs describe-log-streams --log-group-name /ecs/clarity-backend --limit 1" \
    "allow"

test_permission "Put metrics" \
    "aws cloudwatch put-metric-data --namespace Clarity/Auth --metric-name TestMetric --value 1" \
    "allow"

echo ""
echo "===================================="
echo "Test Results:"
echo "✅ Passed: $PASSED"
echo "❌ Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! IAM permissions are correctly configured.${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please review IAM permissions.${NC}"
    exit 1
fi