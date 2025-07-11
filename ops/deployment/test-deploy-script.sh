#!/bin/bash

# Test script for deploy.sh JSON manipulation functionality
# This prevents the deployment script from breaking again

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Testing deployment script JSON manipulation...${NC}"

# Test 1: Verify jq is available
echo -e "\n${YELLOW}Test 1: Checking jq availability...${NC}"
if command -v jq >/dev/null 2>&1; then
    echo -e "${GREEN}✅ jq is available${NC}"
else
    echo -e "${RED}❌ jq is not available - deployment script will fail${NC}"
    exit 1
fi

# Test 2: Test JSON manipulation with sample Docker image URL
echo -e "\n${YELLOW}Test 2: Testing JSON manipulation...${NC}"

# Create a test task definition similar to our real one
TEST_JSON=$(cat << 'EOF'
{
  "family": "test-family",
  "containerDefinitions": [
    {
      "name": "test-container",
      "image": "IMAGE_PLACEHOLDER",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
EOF
)

# Test image URL (complex like our real ECR URLs)
TEST_IMAGE="124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend:bd6bc5b"

# Test the jq command from our script
echo "$TEST_JSON" > /tmp/test-task-def.json
RESULT=$(cat /tmp/test-task-def.json | jq --arg image "$TEST_IMAGE" '.containerDefinitions[0].image = $image')

# Verify the replacement worked
EXTRACTED_IMAGE=$(echo "$RESULT" | jq -r '.containerDefinitions[0].image')

if [ "$EXTRACTED_IMAGE" = "$TEST_IMAGE" ]; then
    echo -e "${GREEN}✅ JSON manipulation works correctly${NC}"
    echo -e "${GREEN}   Input image: $TEST_IMAGE${NC}"
    echo -e "${GREEN}   Output image: $EXTRACTED_IMAGE${NC}"
else
    echo -e "${RED}❌ JSON manipulation failed${NC}"
    echo -e "${RED}   Expected: $TEST_IMAGE${NC}"
    echo -e "${RED}   Got: $EXTRACTED_IMAGE${NC}"
    exit 1
fi

# Test 3: Verify the actual task definition file can be processed
echo -e "\n${YELLOW}Test 3: Testing with actual task definition file...${NC}"

if [ -f "ops/ecs-task-definition.json" ]; then
    # Test with our actual task definition
    ACTUAL_RESULT=$(cat ops/ecs-task-definition.json | jq --arg image "$TEST_IMAGE" '.containerDefinitions[0].image = $image')
    ACTUAL_EXTRACTED=$(echo "$ACTUAL_RESULT" | jq -r '.containerDefinitions[0].image')
    
    if [ "$ACTUAL_EXTRACTED" = "$TEST_IMAGE" ]; then
        echo -e "${GREEN}✅ Actual task definition processes correctly${NC}"
    else
        echo -e "${RED}❌ Actual task definition processing failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Task definition file not found: ops/ecs-task-definition.json${NC}"
    exit 1
fi

# Test 4: Test with problematic characters that would break sed
echo -e "\n${YELLOW}Test 4: Testing with special characters that break sed...${NC}"

PROBLEMATIC_IMAGE="repo/app:v1.0-release/special&chars|with$slashes"
PROBLEMATIC_RESULT=$(echo "$TEST_JSON" | jq --arg image "$PROBLEMATIC_IMAGE" '.containerDefinitions[0].image = $image')
PROBLEMATIC_EXTRACTED=$(echo "$PROBLEMATIC_RESULT" | jq -r '.containerDefinitions[0].image')

if [ "$PROBLEMATIC_EXTRACTED" = "$PROBLEMATIC_IMAGE" ]; then
    echo -e "${GREEN}✅ Special characters handled correctly${NC}"
    echo -e "${GREEN}   Problematic image: $PROBLEMATIC_IMAGE${NC}"
else
    echo -e "${RED}❌ Special characters not handled${NC}"
    exit 1
fi

# Test 5: Test the actual register_task_definition function structure
echo -e "\n${YELLOW}Test 5: Testing deployment script function...${NC}"

# Source the deploy script and test the function (mock mode)
if grep -q "register_task_definition()" ops/deploy.sh; then
    echo -e "${GREEN}✅ register_task_definition function exists${NC}"
else
    echo -e "${RED}❌ register_task_definition function missing${NC}"
    exit 1
fi

# Check that we're using jq instead of sed
if grep -q "jq --arg image" ops/deploy.sh; then
    echo -e "${GREEN}✅ Deployment script uses jq for JSON manipulation${NC}"
else
    echo -e "${RED}❌ Deployment script not using jq${NC}"
    exit 1
fi

# Clean up test files
rm -f /tmp/test-task-def.json

echo -e "\n${GREEN}🎉 All tests passed! Deployment script should work reliably.${NC}"
echo -e "${GREEN}💡 The script now uses jq instead of sed for robust JSON manipulation.${NC}" 