#!/bin/bash
# CLARITY Configuration Validator
# Ensures all AWS resources exist and are properly configured

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
REGION="us-east-1"

# Expected values (SOURCE OF TRUTH)
COGNITO_USER_POOL_ID="us-east-1_efXaR5EcP"
COGNITO_CLIENT_ID="7sm7ckrkovg78b03n1595euc71"
COGNITO_USER_POOL_NAME="clarity-users"
COGNITO_CLIENT_NAME="clarity-backend"

echo -e "${BLUE}🔍 CLARITY Configuration Validator${NC}"
echo -e "${BLUE}===================================${NC}"

# Function to check Cognito
check_cognito() {
    echo -e "\n${YELLOW}Checking AWS Cognito...${NC}"
    
    # Check if user pool exists
    if aws cognito-idp describe-user-pool \
        --user-pool-id $COGNITO_USER_POOL_ID \
        --region $REGION &> /dev/null; then
        echo -e "${GREEN}✅ User Pool exists: $COGNITO_USER_POOL_ID${NC}"
        
        # Get user pool details
        POOL_NAME=$(aws cognito-idp describe-user-pool \
            --user-pool-id $COGNITO_USER_POOL_ID \
            --region $REGION \
            --query 'UserPool.Name' \
            --output text)
        
        echo -e "   Name: $POOL_NAME"
        
        if [ "$POOL_NAME" != "$COGNITO_USER_POOL_NAME" ]; then
            echo -e "${YELLOW}   ⚠️  Warning: Expected name '$COGNITO_USER_POOL_NAME'${NC}"
        fi
    else
        echo -e "${RED}❌ User Pool NOT FOUND: $COGNITO_USER_POOL_ID${NC}"
        return 1
    fi
    
    # Check if client exists
    if aws cognito-idp describe-user-pool-client \
        --user-pool-id $COGNITO_USER_POOL_ID \
        --client-id $COGNITO_CLIENT_ID \
        --region $REGION &> /dev/null; then
        echo -e "${GREEN}✅ Client exists: $COGNITO_CLIENT_ID${NC}"
        
        # Get client details
        CLIENT_NAME=$(aws cognito-idp describe-user-pool-client \
            --user-pool-id $COGNITO_USER_POOL_ID \
            --client-id $COGNITO_CLIENT_ID \
            --region $REGION \
            --query 'UserPoolClient.ClientName' \
            --output text)
        
        echo -e "   Name: $CLIENT_NAME"
        
        if [ "$CLIENT_NAME" != "$COGNITO_CLIENT_NAME" ]; then
            echo -e "${YELLOW}   ⚠️  Warning: Expected name '$COGNITO_CLIENT_NAME'${NC}"
        fi
    else
        echo -e "${RED}❌ Client NOT FOUND: $COGNITO_CLIENT_ID${NC}"
        return 1
    fi
}

# Function to check ECS task definition
check_ecs_task_definition() {
    echo -e "\n${YELLOW}Checking ECS Task Definition...${NC}"
    
    # Get latest task definition
    LATEST_TASK_DEF=$(aws ecs describe-task-definition \
        --task-definition clarity-backend \
        --region $REGION \
        --query 'taskDefinition.containerDefinitions[0].environment' 2>/dev/null || echo "[]")
    
    if [ "$LATEST_TASK_DEF" == "[]" ]; then
        echo -e "${RED}❌ Task definition not found${NC}"
        return 1
    fi
    
    # Check Cognito values in task definition
    TASK_USER_POOL_ID=$(echo "$LATEST_TASK_DEF" | jq -r '.[] | select(.name=="COGNITO_USER_POOL_ID") | .value')
    TASK_CLIENT_ID=$(echo "$LATEST_TASK_DEF" | jq -r '.[] | select(.name=="COGNITO_CLIENT_ID") | .value')
    
    if [ "$TASK_USER_POOL_ID" == "$COGNITO_USER_POOL_ID" ]; then
        echo -e "${GREEN}✅ Task Definition User Pool ID: $TASK_USER_POOL_ID${NC}"
    else
        echo -e "${RED}❌ Task Definition User Pool ID MISMATCH: $TASK_USER_POOL_ID${NC}"
        echo -e "${YELLOW}   Expected: $COGNITO_USER_POOL_ID${NC}"
    fi
    
    if [ "$TASK_CLIENT_ID" == "$COGNITO_CLIENT_ID" ]; then
        echo -e "${GREEN}✅ Task Definition Client ID: $TASK_CLIENT_ID${NC}"
    else
        echo -e "${RED}❌ Task Definition Client ID MISMATCH: $TASK_CLIENT_ID${NC}"
        echo -e "${YELLOW}   Expected: $COGNITO_CLIENT_ID${NC}"
    fi
}

# Function to check running tasks
check_running_tasks() {
    echo -e "\n${YELLOW}Checking Running Tasks...${NC}"
    
    # Get running tasks
    TASK_ARNS=$(aws ecs list-tasks \
        --cluster clarity-backend-cluster \
        --service-name clarity-backend-service \
        --desired-status RUNNING \
        --region $REGION \
        --query 'taskArns[]' \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$TASK_ARNS" ]; then
        echo -e "${YELLOW}⚠️  No running tasks found${NC}"
        return 0
    fi
    
    # Check environment of running tasks
    for TASK_ARN in $TASK_ARNS; do
        echo -e "\n${BLUE}Checking task: ${TASK_ARN##*/}${NC}"
        
        # Get task definition ARN
        TASK_DEF_ARN=$(aws ecs describe-tasks \
            --cluster clarity-backend-cluster \
            --tasks $TASK_ARN \
            --region $REGION \
            --query 'tasks[0].taskDefinitionArn' \
            --output text)
        
        echo -e "Task Definition: ${TASK_DEF_ARN##*/}"
        
        # Get environment variables from the task's definition
        TASK_ENV=$(aws ecs describe-task-definition \
            --task-definition $TASK_DEF_ARN \
            --region $REGION \
            --query 'taskDefinition.containerDefinitions[0].environment')
        
        RUNNING_USER_POOL_ID=$(echo "$TASK_ENV" | jq -r '.[] | select(.name=="COGNITO_USER_POOL_ID") | .value')
        RUNNING_CLIENT_ID=$(echo "$TASK_ENV" | jq -r '.[] | select(.name=="COGNITO_CLIENT_ID") | .value')
        
        if [ "$RUNNING_USER_POOL_ID" == "$COGNITO_USER_POOL_ID" ]; then
            echo -e "${GREEN}✅ Running Task User Pool ID: $RUNNING_USER_POOL_ID${NC}"
        else
            echo -e "${RED}❌ Running Task User Pool ID MISMATCH: $RUNNING_USER_POOL_ID${NC}"
            echo -e "${YELLOW}   Expected: $COGNITO_USER_POOL_ID${NC}"
            echo -e "${RED}   ⚠️  THIS IS WHY REGISTRATION IS FAILING!${NC}"
        fi
        
        if [ "$RUNNING_CLIENT_ID" == "$COGNITO_CLIENT_ID" ]; then
            echo -e "${GREEN}✅ Running Task Client ID: $RUNNING_CLIENT_ID${NC}"
        else
            echo -e "${RED}❌ Running Task Client ID MISMATCH: $RUNNING_CLIENT_ID${NC}"
            echo -e "${YELLOW}   Expected: $COGNITO_CLIENT_ID${NC}"
            echo -e "${RED}   ⚠️  THIS IS WHY REGISTRATION IS FAILING!${NC}"
        fi
    done
}

# Function to generate fix commands
generate_fix_commands() {
    echo -e "\n${YELLOW}🔧 To fix the configuration:${NC}"
    echo -e "${BLUE}1. Deploy with the corrected task definition:${NC}"
    echo -e "   cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/clarity-loop-backend"
    echo -e "   ./ops/deploy.sh"
    echo -e "\n${BLUE}2. Or force a new deployment:${NC}"
    echo -e "   aws ecs update-service --cluster clarity-backend-cluster --service clarity-backend-service --force-new-deployment --region us-east-1"
}

# Main validation
main() {
    echo -e "${BLUE}Starting validation at $(date)${NC}"
    
    # Check Cognito
    check_cognito || true
    
    # Check ECS task definition
    check_ecs_task_definition || true
    
    # Check running tasks
    check_running_tasks || true
    
    # Generate fix commands
    generate_fix_commands
    
    echo -e "\n${BLUE}Validation completed at $(date)${NC}"
}

# Run main
main