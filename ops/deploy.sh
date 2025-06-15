#!/bin/bash
# CLARITY Backend Professional Deployment Script
# This ensures consistent deployments with proper validation

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGION="us-east-1"
CLUSTER="clarity-backend-cluster"
SERVICE="clarity-backend-service"
TASK_FAMILY="clarity-backend"
ECR_REPO="124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend"

# Expected Cognito configuration
EXPECTED_USER_POOL_ID="us-east-1_efXaR5EcP"
EXPECTED_CLIENT_ID="7sm7ckrkovg78b03n1595euc71"

echo -e "${BLUE}🚀 CLARITY Backend Deployment Script${NC}"
echo -e "${BLUE}=====================================\${NC}"

# Function to check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}Checking prerequisites...${NC}"
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}❌ AWS CLI not found${NC}"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found${NC}"
        exit 1
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}❌ jq not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites met${NC}"
}

# Function to validate task definition
validate_task_definition() {
    echo -e "\n${YELLOW}Validating task definition...${NC}"
    
    # Check Cognito configuration
    USER_POOL_ID=$(jq -r '.containerDefinitions[0].environment[] | select(.name=="COGNITO_USER_POOL_ID") | .value' ops/ecs-task-definition.json)
    CLIENT_ID=$(jq -r '.containerDefinitions[0].environment[] | select(.name=="COGNITO_CLIENT_ID") | .value' ops/ecs-task-definition.json)
    
    if [ "$USER_POOL_ID" != "$EXPECTED_USER_POOL_ID" ]; then
        echo -e "${RED}❌ Invalid User Pool ID: $USER_POOL_ID${NC}"
        echo -e "${YELLOW}   Expected: $EXPECTED_USER_POOL_ID${NC}"
        exit 1
    fi
    
    if [ "$CLIENT_ID" != "$EXPECTED_CLIENT_ID" ]; then
        echo -e "${RED}❌ Invalid Client ID: $CLIENT_ID${NC}"
        echo -e "${YELLOW}   Expected: $EXPECTED_CLIENT_ID${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Task definition validated${NC}"
}

# Function to build and push Docker image
build_and_push() {
    echo -e "\n${YELLOW}Building Docker image...${NC}"
    
    # Generate tag with timestamp
    TAG="v$(date +%Y%m%d-%H%M%S)"
    FULL_IMAGE="$ECR_REPO:$TAG"
    
    # Build for linux/amd64 (CRITICAL FOR ECS!)
    echo -e "${BLUE}Building for linux/amd64...${NC}"
    echo -e "${RED}⚠️  CRITICAL: Always build for linux/amd64 platform for AWS ECS${NC}"
    
    # Use buildx for cross-platform build
    if ! docker buildx version >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker buildx not found. Cannot build for linux/amd64${NC}"
        exit 1
    fi
    
    # Ensure we have a builder that can handle linux/amd64
    if ! docker buildx ls | grep -q "linux/amd64"; then
        echo -e "${YELLOW}Creating buildx builder for linux/amd64...${NC}"
        docker buildx create --use --name clarity-builder --platform linux/amd64
    fi
    
    docker buildx build --platform linux/amd64 --load -t clarity-backend:$TAG .
    
    # Tag for ECR
    docker tag clarity-backend:$TAG $FULL_IMAGE
    
    # Login to ECR
    echo -e "${BLUE}Logging in to ECR...${NC}"
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO
    
    # Push to ECR
    echo -e "${BLUE}Pushing to ECR...${NC}"
    docker push $FULL_IMAGE
    
    echo -e "${GREEN}✅ Image pushed: $FULL_IMAGE${NC}"
    
    # Update task definition with new image
    echo -e "${BLUE}Updating task definition...${NC}"
    jq ".containerDefinitions[0].image = \"$FULL_IMAGE\"" ops/ecs-task-definition.json > ops/ecs-task-definition-temp.json
    mv ops/ecs-task-definition-temp.json ops/ecs-task-definition.json
    
    echo "$TAG"
}

# Function to register task definition
register_task_definition() {
    echo -e "\n${YELLOW}Registering task definition...${NC}"
    
    TASK_DEF_ARN=$(aws ecs register-task-definition \
        --cli-input-json file://ops/ecs-task-definition.json \
        --region $REGION \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)
    
    echo -e "${GREEN}✅ Task definition registered: $TASK_DEF_ARN${NC}"
    echo "$TASK_DEF_ARN"
}

# Function to update service
update_service() {
    local TASK_DEF_ARN=$1
    
    echo -e "\n${YELLOW}Updating ECS service...${NC}"
    
    aws ecs update-service \
        --cluster $CLUSTER \
        --service $SERVICE \
        --task-definition $TASK_DEF_ARN \
        --force-new-deployment \
        --region $REGION \
        --output table
    
    echo -e "${GREEN}✅ Service update initiated${NC}"
}

# Function to wait for deployment
wait_for_deployment() {
    echo -e "\n${YELLOW}Waiting for deployment to complete...${NC}"
    
    # Wait for service to stabilize
    aws ecs wait services-stable \
        --cluster $CLUSTER \
        --services $SERVICE \
        --region $REGION
    
    echo -e "${GREEN}✅ Deployment completed successfully${NC}"
}

# Function to verify deployment
verify_deployment() {
    echo -e "\n${YELLOW}Verifying deployment...${NC}"
    
    # Get running task
    TASK_ARN=$(aws ecs list-tasks \
        --cluster $CLUSTER \
        --service-name $SERVICE \
        --desired-status RUNNING \
        --region $REGION \
        --query 'taskArns[0]' \
        --output text)
    
    if [ "$TASK_ARN" != "None" ] && [ -n "$TASK_ARN" ]; then
        # Get task details
        TASK_DEF=$(aws ecs describe-tasks \
            --cluster $CLUSTER \
            --tasks $TASK_ARN \
            --region $REGION \
            --query 'tasks[0].taskDefinitionArn' \
            --output text)
        
        echo -e "${GREEN}✅ Running task: $TASK_ARN${NC}"
        echo -e "${GREEN}✅ Task definition: $TASK_DEF${NC}"
        
        # Test health endpoint
        echo -e "\n${YELLOW}Testing health endpoint...${NC}"
        HEALTH_RESPONSE=$(curl -s https://clarity.novamindnyc.com/health || echo "Failed")
        
        if echo "$HEALTH_RESPONSE" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Health check passed${NC}"
            echo "$HEALTH_RESPONSE" | jq '.'
        else
            echo -e "${RED}❌ Health check failed${NC}"
            echo "$HEALTH_RESPONSE"
        fi
    else
        echo -e "${RED}❌ No running tasks found${NC}"
        exit 1
    fi
}

# Main deployment flow
main() {
    echo -e "${BLUE}Starting deployment at $(date)${NC}"
    
    # Check prerequisites
    check_prerequisites
    
    # Validate task definition
    validate_task_definition
    
    # Build and push if requested
    if [ "${1:-}" == "--build" ]; then
        TAG=$(build_and_push)
        echo -e "${GREEN}✅ Built and pushed tag: $TAG${NC}"
    fi
    
    # Register task definition
    TASK_DEF_ARN=$(register_task_definition)
    
    # Update service
    update_service "$TASK_DEF_ARN"
    
    # Wait for deployment
    wait_for_deployment
    
    # Verify deployment
    verify_deployment
    
    echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${BLUE}Deployment finished at $(date)${NC}"
}

# Run main function
main "$@"