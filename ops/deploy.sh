#!/bin/bash
# CLARITY Backend Professional Deployment Script
# This ensures consistent deployments with proper validation

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of ops/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root to ensure all paths work correctly
cd "$PROJECT_ROOT"

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
    # Task definition is always in ops/ directory relative to project root
    TASK_DEF_PATH="ops/ecs-task-definition.json"
    
    USER_POOL_ID=$(jq -r '.containerDefinitions[0].environment[] | select(.name=="COGNITO_USER_POOL_ID") | .value' "$TASK_DEF_PATH")
    CLIENT_ID=$(jq -r '.containerDefinitions[0].environment[] | select(.name=="COGNITO_CLIENT_ID") | .value' "$TASK_DEF_PATH")
    
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
    
    # Generate tag from git commit
    TAG=$(git rev-parse --short HEAD)
    FULL_IMAGE="$ECR_REPO:$TAG"
    
    echo -e "${BLUE}Using git commit tag: $TAG${NC}"
    
    # Login to ECR first
    echo -e "${BLUE}Logging in to ECR...${NC}"
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO
    
    # Build for linux/amd64 (CRITICAL FOR ECS!)
    echo -e "${BLUE}Building for linux/amd64...${NC}"
    echo -e "${RED}⚠️  CRITICAL: Always build for linux/amd64 platform for AWS ECS${NC}"
    
    # Setup buildx if not already done
    docker buildx create --use --name clarity-builder --driver docker-container 2>/dev/null || true
    
    # Use buildx with caching for faster builds
    echo -e "${YELLOW}Building with docker buildx for linux/amd64 with cache...${NC}"
    
    # Build with cache and push
    docker buildx build \
        --platform linux/amd64 \
        --cache-from type=registry,ref=$ECR_REPO:buildcache \
        --cache-to type=registry,ref=$ECR_REPO:buildcache,mode=max \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        --push \
        -t $FULL_IMAGE \
        -t $ECR_REPO:latest \
        .
    
    # Verify the image was pushed
    echo -e "${BLUE}Verifying image in ECR...${NC}"
    aws ecr describe-images --repository-name clarity-backend --image-ids imageTag=$TAG --region $REGION >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Image not found in ECR. Build may have failed.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Image pushed: $FULL_IMAGE${NC}"
    
    # Return just the tag for later use
    echo "$TAG"
}

# Function to register task definition
register_task_definition() {
    echo -e "\n${YELLOW}Registering task definition...${NC}" >&2
    
    # Get the latest tag or use provided tag
    if [ -z "${TAG:-}" ]; then
        TAG="latest"
        echo -e "${YELLOW}No TAG specified, using 'latest'${NC}" >&2
    fi
    
    # Replace IMAGE_PLACEHOLDER with actual image
    IMAGE="$ECR_REPO:$TAG"
    echo -e "${BLUE}Using image: $IMAGE${NC}" >&2
    
    # Create task definition JSON with the correct image
    # Task definition is always in ops/ directory relative to project root
    TASK_DEF_PATH="ops/ecs-task-definition.json"
    
    # Use jq instead of sed for reliable JSON manipulation
    TASK_DEF_JSON=$(cat "$TASK_DEF_PATH" | jq --arg image "$IMAGE" '.containerDefinitions[0].image = $image')
    
    # Register task definition with the updated JSON
    # Write to temp file to avoid stdin issues
    TEMP_TASK_DEF="/tmp/task-definition-$(date +%s).json"
    echo "$TASK_DEF_JSON" > "$TEMP_TASK_DEF"
    
    TASK_DEF_ARN=$(aws ecs register-task-definition \
        --cli-input-json "file://$TEMP_TASK_DEF" \
        --region $REGION \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)
    
    # Clean up temp file
    rm -f "$TEMP_TASK_DEF"
    
    # Verify the image in the registered task definition
    REGISTERED_IMAGE=$(aws ecs describe-task-definition \
        --task-definition "$TASK_DEF_ARN" \
        --region $REGION \
        --query 'taskDefinition.containerDefinitions[0].image' \
        --output text)
    
    echo -e "${GREEN}✅ Task definition registered: $TASK_DEF_ARN${NC}" >&2
    echo -e "${GREEN}✅ Using image: $REGISTERED_IMAGE${NC}" >&2
    
    # Sanity check
    if [[ "$REGISTERED_IMAGE" != *"$TAG"* ]]; then
        echo -e "${RED}❌ ERROR: Task definition is not using expected tag '$TAG'${NC}" >&2
        echo -e "${RED}❌ Registered image: $REGISTERED_IMAGE${NC}" >&2
        exit 1
    fi
    
    # Return only the ARN
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
    
    # Parse command line arguments first
    BUILD_MODE=false
    TAG=""
    
    # Process arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --build)
                BUILD_MODE=true
                shift
                ;;
            --tag)
                if [ -z "${2:-}" ]; then
                    echo -e "${RED}--tag requires a value${NC}"
                    echo "Usage: $0 [--build] [--tag TAG]"
                    exit 1
                fi
                TAG="$2"
                shift 2
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Usage: $0 [--build] [--tag TAG]"
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    
    # Validate task definition
    validate_task_definition
    
    # Build and push if requested
    if [ "$BUILD_MODE" = true ]; then
        TAG=$(build_and_push)
        echo -e "${GREEN}✅ Built and pushed tag: $TAG${NC}"
    elif [ -n "$TAG" ]; then
        echo -e "${YELLOW}Using provided tag: $TAG${NC}"
        # Verify the tag exists in ECR
        echo -e "${BLUE}Verifying image tag in ECR...${NC}"
        IMAGE_EXISTS=$(aws ecr list-images --repository-name clarity-backend --region $REGION --filter tagStatus=TAGGED --query "imageIds[?imageTag=='$TAG']" --output text)
        if [ -z "$IMAGE_EXISTS" ]; then
            echo -e "${RED}❌ Image with tag '$TAG' not found in ECR${NC}"
            exit 1
        else
            echo -e "${GREEN}✅ Image tag verified in ECR${NC}"
        fi
    else
        # Default to latest
        TAG="latest"
        echo -e "${YELLOW}Using existing image with tag: $TAG${NC}"
    fi
    
    # Export TAG so register_task_definition can use it
    export TAG
    
    # Register task definition
    TASK_DEF_ARN=$(register_task_definition)
    
    # Update service
    update_service "$TASK_DEF_ARN"
    
    # Wait for deployment
    wait_for_deployment
    
    # Verify deployment
    verify_deployment
    
    # Run smoke tests
    echo -e "\n${YELLOW}Running smoke tests...${NC}"
    if [ -f "./scripts/smoke-test-main.sh" ]; then
        if ./scripts/smoke-test-main.sh https://clarity.novamindnyc.com; then
            echo -e "${GREEN}✅ Smoke tests passed${NC}"
        else
            echo -e "${RED}❌ Smoke tests failed${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠️  No smoke test script found${NC}"
    fi
    
    echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${BLUE}Deployment finished at $(date)${NC}"
}

# Run main function
main "$@"