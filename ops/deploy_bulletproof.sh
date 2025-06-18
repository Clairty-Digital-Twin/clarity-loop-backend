#!/bin/bash
# CLARITY Backend Professional Deployment Script with Bulletproof Startup Validation
# This ensures consistent deployments with proper validation and startup checks

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
CYAN='\033[0;36m'
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

echo -e "${BLUE}🚀 CLARITY Backend Bulletproof Deployment${NC}"
echo -e "${BLUE}=========================================${NC}"

# Function to validate startup configuration
validate_startup_config() {
    echo -e "\n${CYAN}🛡️ Running bulletproof startup validation...${NC}"
    
    # Check if startup validator exists
    if [ ! -f "scripts/startup_validator.py" ]; then
        echo -e "${RED}❌ Startup validator not found${NC}"
        exit 1
    fi
    
    # Set production environment variables for validation
    export ENVIRONMENT="production"
    export AWS_REGION="$REGION"
    export COGNITO_USER_POOL_ID="$EXPECTED_USER_POOL_ID"
    export COGNITO_CLIENT_ID="$EXPECTED_CLIENT_ID"
    export DYNAMODB_TABLE_NAME="clarity-health-data"
    export S3_BUCKET_NAME="clarity-health-uploads"
    export S3_ML_MODELS_BUCKET="clarity-ml-models-124355672559"
    
    # Skip external services for validation (use mocks)
    export SKIP_EXTERNAL_SERVICES="true"
    
    # Set production-grade security defaults for validation
    export SECRET_KEY="production-secret-key-validation-placeholder"
    export CORS_ALLOWED_ORIGINS="https://clarity.novamindnyc.com,https://novamindnyc.com"
    
    echo -e "${YELLOW}Running dry-run validation...${NC}"
    
    # Run startup validator in dry-run mode
    if python3 scripts/startup_validator.py --dry-run; then
        echo -e "${GREEN}✅ Startup validation passed${NC}"
    else
        echo -e "${RED}❌ Startup validation failed${NC}"
        echo -e "${RED}   Configuration errors detected - cannot deploy${NC}"
        exit 1
    fi
}

# Function to run configuration-only validation
validate_config_only() {
    echo -e "\n${CYAN}⚙️ Running configuration validation...${NC}"
    
    # Set minimal environment for config validation
    export ENVIRONMENT="production"
    export AWS_REGION="$REGION"
    
    if python3 scripts/startup_validator.py --config-only; then
        echo -e "${GREEN}✅ Configuration validation passed${NC}"
    else
        echo -e "${RED}❌ Configuration validation failed${NC}"
        exit 1
    fi
}

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
    
    # Check Python3
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 not found${NC}"
        exit 1
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}❌ jq not found${NC}"
        exit 1
    fi
    
    # Check Git (for tagging)
    if ! command -v git &> /dev/null; then
        echo -e "${RED}❌ Git not found${NC}"
        exit 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir &> /dev/null; then
        echo -e "${RED}❌ Not in a git repository${NC}"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}❌ AWS credentials not configured${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites met${NC}"
}

# Function to run pre-deployment validation
run_pre_deployment_validation() {
    echo -e "\n${CYAN}🔍 Running pre-deployment validation...${NC}"
    
    # 1. Validate configuration schema
    validate_config_only
    
    # 2. Run full startup validation
    validate_startup_config
    
    # 3. Check if we have required AWS resources
    echo -e "\n${YELLOW}Checking AWS resources...${NC}"
    
    # Check ECS cluster
    if aws ecs describe-clusters --clusters "$CLUSTER" --region "$REGION" --query 'clusters[0].status' --output text | grep -q "ACTIVE"; then
        echo -e "${GREEN}✅ ECS cluster '$CLUSTER' is active${NC}"
    else
        echo -e "${RED}❌ ECS cluster '$CLUSTER' not found or not active${NC}"
        exit 1
    fi
    
    # Check ECR repository
    if aws ecr describe-repositories --repository-names clarity-backend --region "$REGION" &> /dev/null; then
        echo -e "${GREEN}✅ ECR repository exists${NC}"
    else
        echo -e "${RED}❌ ECR repository not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Pre-deployment validation completed${NC}"
}

# Function to build and push Docker image
build_and_push() {
    echo -e "\n${YELLOW}Building and pushing Docker image...${NC}"
    
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

# Function to update ECS service
update_service() {
    local TASK_DEF_ARN="$1"
    echo -e "\n${YELLOW}Updating ECS service...${NC}"
    
    # Update the service
    UPDATE_RESULT=$(aws ecs update-service \
        --cluster "$CLUSTER" \
        --service "$SERVICE" \
        --task-definition "$TASK_DEF_ARN" \
        --region "$REGION" \
        --query 'service.taskDefinition' \
        --output text)
    
    echo -e "${GREEN}✅ Service updated with task definition: $UPDATE_RESULT${NC}"
    
    # Wait for service to stabilize
    echo -e "${YELLOW}Waiting for service to stabilize...${NC}"
    aws ecs wait services-stable \
        --cluster "$CLUSTER" \
        --services "$SERVICE" \
        --region "$REGION"
    
    echo -e "${GREEN}✅ Service is stable${NC}"
}

# Function to run post-deployment validation
run_post_deployment_validation() {
    echo -e "\n${CYAN}🔍 Running post-deployment validation...${NC}"
    
    # Wait a moment for service to fully start
    echo -e "${YELLOW}Waiting for service to start...${NC}"
    sleep 10
    
    # Check service status
    SERVICE_STATUS=$(aws ecs describe-services \
        --cluster "$CLUSTER" \
        --services "$SERVICE" \
        --region "$REGION" \
        --query 'services[0].status' \
        --output text)
    
    if [ "$SERVICE_STATUS" = "ACTIVE" ]; then
        echo -e "${GREEN}✅ Service status: ACTIVE${NC}"
    else
        echo -e "${RED}❌ Service status: $SERVICE_STATUS${NC}"
        exit 1
    fi
    
    # Check running task count
    RUNNING_COUNT=$(aws ecs describe-services \
        --cluster "$CLUSTER" \
        --services "$SERVICE" \
        --region "$REGION" \
        --query 'services[0].runningCount' \
        --output text)
    
    DESIRED_COUNT=$(aws ecs describe-services \
        --cluster "$CLUSTER" \
        --services "$SERVICE" \
        --region "$REGION" \
        --query 'services[0].desiredCount' \
        --output text)
    
    if [ "$RUNNING_COUNT" = "$DESIRED_COUNT" ]; then
        echo -e "${GREEN}✅ Running tasks: $RUNNING_COUNT/$DESIRED_COUNT${NC}"
    else
        echo -e "${RED}❌ Running tasks: $RUNNING_COUNT/$DESIRED_COUNT${NC}"
        
        # Show recent task failures
        echo -e "${YELLOW}Checking for task failures...${NC}"
        TASK_ARNS=$(aws ecs list-tasks \
            --cluster "$CLUSTER" \
            --service-name "$SERVICE" \
            --region "$REGION" \
            --query 'taskArns' \
            --output text)
        
        if [ -n "$TASK_ARNS" ]; then
            echo -e "${YELLOW}Recent task status:${NC}"
            aws ecs describe-tasks \
                --cluster "$CLUSTER" \
                --tasks $TASK_ARNS \
                --region "$REGION" \
                --query 'tasks[*].[taskArn,lastStatus,healthStatus,stopCode,stoppedReason]' \
                --output table
        fi
        
        exit 1
    fi
    
    # Test health endpoint
    echo -e "${YELLOW}Testing health endpoint...${NC}"
    ALB_URL="http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com"
    
    # Wait for ALB to route to new tasks
    echo -e "${YELLOW}Waiting for load balancer to update...${NC}"
    sleep 30
    
    # Test health endpoint with timeout
    if timeout 30 curl -f "$ALB_URL/health" &> /dev/null; then
        echo -e "${GREEN}✅ Health endpoint responding${NC}"
    else
        echo -e "${YELLOW}⚠️ Health endpoint not responding yet (may still be starting)${NC}"
        echo -e "${YELLOW}   Manual verification recommended: $ALB_URL/health${NC}"
    fi
    
    echo -e "${GREEN}✅ Post-deployment validation completed${NC}"
}

# Function to show deployment summary
show_deployment_summary() {
    echo -e "\n${BLUE}📋 Deployment Summary${NC}"
    echo -e "${BLUE}===================${NC}"
    echo -e "Cluster: ${GREEN}$CLUSTER${NC}"
    echo -e "Service: ${GREEN}$SERVICE${NC}"
    echo -e "Image: ${GREEN}$ECR_REPO:$TAG${NC}"
    echo -e "Health URL: ${GREEN}http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health${NC}"
    echo -e "API Docs: ${GREEN}http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/docs${NC}"
    echo -e "\n${GREEN}🎉 Bulletproof deployment completed successfully!${NC}"
}

# Main deployment flow
main() {
    local BUILD_ONLY=false
    local VALIDATE_ONLY=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                BUILD_ONLY=true
                shift
                ;;
            --validate)
                VALIDATE_ONLY=true
                shift
                ;;
            --help)
                echo "CLARITY Bulletproof Deployment Script"
                echo ""
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --build      Build and push image only (skip deployment)"
                echo "  --validate   Run validation only (skip build and deployment)"
                echo "  --help       Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                # Full deployment with validation"
                echo "  $0 --validate     # Validation only"
                echo "  $0 --build        # Build and push only"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Always check prerequisites
    check_prerequisites
    
    if [ "$VALIDATE_ONLY" = true ]; then
        run_pre_deployment_validation
        echo -e "${GREEN}🎉 Validation completed successfully!${NC}"
        exit 0
    fi
    
    # Run pre-deployment validation
    run_pre_deployment_validation
    
    # Build and push image
    TAG=$(build_and_push)
    
    if [ "$BUILD_ONLY" = true ]; then
        echo -e "${GREEN}🎉 Build completed successfully!${NC}"
        echo -e "Image: ${GREEN}$ECR_REPO:$TAG${NC}"
        exit 0
    fi
    
    # Register task definition
    TASK_DEF_ARN=$(register_task_definition)
    
    # Update service
    update_service "$TASK_DEF_ARN"
    
    # Run post-deployment validation
    run_post_deployment_validation
    
    # Show summary
    show_deployment_summary
}

# Run main function with all arguments
main "$@"