#!/bin/bash
# CLARITY Backend Deployment Script - Claude V1 Implementation
# Enhanced with improved error handling, logging, and observability
# This represents Claude's initial best practices implementation

set -euo pipefail

# Script metadata
readonly SCRIPT_VERSION="v1.0.0-claude"
readonly SCRIPT_START_TIME=$(date +%s)

# Get script and project directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Enhanced logging with timestamps and levels
readonly LOG_FILE="/tmp/clarity-deploy-$(date +%Y%m%d-%H%M%S).log"
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Color codes for enhanced output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly NC='\033[0m'

# Configuration with validation
readonly REGION="${AWS_REGION:-us-east-1}"
readonly CLUSTER="${ECS_CLUSTER:-clarity-backend-cluster}"
readonly SERVICE="${ECS_SERVICE:-clarity-backend-service}"
readonly TASK_FAMILY="${ECS_TASK_FAMILY:-clarity-backend}"
readonly ECR_REPO="${ECR_REPOSITORY:-124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend}"

# Enhanced retry configuration
readonly MAX_RETRIES=3
readonly RETRY_DELAY=5
readonly BUILD_TIMEOUT=1800  # 30 minutes
readonly DEPLOY_TIMEOUT=600  # 10 minutes

# Metrics collection
declare -A METRICS=(
    ["start_time"]="$SCRIPT_START_TIME"
    ["build_duration"]="0"
    ["deploy_duration"]="0"
    ["total_duration"]="0"
    ["retry_count"]="0"
    ["errors_encountered"]="0"
)

# Enhanced error handling with context
error_handler() {
    local exit_code=$1
    local line_number=$2
    local command="$3"
    
    METRICS["errors_encountered"]=$((${METRICS["errors_encountered"]} + 1))
    
    log_error "Script failed with exit code $exit_code at line $line_number"
    log_error "Failed command: $command"
    log_error "Check log file: $LOG_FILE"
    
    # Attempt to get deployment status for debugging
    if command -v aws &> /dev/null; then
        log_info "Getting current deployment status for debugging..."
        aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
            --query 'services[0].{Status:status,RunningCount:runningCount,PendingCount:pendingCount}' \
            --output table 2>/dev/null || true
    fi
    
    # Generate error summary
    generate_error_summary
    exit $exit_code
}

trap 'error_handler $? $LINENO "$BASH_COMMAND"' ERR

# Retry mechanism with exponential backoff
retry_with_backoff() {
    local max_attempts="$1"
    local delay="$2"
    local command="$3"
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Attempt $attempt/$max_attempts: $command"
        
        if eval "$command"; then
            return 0
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            log_warn "Command failed, retrying in ${delay}s..."
            sleep $delay
            delay=$((delay * 2))  # Exponential backoff
        fi
        
        attempt=$((attempt + 1))
        METRICS["retry_count"]=$((${METRICS["retry_count"]} + 1))
    done
    
    log_error "Command failed after $max_attempts attempts: $command"
    return 1
}

# Enhanced prerequisite checks with versions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local required_tools=("aws" "docker" "jq" "curl")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        else
            local version=$(get_tool_version "$tool")
            log_info "$tool: $version"
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

get_tool_version() {
    case "$1" in
        "aws") aws --version 2>&1 | head -n1 ;;
        "docker") docker --version ;;
        "jq") jq --version ;;
        "curl") curl --version | head -n1 ;;
        *) echo "unknown" ;;
    esac
}

# Enhanced validation with comprehensive checks
validate_deployment_config() {
    log_info "Validating deployment configuration..."
    
    local task_def_path="ops/ecs-task-definition.json"
    local expected_user_pool_id="us-east-1_efXaR5EcP"
    local expected_client_id="7sm7ckrkovg78b03n1595euc71"
    
    # Validate task definition exists and is valid JSON
    if [[ ! -f "$task_def_path" ]]; then
        log_error "Task definition not found: $task_def_path"
        exit 1
    fi
    
    if ! jq empty "$task_def_path" 2>/dev/null; then
        log_error "Task definition is not valid JSON"
        exit 1
    fi
    
    # Validate Cognito configuration
    local user_pool_id=$(jq -r '.containerDefinitions[0].environment[] | select(.name=="COGNITO_USER_POOL_ID") | .value' "$task_def_path")
    local client_id=$(jq -r '.containerDefinitions[0].environment[] | select(.name=="COGNITO_CLIENT_ID") | .value' "$task_def_path")
    
    if [[ "$user_pool_id" != "$expected_user_pool_id" ]]; then
        log_error "Invalid User Pool ID: $user_pool_id (expected: $expected_user_pool_id)"
        exit 1
    fi
    
    if [[ "$client_id" != "$expected_client_id" ]]; then
        log_error "Invalid Client ID: $client_id (expected: $expected_client_id)"
        exit 1
    fi
    
    # Validate AWS resources exist
    if ! aws ecs describe-clusters --clusters "$CLUSTER" --region "$REGION" &> /dev/null; then
        log_error "ECS cluster not found: $CLUSTER"
        exit 1
    fi
    
    if ! aws ecr describe-repositories --repository-names "${ECR_REPO##*/}" --region "$REGION" &> /dev/null; then
        log_error "ECR repository not found: ${ECR_REPO##*/}"
        exit 1
    fi
    
    log_success "Configuration validation completed"
}

# Enhanced build process with metrics and caching
build_and_push_image() {
    log_info "Starting Docker image build process..."
    local build_start=$(date +%s)
    
    # Generate tag from git commit with fallback
    local tag
    if git rev-parse --git-dir > /dev/null 2>&1; then
        tag=$(git rev-parse --short HEAD)
        log_info "Using git commit tag: $tag"
    else
        tag="manual-$(date +%Y%m%d-%H%M%S)"
        log_warn "Not in git repository, using timestamp tag: $tag"
    fi
    
    local full_image="$ECR_REPO:$tag"
    
    # ECR login with retry
    log_info "Authenticating with ECR..."
    retry_with_backoff 3 5 "aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO"
    
    # Setup buildx with enhanced configuration
    log_info "Setting up Docker buildx..."
    docker buildx create --use --name clarity-builder --driver docker-container 2>/dev/null || true
    docker buildx inspect --bootstrap
    
    # Build with enhanced caching and monitoring
    log_info "Building Docker image for linux/amd64..."
    log_warn "CRITICAL: Building for linux/amd64 platform for AWS ECS compatibility"
    
    timeout "$BUILD_TIMEOUT" docker buildx build \
        --platform linux/amd64 \
        --cache-from type=registry,ref="$ECR_REPO:buildcache" \
        --cache-to type=registry,ref="$ECR_REPO:buildcache,mode=max" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        --push \
        -t "$full_image" \
        -t "$ECR_REPO:latest" \
        . || {
            log_error "Docker build timed out after ${BUILD_TIMEOUT}s"
            return 1
        }
    
    # Verify image was pushed
    log_info "Verifying image in ECR..."
    retry_with_backoff 3 5 "aws ecr describe-images --repository-name ${ECR_REPO##*/} --image-ids imageTag=$tag --region $REGION"
    
    # Get image details for metrics
    local image_size=$(aws ecr describe-images --repository-name "${ECR_REPO##*/}" --image-ids imageTag="$tag" --region "$REGION" \
        --query 'imageDetails[0].imageSizeInBytes' --output text)
    log_info "Image size: $((image_size / 1024 / 1024)) MB"
    
    local build_end=$(date +%s)
    METRICS["build_duration"]=$((build_end - build_start))
    log_success "Image built and pushed in ${METRICS["build_duration"]}s: $full_image"
    
    echo "$tag"
}

# Enhanced deployment with comprehensive monitoring
deploy_with_monitoring() {
    local tag="$1"
    local deploy_start=$(date +%s)
    
    log_info "Starting deployment process..."
    
    # Register task definition
    local task_def_arn
    task_def_arn=$(register_enhanced_task_definition "$tag")
    
    # Update service with enhanced monitoring
    log_info "Updating ECS service..."
    aws ecs update-service \
        --cluster "$CLUSTER" \
        --service "$SERVICE" \
        --task-definition "$task_def_arn" \
        --force-new-deployment \
        --region "$REGION" \
        --output table
    
    # Wait for stability with progress monitoring
    log_info "Waiting for deployment to stabilize..."
    monitor_deployment_progress &
    local monitor_pid=$!
    
    if timeout "$DEPLOY_TIMEOUT" aws ecs wait services-stable \
        --cluster "$CLUSTER" \
        --services "$SERVICE" \
        --region "$REGION"; then
        kill $monitor_pid 2>/dev/null || true
        log_success "Deployment completed successfully"
    else
        kill $monitor_pid 2>/dev/null || true
        log_error "Deployment timed out after ${DEPLOY_TIMEOUT}s"
        return 1
    fi
    
    local deploy_end=$(date +%s)
    METRICS["deploy_duration"]=$((deploy_end - deploy_start))
    log_success "Deployment completed in ${METRICS["deploy_duration"]}s"
}

monitor_deployment_progress() {
    while true; do
        sleep 30
        local service_info=$(aws ecs describe-services \
            --cluster "$CLUSTER" \
            --services "$SERVICE" \
            --region "$REGION" \
            --query 'services[0].{Running:runningCount,Pending:pendingCount,Desired:desiredCount}' \
            --output text 2>/dev/null || echo "0 0 0")
        
        log_info "Service status: $service_info (Running/Pending/Desired)"
    done
}

register_enhanced_task_definition() {
    local tag="$1"
    local image="$ECR_REPO:$tag"
    
    log_info "Registering task definition with image: $image"
    
    # Create enhanced task definition with the correct image
    local task_def_path="ops/ecs-task-definition.json"
    local task_def_json
    task_def_json=$(jq --arg image "$image" '.containerDefinitions[0].image = $image' "$task_def_path")
    
    # Add deployment metadata
    task_def_json=$(echo "$task_def_json" | jq --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '.containerDefinitions[0].environment += [{"name": "DEPLOYMENT_TIMESTAMP", "value": $timestamp}]')
    
    # Write to temporary file
    local temp_file="/tmp/task-definition-$(date +%s).json"
    echo "$task_def_json" > "$temp_file"
    
    # Register with retry
    local task_def_arn
    task_def_arn=$(retry_with_backoff 3 5 "aws ecs register-task-definition \
        --cli-input-json file://$temp_file \
        --region $REGION \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text")
    
    rm -f "$temp_file"
    
    # Verify the registration
    local registered_image
    registered_image=$(aws ecs describe-task-definition \
        --task-definition "$task_def_arn" \
        --region "$REGION" \
        --query 'taskDefinition.containerDefinitions[0].image' \
        --output text)
    
    if [[ "$registered_image" != *"$tag"* ]]; then
        log_error "Task definition registration failed - wrong image tag"
        exit 1
    fi
    
    log_success "Task definition registered: $task_def_arn"
    echo "$task_def_arn"
}

# Enhanced verification with comprehensive health checks
verify_deployment() {
    log_info "Performing comprehensive deployment verification..."
    
    # Get running task details
    local task_arn
    task_arn=$(aws ecs list-tasks \
        --cluster "$CLUSTER" \
        --service-name "$SERVICE" \
        --desired-status RUNNING \
        --region "$REGION" \
        --query 'taskArns[0]' \
        --output text)
    
    if [[ "$task_arn" == "None" || -z "$task_arn" ]]; then
        log_error "No running tasks found"
        return 1
    fi
    
    # Get task details
    local task_info
    task_info=$(aws ecs describe-tasks \
        --cluster "$CLUSTER" \
        --tasks "$task_arn" \
        --region "$REGION" \
        --query 'tasks[0].{TaskDef:taskDefinitionArn,Status:lastStatus,Health:healthStatus}' \
        --output table)
    
    log_info "Running task details:"$'\n'"$task_info"
    
    # Comprehensive health checks
    perform_health_checks
    
    # Run smoke tests if available
    run_smoke_tests
    
    log_success "Deployment verification completed"
}

perform_health_checks() {
    log_info "Performing health checks..."
    
    local health_url="https://clarity.novamindnyc.com/health"
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        local response
        if response=$(curl -s --max-time 10 "$health_url" 2>/dev/null); then
            if echo "$response" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
                log_success "Health check passed"
                echo "$response" | jq '.'
                return 0
            fi
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            sleep 15
        fi
        attempt=$((attempt + 1))
    done
    
    log_error "Health checks failed after $max_attempts attempts"
    return 1
}

run_smoke_tests() {
    log_info "Running smoke tests..."
    
    if [[ -f "./scripts/smoke-test-main.sh" ]]; then
        if timeout 300 ./scripts/smoke-test-main.sh https://clarity.novamindnyc.com; then
            log_success "Smoke tests passed"
        else
            log_error "Smoke tests failed"
            return 1
        fi
    else
        log_warn "No smoke test script found, skipping"
    fi
}

# Enhanced metrics and reporting
generate_deployment_report() {
    local end_time=$(date +%s)
    METRICS["total_duration"]=$((end_time - SCRIPT_START_TIME))
    
    log_info "Generating deployment report..."
    
    cat > "$LOG_FILE.report" << EOF
CLARITY Backend Deployment Report - $SCRIPT_VERSION
====================================================
Deployment Date: $(date)
Duration: ${METRICS["total_duration"]}s
Build Duration: ${METRICS["build_duration"]}s
Deploy Duration: ${METRICS["deploy_duration"]}s
Retry Count: ${METRICS["retry_count"]}
Errors Encountered: ${METRICS["errors_encountered"]}

Configuration:
- Region: $REGION
- Cluster: $CLUSTER
- Service: $SERVICE
- Repository: $ECR_REPO
- Tag: ${TAG:-"N/A"}

Status: SUCCESS
====================================================
EOF
    
    log_info "Deployment report saved to: $LOG_FILE.report"
}

generate_error_summary() {
    local end_time=$(date +%s)
    METRICS["total_duration"]=$((end_time - SCRIPT_START_TIME))
    
    cat > "$LOG_FILE.error" << EOF
CLARITY Backend Deployment Error Summary
========================================
Deployment Date: $(date)
Duration Before Failure: ${METRICS["total_duration"]}s
Retry Count: ${METRICS["retry_count"]}
Errors Encountered: ${METRICS["errors_encountered"]}

Status: FAILED
========================================
EOF
    
    log_error "Error summary saved to: $LOG_FILE.error"
}

# Enhanced main function with comprehensive flow
main() {
    log_info "Starting CLARITY Backend deployment $SCRIPT_VERSION at $(date)"
    
    # Parse command line arguments
    local build_mode=false
    local tag=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --build)
                build_mode=true
                shift
                ;;
            --tag)
                if [[ -z "${2:-}" ]]; then
                    log_error "--tag requires a value"
                    exit 1
                fi
                tag="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Pre-deployment validation
    check_prerequisites
    validate_deployment_config
    
    # Build and deployment flow
    if [[ "$build_mode" == true ]]; then
        TAG=$(build_and_push_image)
        log_success "Build completed with tag: $TAG"
    elif [[ -n "$tag" ]]; then
        TAG="$tag"
        log_info "Using provided tag: $TAG"
        
        # Verify tag exists in ECR
        if ! aws ecr describe-images --repository-name "${ECR_REPO##*/}" --image-ids imageTag="$TAG" --region "$REGION" &> /dev/null; then
            log_error "Image with tag '$TAG' not found in ECR"
            exit 1
        fi
        log_success "Image tag verified in ECR"
    else
        TAG="latest"
        log_info "Using existing image with tag: $TAG"
    fi
    
    # Deploy with monitoring
    deploy_with_monitoring "$TAG"
    
    # Comprehensive verification
    verify_deployment
    
    # Generate reports
    generate_deployment_report
    
    log_success "🎉 Deployment completed successfully!"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --build         Build new image before deployment
    --tag TAG       Use specific image tag for deployment
    --help          Show this help message

Examples:
    $0 --build              # Build new image and deploy
    $0 --tag v1.2.3         # Deploy specific tag
    $0                      # Deploy latest image

EOF
}

# Execute main function
main "$@"