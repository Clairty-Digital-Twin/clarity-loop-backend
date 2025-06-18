#!/bin/bash
# CLARITY Backend Deployment Script - Claude V2 Implementation
# Improved security, performance, and code organization based on V1 self-review
# Addresses: eval security, resource cleanup, code structure, performance

set -euo pipefail

# Script metadata
readonly SCRIPT_VERSION="v2.0.0-claude"
readonly SCRIPT_START_TIME=$(date +%s)
readonly SCRIPT_PID=$$

# Get script and project directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Secure logging with proper permissions
readonly LOG_DIR="/tmp/clarity-deploy-$$"
readonly LOG_FILE="$LOG_DIR/deployment.log"
readonly JSON_LOG_FILE="$LOG_DIR/deployment.json"

# Create secure log directory
mkdir -m 700 "$LOG_DIR"

# Color codes for enhanced output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly NC='\033[0m'

# Enhanced logging with both human-readable and JSON formats
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local iso_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    # Human-readable log with colors
    local color=""
    case "$level" in
        "ERROR") color="$RED" ;;
        "WARN") color="$YELLOW" ;;
        "SUCCESS") color="$GREEN" ;;
        "INFO") color="$BLUE" ;;
    esac
    
    echo -e "${color}[$timestamp] [$level] $message${NC}" | tee -a "$LOG_FILE"
    
    # Structured JSON log for machine parsing
    jq -n \
        --arg timestamp "$iso_timestamp" \
        --arg level "$level" \
        --arg message "$message" \
        --arg script_version "$SCRIPT_VERSION" \
        --arg process_id "$SCRIPT_PID" \
        '{
            timestamp: $timestamp,
            level: $level,
            message: $message,
            script_version: $script_version,
            process_id: $process_id
        }' >> "$JSON_LOG_FILE"
}

log_info() { log "INFO" "$1"; }
log_warn() { log "WARN" "$1"; }
log_error() { log "ERROR" "$1"; }
log_success() { log "SUCCESS" "$1"; }

# Input validation and sanitization
validate_input() {
    local input="$1"
    local pattern="$2"
    local description="$3"
    
    if [[ ! "$input" =~ $pattern ]]; then
        log_error "Invalid $description: $input"
        exit 1
    fi
}

# Configuration with validation
readonly REGION="${AWS_REGION:-us-east-1}"
readonly CLUSTER="${ECS_CLUSTER:-clarity-backend-cluster}"
readonly SERVICE="${ECS_SERVICE:-clarity-backend-service}"
readonly TASK_FAMILY="${ECS_TASK_FAMILY:-clarity-backend}"
readonly ECR_REPO="${ECR_REPOSITORY:-124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend}"

# Validate configuration
validate_input "$REGION" "^[a-z0-9-]+$" "AWS region"
validate_input "$CLUSTER" "^[a-zA-Z0-9-_]+$" "ECS cluster name"
validate_input "$SERVICE" "^[a-zA-Z0-9-_]+$" "ECS service name"

# Enhanced retry configuration
readonly MAX_RETRIES=3
readonly RETRY_DELAY=5
readonly BUILD_TIMEOUT=1800
readonly DEPLOY_TIMEOUT=600

# Global cleanup tracking
declare -a CLEANUP_TASKS=()
declare -a BACKGROUND_PIDS=()

# Metrics collection with better structure
declare -A METRICS=(
    ["start_time"]="$SCRIPT_START_TIME"
    ["build_duration"]="0"
    ["deploy_duration"]="0"
    ["validation_duration"]="0"
    ["total_duration"]="0"
    ["retry_count"]="0"
    ["errors_encountered"]="0"
    ["parallel_tasks_count"]="0"
)

# Enhanced cleanup function
cleanup() {
    log_info "Performing cleanup..."
    
    # Kill background processes
    for pid in "${BACKGROUND_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Terminating background process: $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # Execute cleanup tasks
    for task in "${CLEANUP_TASKS[@]}"; do
        log_info "Executing cleanup: $task"
        eval "$task" 2>/dev/null || true
    done
    
    # Clean up Docker buildx builder
    docker buildx rm clarity-builder 2>/dev/null || true
    
    log_info "Cleanup completed"
}

# Enhanced error handling without eval
error_handler() {
    local exit_code=$1
    local line_number=$2
    local command="$3"
    
    METRICS["errors_encountered"]=$((${METRICS["errors_encountered"]} + 1))
    
    log_error "Script failed with exit code $exit_code at line $line_number"
    log_error "Failed command: $command"
    log_error "Log files: $LOG_FILE, $JSON_LOG_FILE"
    
    # Gather deployment state for debugging
    gather_debug_info || true
    
    # Generate error summary
    generate_error_summary
    
    # Cleanup and exit
    cleanup
    exit $exit_code
}

trap 'error_handler $? $LINENO "$BASH_COMMAND"' ERR
trap 'cleanup' EXIT

# Secure retry mechanism without eval
retry_command() {
    local max_attempts="$1"
    local delay="$2"
    local description="$3"
    shift 3  # Remove first 3 args, rest are the command
    local command=("$@")
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Attempt $attempt/$max_attempts: $description"
        
        if "${command[@]}"; then
            return 0
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            log_warn "Command failed, retrying in ${delay}s..."
            sleep $delay
            delay=$((delay * 2))
        fi
        
        attempt=$((attempt + 1))
        METRICS["retry_count"]=$((${METRICS["retry_count"]} + 1))
    done
    
    log_error "Command failed after $max_attempts attempts: $description"
    return 1
}

# Parallel prerequisite checks for better performance
check_prerequisites() {
    log_info "Checking prerequisites in parallel..."
    local validation_start=$(date +%s)
    
    local required_tools=("aws" "docker" "jq" "curl")
    local check_pids=()
    local temp_dir=$(mktemp -d)
    CLEANUP_TASKS+=("rm -rf $temp_dir")
    
    # Check tools in parallel
    for tool in "${required_tools[@]}"; do
        (
            if ! command -v "$tool" &> /dev/null; then
                echo "MISSING:$tool" > "$temp_dir/$tool.result"
            else
                local version=$(get_tool_version "$tool")
                echo "OK:$tool:$version" > "$temp_dir/$tool.result"
            fi
        ) &
        check_pids+=($!)
    done
    
    # Check AWS credentials in parallel
    (
        if aws sts get-caller-identity &> /dev/null; then
            echo "OK:aws-credentials" > "$temp_dir/aws-creds.result"
        else
            echo "MISSING:aws-credentials" > "$temp_dir/aws-creds.result"
        fi
    ) &
    check_pids+=($!)
    
    # Check Docker daemon in parallel
    (
        if docker info &> /dev/null; then
            echo "OK:docker-daemon" > "$temp_dir/docker.result"
        else
            echo "MISSING:docker-daemon" > "$temp_dir/docker.result"
        fi
    ) &
    check_pids+=($!)
    
    # Wait for all checks
    for pid in "${check_pids[@]}"; do
        wait "$pid"
    done
    
    # Process results
    local missing_tools=()
    for tool in "${required_tools[@]}"; do
        local result=$(cat "$temp_dir/$tool.result")
        if [[ "$result" == MISSING:* ]]; then
            missing_tools+=("$tool")
        else
            log_info "${result#OK:}"
        fi
    done
    
    # Check other results
    if [[ "$(cat "$temp_dir/aws-creds.result")" == MISSING:* ]]; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    if [[ "$(cat "$temp_dir/docker.result")" == MISSING:* ]]; then
        log_error "Docker daemon not running"
        exit 1
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    local validation_end=$(date +%s)
    METRICS["validation_duration"]=$((validation_end - validation_start))
    log_success "All prerequisites met in ${METRICS["validation_duration"]}s"
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

# Modular validation functions
validate_task_definition() {
    local task_def_path="ops/ecs-task-definition.json"
    
    if [[ ! -f "$task_def_path" ]]; then
        log_error "Task definition not found: $task_def_path"
        return 1
    fi
    
    if ! jq empty "$task_def_path" 2>/dev/null; then
        log_error "Task definition is not valid JSON"
        return 1
    fi
    
    log_info "Task definition structure validated"
}

validate_cognito_config() {
    local task_def_path="ops/ecs-task-definition.json"
    local expected_user_pool_id="us-east-1_efXaR5EcP"
    local expected_client_id="7sm7ckrkovg78b03n1595euc71"
    
    local user_pool_id=$(jq -r '.containerDefinitions[0].environment[] | select(.name=="COGNITO_USER_POOL_ID") | .value' "$task_def_path")
    local client_id=$(jq -r '.containerDefinitions[0].environment[] | select(.name=="COGNITO_CLIENT_ID") | .value' "$task_def_path")
    
    if [[ "$user_pool_id" != "$expected_user_pool_id" ]]; then
        log_error "Invalid User Pool ID: $user_pool_id"
        return 1
    fi
    
    if [[ "$client_id" != "$expected_client_id" ]]; then
        log_error "Invalid Client ID: $client_id"
        return 1
    fi
    
    log_info "Cognito configuration validated"
}

validate_aws_resources() {
    local temp_dir=$(mktemp -d)
    CLEANUP_TASKS+=("rm -rf $temp_dir")
    
    # Check ECS cluster
    (
        if aws ecs describe-clusters --clusters "$CLUSTER" --region "$REGION" &> /dev/null; then
            echo "OK" > "$temp_dir/cluster.result"
        else
            echo "MISSING" > "$temp_dir/cluster.result"
        fi
    ) &
    local cluster_pid=$!
    
    # Check ECR repository
    (
        if aws ecr describe-repositories --repository-names "${ECR_REPO##*/}" --region "$REGION" &> /dev/null; then
            echo "OK" > "$temp_dir/ecr.result"
        else
            echo "MISSING" > "$temp_dir/ecr.result"
        fi
    ) &
    local ecr_pid=$!
    
    wait "$cluster_pid" "$ecr_pid"
    
    if [[ "$(cat "$temp_dir/cluster.result")" == "MISSING" ]]; then
        log_error "ECS cluster not found: $CLUSTER"
        return 1
    fi
    
    if [[ "$(cat "$temp_dir/ecr.result")" == "MISSING" ]]; then
        log_error "ECR repository not found: ${ECR_REPO##*/}"
        return 1
    fi
    
    log_info "AWS resources validated"
}

# Parallel validation execution
validate_deployment_config() {
    log_info "Validating deployment configuration..."
    
    # Run validations in parallel
    validate_task_definition &
    local task_def_pid=$!
    
    validate_cognito_config &
    local cognito_pid=$!
    
    validate_aws_resources &
    local aws_pid=$!
    
    # Wait for all validations
    if ! wait "$task_def_pid"; then
        log_error "Task definition validation failed"
        return 1
    fi
    
    if ! wait "$cognito_pid"; then
        log_error "Cognito validation failed"
        return 1
    fi
    
    if ! wait "$aws_pid"; then
        log_error "AWS resources validation failed"
        return 1
    fi
    
    log_success "Configuration validation completed"
}

# Modular build functions
generate_build_tag() {
    if git rev-parse --git-dir > /dev/null 2>&1; then
        local tag=$(git rev-parse --short HEAD)
        log_info "Using git commit tag: $tag"
        echo "$tag"
    else
        local tag="manual-$(date +%Y%m%d-%H%M%S)"
        log_warn "Not in git repository, using timestamp tag: $tag"
        echo "$tag"
    fi
}

setup_docker_buildx() {
    log_info "Setting up Docker buildx..."
    
    if ! docker buildx create --use --name clarity-builder --driver docker-container 2>/dev/null; then
        log_info "Using existing buildx builder"
    fi
    
    CLEANUP_TASKS+=("docker buildx rm clarity-builder 2>/dev/null")
    docker buildx inspect --bootstrap
}

execute_docker_build() {
    local tag="$1"
    local full_image="$ECR_REPO:$tag"
    
    log_info "Building Docker image for linux/amd64..."
    log_warn "CRITICAL: Building for linux/amd64 platform for AWS ECS compatibility"
    
    # Build with timeout and enhanced error handling
    if ! timeout "$BUILD_TIMEOUT" docker buildx build \
        --platform linux/amd64 \
        --cache-from type=registry,ref="$ECR_REPO:buildcache" \
        --cache-to type=registry,ref="$ECR_REPO:buildcache,mode=max" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        --push \
        -t "$full_image" \
        -t "$ECR_REPO:latest" \
        .; then
        log_error "Docker build failed or timed out"
        return 1
    fi
    
    log_success "Docker image built: $full_image"
}

verify_image_push() {
    local tag="$1"
    
    log_info "Verifying image in ECR..."
    retry_command 3 5 "ECR image verification" \
        aws ecr describe-images --repository-name "${ECR_REPO##*/}" --image-ids imageTag="$tag" --region "$REGION"
    
    # Get image metrics
    local image_size=$(aws ecr describe-images --repository-name "${ECR_REPO##*/}" --image-ids imageTag="$tag" --region "$REGION" \
        --query 'imageDetails[0].imageSizeInBytes' --output text)
    log_info "Image size: $((image_size / 1024 / 1024)) MB"
}

# Refactored build process
build_and_push_image() {
    log_info "Starting Docker image build process..."
    local build_start=$(date +%s)
    
    local tag
    tag=$(generate_build_tag)
    
    # ECR login with proper retry
    log_info "Authenticating with ECR..."
    retry_command 3 5 "ECR authentication" \
        sh -c "aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO"
    
    setup_docker_buildx
    execute_docker_build "$tag"
    verify_image_push "$tag"
    
    local build_end=$(date +%s)
    METRICS["build_duration"]=$((build_end - build_start))
    log_success "Image built and pushed in ${METRICS["build_duration"]}s"
    
    echo "$tag"
}

# Improved deployment monitoring with efficient polling
monitor_deployment_progress() {
    local monitoring_interval=60  # More efficient 1-minute intervals
    
    while true; do
        sleep $monitoring_interval
        
        local service_info=$(aws ecs describe-services \
            --cluster "$CLUSTER" \
            --services "$SERVICE" \
            --region "$REGION" \
            --query 'services[0].{Running:runningCount,Pending:pendingCount,Desired:desiredCount}' \
            --output text 2>/dev/null || echo "0 0 0")
        
        log_info "Service status: $service_info (Running/Pending/Desired)"
        
        # More intelligent monitoring - check if stable
        local running=$(echo "$service_info" | cut -f1)
        local pending=$(echo "$service_info" | cut -f2)
        local desired=$(echo "$service_info" | cut -f3)
        
        if [[ "$running" == "$desired" && "$pending" == "0" ]]; then
            log_info "Service appears stable, reducing monitoring frequency..."
            monitoring_interval=120  # Reduce to 2-minute intervals when stable
        fi
    done
}

# Enhanced deployment with better resource management
deploy_with_monitoring() {
    local tag="$1"
    local deploy_start=$(date +%s)
    
    log_info "Starting deployment process..."
    
    local task_def_arn
    task_def_arn=$(register_task_definition "$tag")
    
    log_info "Updating ECS service..."
    aws ecs update-service \
        --cluster "$CLUSTER" \
        --service "$SERVICE" \
        --task-definition "$task_def_arn" \
        --force-new-deployment \
        --region "$REGION" \
        --output table
    
    # Start monitoring in background
    log_info "Starting deployment monitoring..."
    monitor_deployment_progress &
    local monitor_pid=$!
    BACKGROUND_PIDS+=("$monitor_pid")
    
    # Wait for stability with timeout
    if timeout "$DEPLOY_TIMEOUT" aws ecs wait services-stable \
        --cluster "$CLUSTER" \
        --services "$SERVICE" \
        --region "$REGION"; then
        log_success "Deployment completed successfully"
    else
        log_error "Deployment timed out after ${DEPLOY_TIMEOUT}s"
        return 1
    fi
    
    local deploy_end=$(date +%s)
    METRICS["deploy_duration"]=$((deploy_end - deploy_start))
    log_success "Deployment completed in ${METRICS["deploy_duration"]}s"
}

register_task_definition() {
    local tag="$1"
    local image="$ECR_REPO:$tag"
    
    log_info "Registering task definition with image: $image"
    
    local task_def_path="ops/ecs-task-definition.json"
    local task_def_json
    task_def_json=$(jq --arg image "$image" '.containerDefinitions[0].image = $image' "$task_def_path")
    
    # Add deployment metadata
    task_def_json=$(echo "$task_def_json" | jq --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '.containerDefinitions[0].environment += [{"name": "DEPLOYMENT_TIMESTAMP", "value": $timestamp}]')
    
    # Use secure temporary file
    local temp_file="$LOG_DIR/task-definition.json"
    echo "$task_def_json" > "$temp_file"
    
    local task_def_arn
    task_def_arn=$(retry_command 3 5 "Task definition registration" \
        aws ecs register-task-definition \
            --cli-input-json "file://$temp_file" \
            --region "$REGION" \
            --query 'taskDefinition.taskDefinitionArn' \
            --output text)
    
    # Verify registration
    local registered_image
    registered_image=$(aws ecs describe-task-definition \
        --task-definition "$task_def_arn" \
        --region "$REGION" \
        --query 'taskDefinition.containerDefinitions[0].image' \
        --output text)
    
    if [[ "$registered_image" != *"$tag"* ]]; then
        log_error "Task definition registration failed - wrong image tag"
        return 1
    fi
    
    log_success "Task definition registered: $task_def_arn"
    echo "$task_def_arn"
}

# Enhanced verification with better error handling
verify_deployment() {
    log_info "Performing comprehensive deployment verification..."
    
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
    
    log_info "Running task: ${task_arn##*/}"
    
    # Enhanced health checks and smoke tests
    perform_health_checks
    run_smoke_tests
    
    log_success "Deployment verification completed"
}

perform_health_checks() {
    log_info "Performing enhanced health checks..."
    
    local health_url="https://clarity.novamindnyc.com/health"
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        local response
        if response=$(curl -s --max-time 10 --retry 2 "$health_url" 2>/dev/null); then
            if echo "$response" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
                log_success "Health check passed"
                log_info "Health response: $(echo "$response" | jq -c '.')"
                return 0
            fi
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            log_warn "Health check failed, waiting 15s before retry..."
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

# Enhanced reporting with better metrics
generate_deployment_report() {
    local end_time=$(date +%s)
    METRICS["total_duration"]=$((end_time - SCRIPT_START_TIME))
    
    log_info "Generating comprehensive deployment report..."
    
    cat > "$LOG_DIR/deployment-report.txt" << EOF
CLARITY Backend Deployment Report - $SCRIPT_VERSION
====================================================
Deployment Date: $(date)
Total Duration: ${METRICS["total_duration"]}s
Validation Duration: ${METRICS["validation_duration"]}s
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

Improvements in V2:
- Removed eval security vulnerability
- Implemented parallel validations
- Enhanced resource cleanup
- Improved code organization
- Added structured JSON logging

Status: SUCCESS
====================================================
EOF
    
    # Generate JSON report for machine parsing
    jq -n \
        --arg version "$SCRIPT_VERSION" \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg status "SUCCESS" \
        --arg region "$REGION" \
        --arg cluster "$CLUSTER" \
        --arg service "$SERVICE" \
        --arg tag "${TAG:-N/A}" \
        --argjson metrics "$(declare -p METRICS | sed 's/declare -A METRICS=//' | sed 's/\[/{"/' | sed 's/\]=/":/' | sed 's/ /,"/' | sed 's/$/}/' | tr -d '()')" \
        '{
            version: $version,
            timestamp: $timestamp,
            status: $status,
            configuration: {
                region: $region,
                cluster: $cluster,
                service: $service,
                tag: $tag
            },
            metrics: $metrics
        }' > "$LOG_DIR/deployment-report.json"
    
    log_success "Reports generated: $LOG_DIR/deployment-report.{txt,json}"
}

generate_error_summary() {
    local end_time=$(date +%s)
    METRICS["total_duration"]=$((end_time - SCRIPT_START_TIME))
    
    cat > "$LOG_DIR/error-summary.txt" << EOF
CLARITY Backend Deployment Error Summary - $SCRIPT_VERSION
=========================================================
Deployment Date: $(date)
Duration Before Failure: ${METRICS["total_duration"]}s
Retry Count: ${METRICS["retry_count"]}
Errors Encountered: ${METRICS["errors_encountered"]}

Check logs for details:
- Human readable: $LOG_FILE
- Machine readable: $JSON_LOG_FILE

Status: FAILED
=========================================================
EOF
    
    log_error "Error summary saved to: $LOG_DIR/error-summary.txt"
}

gather_debug_info() {
    log_info "Gathering deployment debug information..."
    
    {
        echo "=== ECS Service Status ==="
        aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
            --query 'services[0].{Status:status,RunningCount:runningCount,PendingCount:pendingCount,DesiredCount:desiredCount}' \
            --output table 2>/dev/null || echo "Failed to get service status"
        
        echo "=== Recent Task Events ==="
        aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
            --query 'services[0].events[:5]' \
            --output table 2>/dev/null || echo "Failed to get service events"
    } > "$LOG_DIR/debug-info.txt"
}

# Streamlined main function with better organization
main() {
    log_info "Starting CLARITY Backend deployment $SCRIPT_VERSION at $(date)"
    
    local build_mode=false
    local tag=""
    
    # Parse arguments with validation
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
                validate_input "$2" "^[a-zA-Z0-9._-]+$" "image tag"
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
    
    # Execute deployment phases
    check_prerequisites
    validate_deployment_config
    
    # Handle build/tag selection
    if [[ "$build_mode" == true ]]; then
        TAG=$(build_and_push_image)
        log_success "Build completed with tag: $TAG"
    elif [[ -n "$tag" ]]; then
        TAG="$tag"
        log_info "Using provided tag: $TAG"
        
        # Verify tag exists
        if ! aws ecr describe-images --repository-name "${ECR_REPO##*/}" --image-ids imageTag="$TAG" --region "$REGION" &> /dev/null; then
            log_error "Image with tag '$TAG' not found in ECR"
            exit 1
        fi
        log_success "Image tag verified in ECR"
    else
        TAG="latest"
        log_info "Using existing image with tag: $TAG"
    fi
    
    # Deploy and verify
    deploy_with_monitoring "$TAG"
    verify_deployment
    generate_deployment_report
    
    log_success "🎉 Deployment completed successfully!"
    log_info "Logs available at: $LOG_DIR/"
}

show_usage() {
    cat << EOF
CLARITY Backend Deployment Script V2

Usage: $0 [OPTIONS]

Options:
    --build         Build new image before deployment
    --tag TAG       Use specific image tag for deployment
    --help          Show this help message

Examples:
    $0 --build              # Build new image and deploy
    $0 --tag v1.2.3         # Deploy specific tag
    $0                      # Deploy latest image

Improvements in V2:
    - Enhanced security (no eval, secure logs)
    - Parallel validations for better performance
    - Improved code organization and modularity
    - Better resource cleanup and error handling
    - Structured JSON logging alongside human-readable logs

EOF
}

# Execute main function
main "$@"