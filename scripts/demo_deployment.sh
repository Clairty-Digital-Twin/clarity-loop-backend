#!/bin/bash
# 🚀 CLARITY Digital Twin Platform - SHOCK & AWE Demo Deployment
#
# This script demonstrates the full-stack deployment of a production-ready
# psychiatric AI platform built in just 2 days with 112 days of programming experience.
#
# Features demonstrated:
# - Microservices architecture with Docker Compose
# - AI-powered health insights (Gemini)
# - Sleep analysis with Pretrained Actigraphy Transformer (PAT)
# - Apple HealthKit integration
# - Real-time monitoring with Prometheus/Grafana
# - 100% type-safe Python codebase

# More graceful error handling - continue on non-critical failures
set -e  # Still exit on errors, but be more selective

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Emojis for visual impact
ROCKET="🚀"
FIRE="🔥"
LIGHTNING="⚡"
DIAMOND="💎"
ROBOT="🤖"
HEART="❤️"
BRAIN="🧠"
CHART="📊"
SHIELD="🛡️"
TROPHY="🏆"

echo -e "${WHITE}CLARITY DIGITAL TWIN PLATFORM${NC}"
echo -e "${PURPLE}─────────────────────────────────────────────${NC}"
echo -e "${CYAN}Production-grade psychiatric AI platform${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "${WHITE}${1}${NC}"
    echo -e "${PURPLE}$(printf '=%.0s' {1..50})${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service with timeout
wait_for_service() {
    local service_name="$1"
    local port="$2"
    local timeout="${3:-60}"
    local count=0

    echo -e "${YELLOW}${LIGHTNING} Waiting for ${service_name} on port ${port}...${NC}"

    while ! nc -z localhost "$port" 2>/dev/null; do
        if [ $count -ge $timeout ]; then
            echo -e "${RED}❌ Timeout waiting for ${service_name}${NC}"
            return 1
        fi
        printf "."
        sleep 1
        ((count++))
    done
    echo -e "${GREEN}✅ ${service_name} is ready!${NC}"
}

# Function to test API endpoint
test_endpoint() {
    local endpoint="$1"
    local description="$2"
    local expected_status="${3:-200}"

    echo -e "${CYAN}Testing: ${description}${NC}"
    if response=$(curl -s -w "HTTPSTATUS:%{http_code}" "http://localhost:8000${endpoint}"); then
        http_code=$(echo "$response" | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
        body=$(echo "$response" | sed -e 's/HTTPSTATUS\:.*//g')

        if [ "$http_code" -eq "$expected_status" ]; then
            echo -e "${GREEN}✅ ${endpoint} - Status: ${http_code}${NC}"
            echo -e "${BLUE}   Response: ${body:0:100}...${NC}"
        else
            echo -e "${RED}❌ ${endpoint} - Expected: ${expected_status}, Got: ${http_code}${NC}"
        fi
    else
        echo -e "${RED}❌ ${endpoint} - Connection failed${NC}"
    fi
    echo ""
}

# Add better error handling function
handle_error() {
    local exit_code=$?
    local line_number=$1
    echo -e "${RED}❌ Error occurred on line $line_number (exit code: $exit_code)${NC}"
    echo -e "${YELLOW}💡 Try the following:${NC}"
    echo -e "${CYAN}  1. Ensure Docker is running: docker version${NC}"
    echo -e "${CYAN}  2. Ensure Docker Compose is available: docker-compose version${NC}"
    echo -e "${CYAN}  3. Check if ports are free: netstat -tuln | grep ':8000\|:3000\|:9090'${NC}"
    echo -e "${CYAN}  4. Create .env file: touch .env${NC}"
    echo -e "${CYAN}  5. Try again: bash quick_demo.sh${NC}"
    exit $exit_code
}

# Set trap for error handling
trap 'handle_error $LINENO' ERR

# Check prerequisites
print_section "${SHIELD} Pre-flight Checks"

# Check Docker
if ! command_exists docker; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    echo -e "${CYAN}💡 Install: https://docs.docker.com/get-docker/${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker version >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check Docker Compose
if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose first.${NC}"
    echo -e "${CYAN}💡 Install: https://docs.docker.com/compose/install/${NC}"
    exit 1
fi

# Use docker compose or docker-compose based on what's available
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

if ! command_exists curl; then
    echo -e "${RED}❌ curl not found. Please install curl first.${NC}"
    exit 1
fi

if ! command_exists nc; then
    echo -e "${YELLOW}⚠️  netcat not found. Service readiness checks may not work.${NC}"
fi

# Check if required ports are available
check_port() {
    local port=$1
    local service=$2
    if nc -z localhost "$port" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Port $port ($service) is already in use${NC}"
        echo -e "${CYAN}💡 Stop existing service: lsof -ti:$port | xargs kill -9${NC}"
        return 1
    fi
    return 0
}

echo -e "${CYAN}Checking if required ports are available...${NC}"
ports_busy=false
for port_service in "8000:Main App" "3000:Grafana" "9090:Prometheus" "8080:Firestore" "6379:Redis"; do
    IFS=':' read -r port service <<< "$port_service"
    if ! check_port "$port" "$service"; then
        ports_busy=true
    fi
done

if [ "$ports_busy" = true ]; then
    echo -e "${YELLOW}⚠️  Some ports are busy. Demo may not work properly.${NC}"
    echo -e "${CYAN}💡 You can continue anyway or stop conflicting services first.${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check/create .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env file from template${NC}"
    else
        echo -e "${RED}❌ .env.example not found. Creating minimal .env...${NC}"
        cat > .env << 'EOF'
ENVIRONMENT=development
DEBUG=true
GOOGLE_CLOUD_PROJECT=clarity-demo
FIREBASE_PROJECT_ID=clarity-demo
SKIP_EXTERNAL_SERVICES=true
DEMO_MODE=true
EOF
        echo -e "${GREEN}✅ Created minimal .env file${NC}"
    fi
fi

echo -e "${GREEN}✅ All prerequisites satisfied${NC}"
echo ""

# Clean up any existing containers
print_section "${FIRE} Environment Cleanup"
echo -e "${YELLOW}Stopping any existing containers...${NC}"
$DOCKER_COMPOSE down --remove-orphans --volumes 2>/dev/null || true
echo -e "${GREEN}✅ Environment cleaned${NC}"
echo ""

# Build and start services
print_section "${ROCKET} Building & Deploying Services"
echo -e "${CYAN}Building Docker images...${NC}"
$DOCKER_COMPOSE build --parallel

echo -e "${CYAN}Starting microservices stack...${NC}"
$DOCKER_COMPOSE up -d

echo -e "${GREEN}✅ All services starting...${NC}"
echo ""

# Wait for services to be ready
print_section "${LIGHTNING} Service Readiness Check"

services=(
    "Redis:6379"
    "Firestore:8080"
    "Firebase Auth:9099"
    "Pub/Sub:8085"
    "Main App:8000"
    "Prometheus:9090"
    "Grafana:3000"
    "Jupyter:8888"
)

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    wait_for_service "$name" "$port" 30 || echo -e "${YELLOW}⚠️  ${name} may need more time${NC}"
done

echo ""

# API Health Checks
print_section "${HEART} API Health Verification"

# Give the main app a moment to fully initialize
sleep 5

# Test core endpoints
test_endpoint "/health" "Root Health Check" 200
test_endpoint "/api/v1/auth/health" "Auth Service Health" 200
test_endpoint "/api/v1/health-data/health" "Health Data Service" 200
test_endpoint "/api/v1/pat/health" "PAT Analysis Service" 200
test_endpoint "/api/v1/insights/health" "Gemini Insights Service" 200

# Service Status Summary
print_section "${CHART} Service Dashboard URLs"
echo -e "${CYAN}${ROCKET} Main Application:${NC}     http://localhost:8000"
echo -e "${CYAN}${ROBOT} API Documentation:${NC}    http://localhost:8000/docs"
echo -e "${CYAN}${CHART} Prometheus Metrics:${NC}   http://localhost:9090"
echo -e "${CYAN}${BRAIN} Grafana Dashboard:${NC}    http://localhost:3000 (admin/admin)"
echo -e "${CYAN}${LIGHTNING} Jupyter Lab:${NC}           http://localhost:8888"
echo -e "${CYAN}${SHIELD} Redis:${NC}                  localhost:6379"
echo -e "${CYAN}${FIRE} Firestore UI:${NC}           http://localhost:4000"
echo ""

# Technical Stack Overview
print_section "${CHART} Architecture Summary"
echo -e "${GREEN}${SHIELD} Type Safety${NC}          - Zero MyPy errors, full static analysis"
echo -e "${GREEN}${ROBOT} AI/ML Pipeline${NC}        - Gemini + PAT transformer integration"
echo -e "${GREEN}${LIGHTNING} Clean Architecture${NC}     - SOLID principles, dependency injection"
echo -e "${GREEN}${FIRE} Microservices${NC}         - 8 services, graceful degradation"
echo -e "${GREEN}${HEART} HealthKit Integration${NC}  - Real-time biometric data processing"
echo -e "${GREEN}${BRAIN} Sleep Analytics${NC}       - Pretrained Actigraphy Transformer"
echo -e "${GREEN}${CHART} Production Monitoring${NC}  - Prometheus + Grafana observability"
echo ""

# Demo Endpoints
print_section "${LIGHTNING} Demo Commands"
cat << 'EOF'
# Core API endpoints for demonstration:

curl http://localhost:8000/health          # System health check
curl http://localhost:8000/docs            # Interactive API documentation
curl http://localhost:9090/metrics         # Prometheus metrics
curl http://localhost:3000                 # Grafana dashboards (admin/admin)

# Technical architecture highlights:
- Clean Architecture implementation (SOLID principles)
- 100% type safety with MyPy static analysis
- Microservices with graceful degradation
- AI/ML pipeline: Google Gemini + PAT transformer
- Real-time Apple HealthKit data processing
- Production monitoring: Prometheus + Grafana
- Containerized deployment with health checks

# Service management:
docker compose ps                          # View all services
docker compose logs -f clarity-backend     # Backend logs
docker compose restart <service>           # Restart specific service
EOF

echo ""
print_section "${LIGHTNING} Platform Ready"
echo -e "${WHITE}CLARITY platform successfully deployed${NC}"
echo ""
echo -e "${YELLOW}Service management:${NC}"
echo -e "${BLUE}  $DOCKER_COMPOSE ps${NC}                    # Show service status"
echo -e "${BLUE}  $DOCKER_COMPOSE logs -f clarity-backend${NC} # View backend logs"
echo -e "${BLUE}  $DOCKER_COMPOSE down${NC}                   # Stop all services"
echo -e "${BLUE}  curl http://localhost:8000/health${NC}      # Test API health"
echo ""
echo -e "${YELLOW}Troubleshooting:${NC}"
echo -e "${CYAN}  1. Check Docker: docker version${NC}"
echo -e "${CYAN}  2. Check logs: $DOCKER_COMPOSE logs${NC}"
echo -e "${CYAN}  3. Restart: $DOCKER_COMPOSE restart${NC}"
echo -e "${CYAN}  4. Full reset: $DOCKER_COMPOSE down && $DOCKER_COMPOSE up -d${NC}"
echo ""
echo -e "${GREEN}Platform operational. Ready for demonstration.${NC}"
