#!/bin/bash
# 🔥 CLARITY Development Entrypoint - Ultimate Hot-Reload Experience
# This script provides blazing fast development with intelligent hot-reload

set -e

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${PURPLE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                🚀 CLARITY DEVELOPMENT ENVIRONMENT               ║"
echo "║                   Hot-Reload • Zero Friction                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Development utilities
setup_development_environment() {
    echo -e "${BLUE}🔧 Setting up development environment...${NC}"
    
    # Create necessary directories
    mkdir -p /app/logs /app/dev-data /app/.cache
    
    # Set up Python path
    export PYTHONPATH="/app/src:/app:$PYTHONPATH"
    
    # Install any missing development dependencies
    if [ -f "/app/pyproject.toml" ]; then
        echo -e "${YELLOW}📦 Ensuring development dependencies are installed...${NC}"
        pip install -e /app[dev] --quiet
    fi
    
    echo -e "${GREEN}✅ Development environment ready!${NC}"
}

# Wait for dependencies
wait_for_services() {
    echo -e "${BLUE}⏳ Waiting for dependencies...${NC}"
    
    # Wait for LocalStack
    echo -e "${YELLOW}  • Waiting for LocalStack...${NC}"
    timeout 60 bash -c 'until curl -s http://localstack:4566/_localstack/health > /dev/null; do sleep 2; done'
    
    # Wait for Redis
    echo -e "${YELLOW}  • Waiting for Redis...${NC}"
    timeout 30 bash -c 'until nc -z redis 6379; do sleep 1; done'
    
    # Wait for PostgreSQL
    echo -e "${YELLOW}  • Waiting for PostgreSQL...${NC}"
    timeout 30 bash -c 'until nc -z postgres-dev 5432; do sleep 1; done'
    
    echo -e "${GREEN}✅ All services are ready!${NC}"
}

# Initialize development data
init_dev_data() {
    echo -e "${BLUE}🎲 Initializing development data...${NC}"
    
    # Run initialization scripts if they exist
    if [ -f "/app/dev-scripts/init-localstack.sh" ]; then
        echo -e "${YELLOW}  • Initializing LocalStack resources...${NC}"
        /app/dev-scripts/init-localstack.sh
    fi
    
    if [ -f "/app/dev-scripts/seed-data.py" ]; then
        echo -e "${YELLOW}  • Seeding development data...${NC}"
        python /app/dev-scripts/seed-data.py
    fi
    
    echo -e "${GREEN}✅ Development data initialized!${NC}"
}

# Start hot-reload development server
start_hot_reload_server() {
    echo -e "${BLUE}🔥 Starting hot-reload server...${NC}"
    echo -e "${CYAN}📡 Server will be available at: http://localhost:8000${NC}"
    echo -e "${CYAN}📚 API docs available at: http://localhost:8000/docs${NC}"
    echo -e "${CYAN}🔍 Health check: http://localhost:8000/health${NC}"
    echo ""
    echo -e "${GREEN}🚀 Hot-reload active - your changes will be reflected instantly!${NC}"
    echo ""
    
    # Use uvicorn with optimal hot-reload settings
    exec uvicorn clarity.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --reload-dir /app/src \
        --reload-dir /app/pyproject.toml \
        --reload-delay 0.5 \
        --log-level debug \
        --access-log \
        --use-colors \
        --loop uvloop \
        --http httptools
}

# Start with file watching for ultra-fast reload
start_with_watchdog() {
    echo -e "${BLUE}👁️  Starting with advanced file watching...${NC}"
    
    # Use watchmedo for even faster reloads
    watchmedo auto-restart \
        --directory /app/src \
        --pattern "*.py" \
        --recursive \
        --ignore-pattern "*.pyc;__pycache__/*;.git/*" \
        -- uvicorn clarity.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level debug \
        --access-log \
        --use-colors \
        --loop uvloop \
        --http httptools
}

# Run tests in watch mode
start_test_watch() {
    echo -e "${BLUE}🧪 Starting test watch mode...${NC}"
    
    exec ptw /app/tests /app/src \
        --runner "pytest -v --tb=short --no-header" \
        --ignore /app/src/clarity/__pycache__ \
        --ignore /app/tests/__pycache__
}

# Start interactive Python shell with context
start_shell() {
    echo -e "${BLUE}🐍 Starting interactive Python shell...${NC}"
    
    # Pre-import common modules
    python -c "
import sys
sys.path.insert(0, '/app/src')

# Pre-import common modules for convenience
from clarity.main import app
from clarity.core.config import get_settings
from clarity.services.health_data_service import HealthDataService

print('🚀 Clarity Development Shell')
print('Available imports:')
print('  • app - FastAPI application')
print('  • get_settings() - Configuration')
print('  • HealthDataService - Health data service')
print('')
print('Happy coding! 🎉')
" -i
}

# Show development dashboard
show_dashboard() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                     🎛️  DEVELOPMENT DASHBOARD                   ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  🚀 Main App:          http://localhost:8000                    ║"
    echo "║  📚 API Docs:          http://localhost:8000/docs               ║"
    echo "║  🔍 Health Check:      http://localhost:8000/health             ║"
    echo "║  📊 Swagger UI:        http://localhost:8001                    ║"
    echo "║  🗄️  Database Admin:    http://localhost:8002                    ║"
    echo "║  🔴 Redis Admin:       http://localhost:8003                    ║"
    echo "║  📁 File Browser:      http://localhost:8004                    ║"
    echo "║  📧 Mail Catcher:      http://localhost:8025                    ║"
    echo "║  📈 Grafana:           http://localhost:3000                    ║"
    echo "║  🎯 Prometheus:        http://localhost:9090                    ║"
    echo "║  📔 Jupyter Lab:       http://localhost:8888                    ║"
    echo "║  🌐 Traefik Dashboard: http://localhost:8080                    ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Main execution
main() {
    local command="${1:-dev}"
    
    case "$command" in
        "dev")
            setup_development_environment
            wait_for_services
            init_dev_data
            show_dashboard
            start_hot_reload_server
            ;;
        "watch")
            setup_development_environment
            wait_for_services
            init_dev_data
            show_dashboard
            start_with_watchdog
            ;;
        "test")
            setup_development_environment
            start_test_watch
            ;;
        "shell")
            setup_development_environment
            start_shell
            ;;
        "dashboard")
            show_dashboard
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $command${NC}"
            echo "Available commands:"
            echo "  dev     - Start development server with hot-reload"
            echo "  watch   - Start with advanced file watching"
            echo "  test    - Start test watch mode"
            echo "  shell   - Start interactive Python shell"
            echo "  dashboard - Show development dashboard"
            exit 1
            ;;
    esac
}

# Handle signals gracefully
trap 'echo -e "\n${YELLOW}🛑 Shutting down development server...${NC}"; exit 0' SIGTERM SIGINT

# Run main function
main "$@"