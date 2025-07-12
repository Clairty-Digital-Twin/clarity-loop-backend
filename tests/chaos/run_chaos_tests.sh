#!/bin/bash
# Run chaos tests for the CLARITY platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting CLARITY Chaos Tests${NC}"
echo "================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

# Parse command line arguments
RUN_SLOW=false
VERBOSE=false
COVERAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --slow)
            RUN_SLOW=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --slow       Include slow tests"
            echo "  --verbose    Verbose output"
            echo "  --coverage   Generate coverage report"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest tests/chaos/"

if [ "$RUN_SLOW" = false ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
fi

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v --tb=long"
else
    PYTEST_CMD="$PYTEST_CMD --tb=short"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=clarity.ml --cov-report=term-missing"
fi

# Add JUnit XML output for CI
PYTEST_CMD="$PYTEST_CMD --junit-xml=test-results/chaos-tests.xml"

# Create test results directory
mkdir -p test-results

# Run the tests
echo -e "${YELLOW}Running command: $PYTEST_CMD${NC}"
echo ""

if $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}✓ Chaos tests passed successfully!${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo "Coverage report generated. Check the output above."
    fi
    
    exit 0
else
    echo ""
    echo -e "${RED}✗ Chaos tests failed!${NC}"
    echo "Check the output above for details."
    exit 1
fi