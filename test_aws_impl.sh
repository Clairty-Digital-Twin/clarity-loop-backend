#!/bin/bash
# Test script for running pytest with AWS implementation

set -e  # Exit on error

echo "🧪 Testing AWS implementation with pytest..."

# Load environment variables
if [ -f .env.development ]; then
    set -a
    source .env.development
    set +a
fi

# Set AWS credentials for local testing
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test

# Set application environment
export ENVIRONMENT=development
export DEBUG=true
export LOG_LEVEL=DEBUG

# Skip external services for local testing
export SKIP_EXTERNAL_SERVICES=true

# Use AWS implementation
export USE_AWS_IMPL=true

# Run specific test or all tests
if [ -n "$1" ]; then
    echo "Running specific test: $1"
    python3 -m pytest "$1" -v
else
    echo "Running all tests..."
    python3 -m pytest tests/ -v --tb=short
fi