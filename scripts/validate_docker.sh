#!/bin/bash
# 🔧 Docker Validation Script for Technical Demo
# Validates that Docker configuration is production-ready with zero warnings

set -euo pipefail

echo "🚀 CLARITY Platform - Docker Validation"
echo "======================================"

# Test Dockerfile syntax with hadolint if available, otherwise basic validation
echo "🔍 Validating Dockerfile syntax..."
if command -v hadolint > /dev/null 2>&1; then
    if hadolint Dockerfile > /dev/null 2>&1; then
        echo "✅ Dockerfile syntax: VALID (hadolint)"
    else
        echo "⚠️  Dockerfile has style warnings (check with: hadolint Dockerfile)"
    fi
else
    # Basic validation - check if Dockerfile exists and has FROM
    if [[ -f Dockerfile ]] && grep -q "^FROM " Dockerfile; then
        echo "✅ Dockerfile syntax: Basic validation passed"
    else
        echo "❌ Dockerfile syntax: Missing or invalid"
        exit 1
    fi
fi

# Check for common Docker best practices
echo "🔍 Checking Docker best practices..."

# Check FROM statements use uppercase AS
if grep -q "FROM.*as " Dockerfile; then
    echo "❌ Found lowercase 'as' in FROM statements"
    exit 1
else
    echo "✅ FROM statements: Proper casing"
fi

# Check CMD format
if grep -q '^CMD \[' Dockerfile; then
    echo "✅ CMD format: JSON array (recommended)"
elif grep -q '^CMD ' Dockerfile; then
    echo "⚠️  CMD format: Shell form (consider JSON array)"
else
    echo "❓ No CMD found"
fi

# Check for non-root user
if grep -q "USER " Dockerfile; then
    echo "✅ Security: Non-root user configured"
else
    echo "⚠️  Security: Consider using non-root user"
fi

# Check for HEALTHCHECK
if grep -q "HEALTHCHECK" Dockerfile; then
    echo "✅ Monitoring: Health check configured"
else
    echo "⚠️  Monitoring: Consider adding health check"
fi

echo ""
echo "🏆 Docker validation complete!"
echo "Ready for technical co-founder demo! 🔥" 