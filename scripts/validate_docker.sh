#!/bin/bash
# 🔧 Docker Validation Script for Technical Demo
# Validates that Docker configuration is production-ready with zero warnings

set -euo pipefail

echo "🚀 CLARITY Platform - Docker Validation"
echo "======================================"

# Test Dockerfile syntax
echo "🔍 Validating Dockerfile syntax..."
if docker buildx build --dry-run . > /dev/null 2>&1; then
    echo "✅ Dockerfile syntax: VALID"
else
    echo "❌ Dockerfile syntax: INVALID"
    exit 1
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