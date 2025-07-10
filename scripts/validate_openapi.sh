#!/bin/bash
# Validate OpenAPI specification

set -e

echo "🔍 Validating OpenAPI specification..."

# Generate fresh OpenAPI spec
echo "📝 Generating OpenAPI spec..."
python3 scripts/generate_openapi.py

# Validate with Spectral
echo "🎯 Running Spectral validation..."
npx @stoplight/spectral-cli lint docs/api/openapi.json --ruleset .spectral.yml

# Validate with Redocly
echo "🔴 Running Redocly validation..."
npx @redocly/cli lint docs/api/openapi.json

# Validate schema structure
echo "📋 Validating schema structure..."
npx @apidevtools/swagger-cli validate docs/api/openapi.json

echo "✅ OpenAPI validation complete!"