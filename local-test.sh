#!/bin/bash
# Local Testing Script - Uses ARM64 on Apple Silicon automatically

echo "🚀 Starting Local Clarity Backend..."
echo "📱 This will use ARM64 architecture on Apple Silicon"
echo ""

# Build and start
docker-compose up --build -d

# Wait for health
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Test endpoints
echo ""
echo "🧪 Testing endpoints..."
echo "Health Check:"
curl -s http://localhost:8000/health | jq .

echo ""
echo "API Info:"
curl -s http://localhost:8000/api/v1/ | jq .

echo ""
echo "✅ Local environment ready!"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🛑 To stop: docker-compose down"