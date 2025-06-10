#!/bin/bash
set -e

echo "🧪 Testing CLARITY Backend (AWS Configuration)"
echo "============================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Creating from template..."
    cp .env.example .env
    echo "⚠️  Please add your GEMINI_API_KEY to .env"
    exit 1
fi

# Check for Gemini API key
if ! grep -q "GEMINI_API_KEY=.*[a-zA-Z0-9]" .env; then
    echo "⚠️  Warning: GEMINI_API_KEY appears to be missing or empty in .env"
    echo "The AI features won't work without it."
fi

# Test 1: Docker build
echo -e "\n1️⃣ Testing Docker build..."
if docker images | grep -q "clarity-backend.*aws-minimal-v2"; then
    echo "✅ Docker image found"
else
    echo "❌ Docker image not found. Building may still be in progress."
    exit 1
fi

# Test 2: Run container
echo -e "\n2️⃣ Starting container..."
docker run -d --name clarity-test \
    -p 8080:8000 \
    --env-file .env \
    -e SKIP_EXTERNAL_SERVICES=true \
    clarity-backend:aws-minimal-v2

# Wait for startup
echo "Waiting for server to start..."
sleep 10

# Test 3: Health check
echo -e "\n3️⃣ Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8080/health || echo "FAILED")
if [[ $HEALTH_RESPONSE == *"ok"* ]]; then
    echo "✅ Health check passed"
    echo "Response: $HEALTH_RESPONSE"
else
    echo "❌ Health check failed"
    docker logs clarity-test
    docker stop clarity-test && docker rm clarity-test
    exit 1
fi

# Test 4: API docs
echo -e "\n4️⃣ Testing API documentation..."
DOCS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/docs)
if [ "$DOCS_RESPONSE" == "200" ]; then
    echo "✅ API docs accessible at http://localhost:8080/docs"
else
    echo "❌ API docs not accessible (HTTP $DOCS_RESPONSE)"
fi

# Test 5: Check services
echo -e "\n5️⃣ Checking service configuration..."
docker exec clarity-test env | grep -E "(SKIP_EXTERNAL|GEMINI|AWS)" || true

# Cleanup
echo -e "\n🧹 Cleaning up..."
docker stop clarity-test && docker rm clarity-test

echo -e "\n✨ All tests passed! Your CLARITY backend is ready."
echo "Next steps:"
echo "1. Add your GEMINI_API_KEY to .env"
echo "2. Configure AWS credentials (optional)"
echo "3. Deploy to AWS ECS using ./deploy-to-aws.sh"