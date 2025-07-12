#!/bin/bash
# Deploy PAT model path fixes to production

set -e

echo "🚀 Deploying PAT model path fixes..."

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t clarity-backend .

echo "🏷️  Tagging image..."
docker tag clarity-backend:latest 124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend:latest

echo "🔐 Logging into ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 124355672559.dkr.ecr.us-east-1.amazonaws.com

echo "⬆️  Pushing image..."
docker push 124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend:latest

echo "📝 Updating ECS task definition..."
aws ecs register-task-definition \
    --cli-input-json file://ops/deployment/ecs-task-definition.json \
    --region us-east-1

echo "🔄 Updating ECS service..."
aws ecs update-service \
    --cluster clarity-backend-cluster \
    --service clarity-backend-service \
    --force-new-deployment \
    --region us-east-1

echo "✅ Deployment initiated!"
echo ""
echo "📊 Monitor deployment progress:"
echo "aws ecs describe-services --cluster clarity-backend-cluster --services clarity-backend-service --region us-east-1 | jq '.services[0].deployments'"
echo ""
echo "📋 View logs:"
echo "aws logs tail /aws/ecs/clarity-backend --follow --region us-east-1"