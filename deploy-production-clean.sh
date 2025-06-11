#!/bin/bash
set -e

# CLARITY Digital Twin Backend - CLEAN Production Deployment Script
# Deploys ALL 38+ endpoints to AWS ECS

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="124355672559"
ECR_REPOSITORY="clarity-backend"
IMAGE_TAG="production-clean-$(date +%Y%m%d-%H%M%S)"
CLUSTER_NAME="clarity-backend-cluster"
SERVICE_NAME="clarity-backend-service"
ALB_DNS="***REMOVED***"

echo "🚀 CLARITY Digital Twin - CLEAN Production Deployment"
echo "===================================================="
echo "Region: ${AWS_REGION}"
echo "Repository: ${ECR_REPOSITORY}"
echo "Tag: ${IMAGE_TAG}"
echo ""

# Build the production image with ALL endpoints
echo "📦 Building production image with ALL 38+ endpoints..."
docker build -f Dockerfile.aws.production -t ${ECR_REPOSITORY}:${IMAGE_TAG} --platform linux/amd64 .

echo "🏷️ Tagging for ECR..."
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Also tag as latest
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest

# Login to ECR
echo "🔐 Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "📤 Pushing images to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest

# Update the main ECS task definition to use latest
echo "📋 Updating ECS task definition..."
aws ecs update-service \
  --cluster ${CLUSTER_NAME} \
  --service ${SERVICE_NAME} \
  --task-definition clarity-backend \
  --force-new-deployment \
  --desired-count 1 \
  --region ${AWS_REGION}

echo ""
echo "✅ DEPLOYMENT COMPLETE"
echo "======================"
echo "⏱️  Wait 3-5 minutes for full deployment"
echo ""
echo "🔍 Monitor deployment:"
echo "aws ecs describe-services --cluster ${CLUSTER_NAME} --services ${SERVICE_NAME} --region ${AWS_REGION} | jq '.services[0].deployments'"
echo ""
echo "📋 View logs:"
echo "aws logs tail /ecs/clarity-backend --region ${AWS_REGION} --follow"
echo ""
echo "🌐 Test endpoints:"
echo "curl http://${ALB_DNS}/health"
echo "curl http://${ALB_DNS}/"
echo "curl http://${ALB_DNS}/openapi.json | jq '.paths | keys | length'"
echo ""
echo "✅ Expected: 34+ endpoints (similar to local deployment)"
echo "✅ Service: clarity-backend-aws-full"
echo ""
echo "🧹 Clean up old untagged images:"
echo "aws ecr list-images --repository-name ${ECR_REPOSITORY} --filter tagStatus=UNTAGGED --query 'imageIds[*]' > /tmp/images-to-delete.json"
echo "aws ecr batch-delete-image --repository-name ${ECR_REPOSITORY} --image-ids file:///tmp/images-to-delete.json"
echo ""
echo "🎉 Production deployment initiated successfully!"