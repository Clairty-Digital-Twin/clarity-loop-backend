#!/bin/bash
set -e

# Configuration
AWS_REGION="us-east-1"  # IMPORTANT: Changed to us-east-1 where ECS is deployed
AWS_ACCOUNT_ID="124355672559"
ECR_REPOSITORY="clarity-backend"
IMAGE_TAG="nuclear-clean-v1"
CLUSTER_NAME="clarity-backend-cluster"
SERVICE_NAME="clarity-backend-service"

echo "🚀 Starting NUCLEAR CLEAN deployment for CLARITY backend..."
echo "📋 Configuration:"
echo "  Region: $AWS_REGION (ECS deployment region)"
echo "  Account: $AWS_ACCOUNT_ID"
echo "  Image: $IMAGE_TAG"
echo "  Main Module: clarity.main_aws_nuclear_clean:app"

# Step 1: Build optimized Docker image
echo "📦 Building NUCLEAR CLEAN Docker image..."
docker buildx build --platform linux/amd64 \
  -f Dockerfile.aws.clean \
  -t ${ECR_REPOSITORY}:${IMAGE_TAG} \
  --load .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker build succeeded!"

# Step 2: Tag for ECR
echo "🏷️  Tagging image for ECR..."
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Step 3: Login to ECR
echo "🔐 Logging into ECR (us-east-1)..."
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Step 4: Push to ECR
echo "⬆️  Pushing image to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Step 5: Update task definition with new image
echo "📝 Updating task definition..."
# Update the existing task definition with the new image tag
sed "s|:minimal-latest|:${IMAGE_TAG}|g" ops/ecs-task-definition-amd64.json > /tmp/nuclear-task-def.json

# Register new task definition
TASK_DEF_ARN=$(aws ecs register-task-definition \
  --cli-input-json file:///tmp/nuclear-task-def.json \
  --region ${AWS_REGION} \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

echo "✅ New task definition: ${TASK_DEF_ARN}"

# Step 6: Force new deployment
echo "🔄 Forcing new deployment..."
aws ecs update-service \
  --cluster ${CLUSTER_NAME} \
  --service ${SERVICE_NAME} \
  --task-definition ${TASK_DEF_ARN} \
  --force-new-deployment \
  --region ${AWS_REGION}

echo "✨ NUCLEAR CLEAN deployment initiated!"
echo ""
echo "📊 Monitor deployment:"
echo "aws ecs describe-services --cluster ${CLUSTER_NAME} --services ${SERVICE_NAME} --region ${AWS_REGION}"
echo ""
echo "📋 View logs:"
echo "aws logs tail /ecs/clarity-backend --region us-east-2 --follow"
echo ""
echo "🔍 Expected result:"
echo "- Health endpoint should return: \"service\": \"clarity-backend-aws-nuclear\""
echo "- Total endpoints: 35+"
echo ""
echo "🌐 Test URL: http://***REMOVED***/health"