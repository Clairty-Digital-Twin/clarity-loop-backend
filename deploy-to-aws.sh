#!/bin/bash
set -e

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="124355672559"
ECR_REPOSITORY="clarity-backend"
IMAGE_TAG="v3-optimized"
CLUSTER_NAME="clarity-backend-cluster"
SERVICE_NAME="clarity-backend-service"

echo "🚀 Starting AWS deployment for CLARITY backend..."

# Step 1: Build optimized Docker image
echo "📦 Building optimized Docker image..."
docker buildx build --platform linux/amd64 \
  -f Dockerfile.aws.optimized \
  -t ${ECR_REPOSITORY}:${IMAGE_TAG} \
  --load .

# Step 2: Tag for ECR
echo "🏷️  Tagging image for ECR..."
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Step 3: Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Step 4: Push to ECR
echo "⬆️  Pushing image to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Step 5: Update task definition
echo "📋 Updating ECS task definition..."
sed -i.bak "s|:v1-amd64|:${IMAGE_TAG}|g" ops/ecs-task-definition-amd64.json

# Step 6: Register new task definition
echo "📝 Registering new task definition..."
TASK_DEF_ARN=$(aws ecs register-task-definition \
  --cli-input-json file://ops/ecs-task-definition-amd64.json \
  --region ${AWS_REGION} \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

echo "✅ New task definition: ${TASK_DEF_ARN}"

# Step 7: Update ECS service
echo "🔄 Updating ECS service..."
aws ecs update-service \
  --cluster ${CLUSTER_NAME} \
  --service ${SERVICE_NAME} \
  --task-definition ${TASK_DEF_ARN} \
  --force-new-deployment \
  --region ${AWS_REGION}

echo "✨ Deployment initiated successfully!"
echo "Monitor progress: aws ecs describe-services --cluster ${CLUSTER_NAME} --services ${SERVICE_NAME} --region ${AWS_REGION}"