#!/bin/bash
set -e

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-"124355672559"}
ECR_REPOSITORY="clarity-backend"
IMAGE_TAG="minimal-latest"
CLUSTER_NAME="clarity-backend-cluster"
SERVICE_NAME="clarity-backend-service"

echo "🚀 Starting AWS deployment for CLARITY backend..."
echo "📋 Configuration:"
echo "  Region: $AWS_REGION"
echo "  Account: $AWS_ACCOUNT_ID"
echo "  Image: $IMAGE_TAG"

# Step 1: Build optimized Docker image
echo "📦 Building minimal Docker image for AWS..."
docker buildx build --platform linux/amd64 \
  -f Dockerfile.aws.clean \
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

# Step 5: Check if task definition exists, if not create a simple one
echo "📋 Checking ECS task definition..."
if [ -f "ops/ecs-task-definition-amd64.json" ]; then
    echo "Using existing task definition file..."
    sed -i.bak "s|:v1-amd64|:${IMAGE_TAG}|g" ops/ecs-task-definition-amd64.json
    TASK_DEF_FILE="ops/ecs-task-definition-amd64.json"
else
    echo "Creating minimal task definition..."
    cat > /tmp/minimal-task-definition.json <<EOF
{
  "family": "clarity-backend-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "clarity-backend",
      "image": "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "SKIP_EXTERNAL_SERVICES",
          "value": "true"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/clarity-backend",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
EOF
    TASK_DEF_FILE="/tmp/minimal-task-definition.json"
fi

# Step 6: Register new task definition
echo "📝 Registering new task definition..."
TASK_DEF_ARN=$(aws ecs register-task-definition \
  --cli-input-json file://${TASK_DEF_FILE} \
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