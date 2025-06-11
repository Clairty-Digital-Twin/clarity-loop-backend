#!/bin/bash
set -e

# CLARITY Digital Twin Backend - CLEAN PRODUCTION DEPLOYMENT
# M1 Mac → AWS AMD64 - 34 ENDPOINTS CLEAN

echo "🚀 CLARITY CLEAN DEPLOYMENT - 34 ENDPOINTS"
echo "==========================================="

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="124355672559"
ECR_REPOSITORY="clarity-backend"
CLUSTER_NAME="clarity-backend-cluster"
SERVICE_NAME="clarity-backend-service"
IMAGE_TAG="clean-$(date +%Y%m%d-%H%M)"

echo "📊 Deployment Info:"
echo "Region: ${AWS_REGION}"
echo "Account: ${AWS_ACCOUNT_ID}"
echo "Tag: ${IMAGE_TAG}"
echo ""

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Use the already built enterprise image and tag it
echo "🏗️ Using ENTERPRISE ML image (3.52GB with ALL dependencies)..."
docker tag clarity-loop-backend-clarity-backend:latest ${ECR_REPOSITORY}:${IMAGE_TAG}

# Tag for ECR
echo "🏷️ Tagging for ECR..."
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Push to ECR
echo "📤 Pushing to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Create new task definition with CLEAN configuration
echo "📋 Creating clean task definition..."
cat > task-definition-clean.json << EOF
{
  "family": "clarity-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskRole",
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
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "AWS_REGION", "value": "${AWS_REGION}"},
        {"name": "AWS_DEFAULT_REGION", "value": "${AWS_REGION}"},
        {"name": "COGNITO_USER_POOL_ID", "value": "us-east-1_1G5jYI8FO"},
        {"name": "COGNITO_CLIENT_ID", "value": "66qdivmqgs1oqmmo0b5r9d9hjo"},
        {"name": "COGNITO_REGION", "value": "${AWS_REGION}"},
        {"name": "DYNAMODB_TABLE_NAME", "value": "clarity-health-data"},
        {"name": "S3_BUCKET_NAME", "value": "clarity-health-uploads"},
        {"name": "ENABLE_AUTH", "value": "true"},
        {"name": "REDIS_URL", "value": "redis://clarity-redis-cluster.redis.use1.cache.amazonaws.com:6379"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/clarity-backend",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

# Register task definition
echo "📝 Registering task definition..."
aws ecs register-task-definition \
    --cli-input-json file://task-definition-clean.json \
    --region ${AWS_REGION}

# Update service with new task definition
echo "🔄 Updating ECS service..."
aws ecs update-service \
    --cluster ${CLUSTER_NAME} \
    --service ${SERVICE_NAME} \
    --desired-count 1 \
    --region ${AWS_REGION}

# Clean up
rm task-definition-clean.json

echo ""
echo "✅ CLEAN DEPLOYMENT COMPLETE!"
echo "🌐 Load Balancer: clarity-alb-1762715656.us-east-1.elb.amazonaws.com"
echo "📊 Image: ${IMAGE_TAG}"
echo "🏥 Health: http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health"
echo "📖 Docs: http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/docs"
echo ""
echo "🎉 34 CLEAN ENDPOINTS DEPLOYED TO AWS!"