#!/bin/bash
set -e

# CLARITY Digital Twin Backend - Production Deployment Script
# Deploys to AWS ECS Fargate in us-east-1

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="124355672559"
ECR_REPOSITORY="clarity-backend"
IMAGE_TAG="production-$(date +%Y%m%d-%H%M%S)"
CLUSTER_NAME="clarity-backend-cluster"
SERVICE_NAME="clarity-backend-service"
ALB_DNS="***REMOVED***"

echo "🚀 CLARITY Digital Twin - Production Deployment"
echo "================================================"
echo "Region: ${AWS_REGION}"
echo "Repository: ${ECR_REPOSITORY}"
echo "Tag: ${IMAGE_TAG}"
echo ""

# Build the production image - USING FULL DOCKERFILE
echo "📦 Building production image with FULL APPLICATION..."
docker build -f Dockerfile.aws.full -t ${ECR_REPOSITORY}:${IMAGE_TAG} --platform linux/amd64 .

echo "🏷️ Tagging for ECR..."
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Login to ECR
echo "🔐 Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "📤 Pushing image to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Create task definition
echo "📝 Creating ECS task definition..."
cat > /tmp/clarity-task-definition.json <<EOF
{
  "family": "clarity-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/clarity-backend-task-role",
  "containerDefinitions": [{
    "name": "clarity-backend",
    "image": "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}",
    "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
    "environment": [
      {"name": "ENVIRONMENT", "value": "production"},
      {"name": "AWS_REGION", "value": "${AWS_REGION}"},
      {"name": "AWS_DEFAULT_REGION", "value": "${AWS_REGION}"},
      {"name": "COGNITO_USER_POOL_ID", "value": "us-east-2_xqTJHGxmY"},
      {"name": "COGNITO_CLIENT_ID", "value": "6s5j0f1aiqddqsutrgvg6mjkfr"},
      {"name": "COGNITO_REGION", "value": "us-east-2"},
      {"name": "DYNAMODB_TABLE_NAME", "value": "clarity-health-data"},
      {"name": "S3_BUCKET_NAME", "value": "clarity-health-uploads"},
      {"name": "ENABLE_AUTH", "value": "false"},
      {"name": "PORT", "value": "8000"}
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
      "timeout": 10,
      "retries": 3,
      "startPeriod": 60
    }
  }]
}
EOF

# Register task definition
echo "📋 Registering task definition..."
TASK_ARN=$(aws ecs register-task-definition \
  --cli-input-json file:///tmp/clarity-task-definition.json \
  --region ${AWS_REGION} \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

echo "✅ Task definition registered: ${TASK_ARN}"

# Deploy to ECS
echo "🚀 Deploying to ECS..."
aws ecs update-service \
  --cluster ${CLUSTER_NAME} \
  --service ${SERVICE_NAME} \
  --task-definition ${TASK_ARN} \
  --force-new-deployment \
  --desired-count 1 \
  --region ${AWS_REGION}

echo ""
echo "✅ DEPLOYMENT COMPLETE"
echo "======================"
echo "⏱️  Wait 2-3 minutes for container deployment"
echo ""
echo "🔍 Monitor deployment:"
echo "aws ecs describe-services --cluster ${CLUSTER_NAME} --services ${SERVICE_NAME} --region ${AWS_REGION}"
echo ""
echo "📋 View logs:"
echo "aws logs tail /ecs/clarity-backend --region ${AWS_REGION} --follow"
echo ""
echo "🌐 Test endpoints:"
echo "curl http://${ALB_DNS}/health"
echo "curl http://${ALB_DNS}/openapi.json | jq '.paths | keys | length'"
echo ""
echo "Expected: 35+ endpoints with healthy status"
echo "Service: clarity-backend-production"

# Clean up
rm -f /tmp/clarity-task-definition.json

echo "🎉 Production deployment initiated successfully!" 