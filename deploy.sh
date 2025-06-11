#!/bin/bash
set -e

# CLARITY Digital Twin - Professional AWS Deployment
# Enterprise ML Health Platform - Single Clean Deployment Script

echo "🚀 CLARITY ENTERPRISE DEPLOYMENT"
echo "================================"

# Configuration - STANDARDIZED to us-east-1
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

# Build enterprise ML image
echo "🏗️ Building enterprise ML image..."
docker build \
    -t ${ECR_REPOSITORY}:${IMAGE_TAG} \
    --platform linux/amd64 \
    .

# Tag for ECR
echo "🏷️ Tagging for ECR..."
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Push to ECR
echo "📤 Pushing to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Create production task definition with FIXED MODULE PATH
echo "📋 Creating production task definition..."
cat > task-definition.json << EOF
{
  "family": "clarity-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/clarity-backend-task-role",
  "runtimePlatform": {
    "cpuArchitecture": "X86_64",
    "operatingSystemFamily": "LINUX"
  },
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
      "essential": true,
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "PORT", "value": "8000"},
        {"name": "AWS_REGION", "value": "${AWS_REGION}"},
        {"name": "AWS_DEFAULT_REGION", "value": "${AWS_REGION}"},
        {"name": "COGNITO_USER_POOL_ID", "value": "us-east-1_1G5jYI8FO"},
        {"name": "COGNITO_CLIENT_ID", "value": "66qdivmqgs1oqmmo0b5r9d9hjo"},
        {"name": "COGNITO_REGION", "value": "${AWS_REGION}"},
        {"name": "DYNAMODB_TABLE_NAME", "value": "clarity-health-data"},
        {"name": "S3_BUCKET_NAME", "value": "clarity-health-uploads"},
        {"name": "ENABLE_AUTH", "value": "true"},
        {"name": "DEBUG", "value": "false"},
        {"name": "SKIP_EXTERNAL_SERVICES", "value": "false"},
        {"name": "GEMINI_API_KEY", "value": "${GEMINI_API_KEY:-dummy_key_for_mock_mode}"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 10,
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
TASK_DEFINITION_ARN=$(aws ecs register-task-definition \
    --cli-input-json file://task-definition.json \
    --region ${AWS_REGION} \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

# Update service
echo "🔄 Updating ECS service..."
aws ecs update-service \
    --cluster ${CLUSTER_NAME} \
    --service ${SERVICE_NAME} \
    --task-definition ${TASK_DEFINITION_ARN} \
    --desired-count 1 \
    --region ${AWS_REGION}

# Wait for deployment to complete
echo "⏳ Waiting for deployment to complete..."
aws ecs wait services-stable \
    --cluster ${CLUSTER_NAME} \
    --services ${SERVICE_NAME} \
    --region ${AWS_REGION}

# Clean up
rm task-definition.json

echo ""
echo "✅ PROFESSIONAL DEPLOYMENT COMPLETE!"
echo "🌐 Load Balancer: clarity-alb-1762715656.us-east-1.elb.amazonaws.com"
echo "📊 Image: ${IMAGE_TAG}"
echo "🏥 Health: http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health"
echo "📖 Docs: http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/docs"
echo ""
echo "🎉 ENTERPRISE ML PLATFORM DEPLOYED!"

# Monitor service
echo "📊 Monitoring service status..."
aws ecs describe-services \
    --cluster ${CLUSTER_NAME} \
    --services ${SERVICE_NAME} \
    --region ${AWS_REGION} \
    --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount,Platform:platformVersion}' \
    --output table