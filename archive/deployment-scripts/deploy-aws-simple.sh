#!/bin/bash
# Deploy the simplified AWS backend to ECS

set -e

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
ECR_REGISTRY=${ECR_REGISTRY:-"***REMOVED***"}
IMAGE_NAME="clarity-backend-simple"
ECS_CLUSTER=${ECS_CLUSTER:-"***REMOVED***"}
ECS_SERVICE=${ECS_SERVICE:-"clarity-backend-simple"}
TASK_FAMILY=${TASK_FAMILY:-"clarity-backend-simple"}

echo "🚀 Starting deployment of simplified AWS backend..."

# Build the Docker image
echo "📦 Building Docker image..."
docker build -f Dockerfile.aws.simple -t ${IMAGE_NAME}:latest .

# Tag for ECR
docker tag ${IMAGE_NAME}:latest ${ECR_REGISTRY}/${IMAGE_NAME}:latest

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Push to ECR
echo "⬆️  Pushing image to ECR..."
docker push ${ECR_REGISTRY}/${IMAGE_NAME}:latest

# Create task definition
echo "📋 Creating ECS task definition..."
cat > ecs-task-definition-simple.json <<EOF
{
  "family": "${TASK_FAMILY}",
  "taskRoleArn": "arn:aws:iam::124355672559:role/clarity-ecs-task-role",
  "executionRoleArn": "arn:aws:iam::124355672559:role/clarity-ecs-execution-role",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "clarity-backend-simple",
      "image": "${ECR_REGISTRY}/${IMAGE_NAME}:latest",
      "essential": true,
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
          "name": "AWS_REGION",
          "value": "${AWS_REGION}"
        },
        {
          "name": "DYNAMODB_TABLE",
          "value": "clarity-health-data"
        },
        {
          "name": "CLARITY_API_KEY",
          "value": "production-api-key-change-me"
        }
      ],
      "secrets": [
        {
          "name": "GEMINI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:${AWS_REGION}:124355672559:secret:clarity/gemini-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/clarity-backend-simple",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF

# Register task definition
echo "📝 Registering task definition..."
aws ecs register-task-definition \
    --cli-input-json file://ecs-task-definition-simple.json \
    --region ${AWS_REGION}

# Check if service exists
if aws ecs describe-services --cluster ${ECS_CLUSTER} --services ${ECS_SERVICE} --region ${AWS_REGION} | grep -q "ACTIVE"; then
    echo "🔄 Updating existing ECS service..."
    aws ecs update-service \
        --cluster ${ECS_CLUSTER} \
        --service ${ECS_SERVICE} \
        --task-definition ${TASK_FAMILY} \
        --force-new-deployment \
        --region ${AWS_REGION}
else
    echo "✨ Creating new ECS service..."
    aws ecs create-service \
        --cluster ${ECS_CLUSTER} \
        --service-name ${ECS_SERVICE} \
        --task-definition ${TASK_FAMILY} \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[subnet-0f5578435b4b48bf2,subnet-09e851182f425a48e],securityGroups=[sg-07ece5885524dfd3b],assignPublicIp=ENABLED}" \
        --region ${AWS_REGION}
fi

echo "✅ Deployment initiated!"
echo ""
echo "Monitor deployment progress:"
echo "aws ecs describe-services --cluster ${ECS_CLUSTER} --services ${ECS_SERVICE} --region ${AWS_REGION}"
echo ""
echo "View logs:"
echo "aws logs tail /ecs/clarity-backend-simple --follow --region ${AWS_REGION}"