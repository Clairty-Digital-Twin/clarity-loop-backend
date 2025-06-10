#!/bin/bash
# Update ECS to use full backend with Cognito

set -e

echo "🚀 Updating ECS to full backend with Cognito..."

# Variables
ECR_REGISTRY="***REMOVED***"
IMAGE_NAME="clarity-backend-full"
TASK_FAMILY="clarity-backend-full"
ECS_CLUSTER="***REMOVED***"
ECS_SERVICE="clarity-backend-simple"

# Tag and push the new image
echo "📦 Pushing full backend image..."
docker tag clarity-backend-full:latest ${ECR_REGISTRY}/${IMAGE_NAME}:latest
docker push ${ECR_REGISTRY}/${IMAGE_NAME}:latest

# Create new task definition with Cognito config
echo "📋 Creating new task definition..."
cat > ecs-task-definition-full.json <<EOF
{
  "family": "${TASK_FAMILY}",
  "taskRoleArn": "arn:aws:iam::124355672559:role/clarity-ecs-task-role",
  "executionRoleArn": "arn:aws:iam::124355672559:role/clarity-ecs-execution-role",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "clarity-backend-full",
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
          "value": "us-east-1"
        },
        {
          "name": "DYNAMODB_TABLE",
          "value": "clarity-health-data"
        },
        {
          "name": "CLARITY_API_KEY",
          "value": "production-api-key-change-me"
        },
        {
          "name": "COGNITO_USER_POOL_ID",
          "value": "us-east-1_efXaR5EcP"
        },
        {
          "name": "COGNITO_CLIENT_ID",
          "value": "7sm7ckrkovg78b03n1595euc71"
        }
      ],
      "secrets": [
        {
          "name": "GEMINI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:124355672559:secret:clarity/gemini-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/clarity-backend-full",
          "awslogs-region": "us-east-1",
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

# Create log group
aws logs create-log-group --log-group-name /ecs/clarity-backend-full --region us-east-1 2>/dev/null || echo "Log group exists"

# Register task definition
TASK_DEF_ARN=$(aws ecs register-task-definition \
    --cli-input-json file://ecs-task-definition-full.json \
    --region us-east-1 \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

echo "✅ Task definition created: $TASK_DEF_ARN"

# Update service
echo "🔄 Updating ECS service..."
aws ecs update-service \
    --cluster ${ECS_CLUSTER} \
    --service ${ECS_SERVICE} \
    --task-definition ${TASK_FAMILY} \
    --force-new-deployment \
    --region us-east-1

echo "✅ Service update initiated!"
echo ""
echo "Monitor deployment:"
echo "aws ecs describe-services --cluster ${ECS_CLUSTER} --services ${ECS_SERVICE} --region us-east-1"