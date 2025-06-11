#!/bin/bash
set -e

# Ultra clean deployment - NO Firebase dependencies

AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="124355672559"
ECR_REPOSITORY="clarity-backend"
IMAGE_TAG="ultra-clean-v1"
CLUSTER_NAME="clarity-backend-cluster"
SERVICE_NAME="clarity-backend-service"

echo "🚀 ULTRA CLEAN DEPLOYMENT - 35+ endpoints, NO Firebase!"
echo "📋 Using main_aws_ultra_clean.py"

# Quick build and push
echo "📦 Building ultra clean image..."
docker build -f Dockerfile.aws.clean -t ${ECR_REPOSITORY}:${IMAGE_TAG} --platform linux/amd64 .

echo "🏷️ Tagging for ECR..."
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

echo "⬆️ Pushing to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

echo "📝 Updating ECS..."
# Create minimal task definition
cat > /tmp/ultra-task-def.json <<EOF
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
      {"name": "AWS_REGION", "value": "us-east-2"},
      {"name": "COGNITO_USER_POOL_ID", "value": "us-east-2_iCRM83uVj"},
      {"name": "COGNITO_CLIENT_ID", "value": "485gn7vn3uev0coc52aefklkjs"},
      {"name": "COGNITO_REGION", "value": "us-east-2"},
      {"name": "DYNAMODB_TABLE_NAME", "value": "clarity-health-data"},
      {"name": "S3_BUCKET_NAME", "value": "clarity-health-uploads"},
      {"name": "ENABLE_AUTH", "value": "false"},
      {"name": "CLARITY_MAIN_MODULE", "value": "clarity.main_aws_ultra_clean:app"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/clarity-backend",
        "awslogs-region": "us-east-2",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
EOF

# Register and deploy
TASK_ARN=$(aws ecs register-task-definition \
  --cli-input-json file:///tmp/ultra-task-def.json \
  --region ${AWS_REGION} \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

aws ecs update-service \
  --cluster ${CLUSTER_NAME} \
  --service ${SERVICE_NAME} \
  --task-definition ${TASK_ARN} \
  --force-new-deployment \
  --region ${AWS_REGION}

echo "✨ ULTRA CLEAN deployment complete!"
echo "🔍 Expected: 35+ endpoints, service: clarity-backend-aws-nuclear"
echo "🌐 http://***REMOVED***/health"