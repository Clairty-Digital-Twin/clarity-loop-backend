#!/bin/bash
# Set up Application Load Balancer for production

set -e

echo "🚀 Setting up Application Load Balancer..."

# Create ALB
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name clarity-alb \
  --subnets subnet-0f5578435b4b48bf2 subnet-09e851182f425a48e \
  --security-groups sg-07ece5885524dfd3b \
  --region us-east-1 \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text)

echo "✅ ALB created: $ALB_ARN"

# Create target group
TG_ARN=$(aws elbv2 create-target-group \
  --name clarity-targets \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-077372d97e956ac82 \
  --target-type ip \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --health-check-timeout-seconds 5 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3 \
  --region us-east-1 \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text)

echo "✅ Target group created: $TG_ARN"

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN \
  --region us-east-1

echo "✅ Listener created"

# Update ECS service to use ALB
aws ecs update-service \
  --cluster ***REMOVED*** \
  --service clarity-backend-simple \
  --load-balancers targetGroupArn=$TG_ARN,containerName=clarity-backend-simple,containerPort=8000 \
  --region us-east-1

echo "✅ ECS service updated to use ALB"

# Get ALB DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --load-balancer-arns $ALB_ARN \
  --region us-east-1 \
  --query 'LoadBalancers[0].DNSName' \
  --output text)

echo ""
echo "🎉 ALB setup complete!"
echo "ALB URL: http://$ALB_DNS"
echo ""
echo "Note: It may take a few minutes for the ALB to become healthy"