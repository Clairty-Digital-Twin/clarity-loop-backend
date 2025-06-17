#!/bin/bash
# Centralized environment configuration for WAF deployment
# Source this file in all WAF-related scripts

export ALB_NAME="clarity-alb"
export REGION="us-east-1"
export ACCOUNT_ID="124355672559"
export WAF_NAME="clarity-backend-rate-limiting"

# Dynamic ALB ARN lookup
export ALB_ARN=$(aws elbv2 describe-load-balancers \
                   --names "$ALB_NAME" \
                   --query 'LoadBalancers[0].LoadBalancerArn' \
                   --output text --region "$REGION")

# Dynamic ALB DNS lookup  
export ALB_DNS=$(aws elbv2 describe-load-balancers \
                   --names "$ALB_NAME" \
                   --query 'LoadBalancers[0].DNSName' \
                   --output text --region "$REGION")

echo "🔧 Environment loaded:"
echo "   ALB Name: $ALB_NAME"
echo "   Region: $REGION"
echo "   ALB ARN: $ALB_ARN"
echo "   ALB DNS: $ALB_DNS" 