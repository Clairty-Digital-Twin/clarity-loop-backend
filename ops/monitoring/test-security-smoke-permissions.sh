#!/bin/bash

# Test Security Smoke Test Permissions
# This script simulates the Security Smoke Test to verify permissions are working

set -e

ALB_NAME="clarity-alb"
EXPECTED_WAF_NAME="clarity-backend-rate-limiting"
REGION="us-east-1"

echo "🧪 Testing Security Smoke Test Permissions..."
echo "📋 ALB: $ALB_NAME"
echo "📋 Expected WAF: $EXPECTED_WAF_NAME"
echo "📋 Region: $REGION"
echo ""

# Test 1: Describe Load Balancer
echo "1️⃣ Testing elasticloadbalancing:DescribeLoadBalancers..."
if ALB_ARN=$(aws elbv2 describe-load-balancers \
    --names $ALB_NAME \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text --region $REGION 2>/dev/null); then
    echo "✅ Successfully described load balancer"
    echo "   ARN: $ALB_ARN"
else
    echo "❌ Failed to describe load balancer"
    echo "   Check if the GitHubActionsDeploy role has elasticloadbalancing:DescribeLoadBalancers permission"
    exit 1
fi

# Test 2: Get WAF for Resource
echo ""
echo "2️⃣ Testing wafv2:GetWebACLForResource..."
if WAF_NAME=$(aws wafv2 get-web-acl-for-resource \
    --resource-arn "$ALB_ARN" \
    --region $REGION \
    --query 'WebACL.Name' \
    --output text 2>/dev/null); then
    echo "✅ Successfully retrieved WAF ACL"
    echo "   WAF Name: $WAF_NAME"
else
    echo "❌ Failed to retrieve WAF ACL"
    echo "   Check if the GitHubActionsDeploy role has wafv2:GetWebACLForResource permission"
    exit 1
fi

# Test 3: Verify WAF Name
echo ""
echo "3️⃣ Testing WAF configuration..."
if [ "$WAF_NAME" = "$EXPECTED_WAF_NAME" ]; then
    echo "✅ WAF is correctly configured"
    echo "   Expected: $EXPECTED_WAF_NAME"
    echo "   Actual: $WAF_NAME"
else
    echo "⚠️  WAF name mismatch"
    echo "   Expected: $EXPECTED_WAF_NAME"
    echo "   Actual: $WAF_NAME"
    echo "   This may indicate a configuration issue"
fi

# Test 4: Test SQL Injection Blocking (optional)
echo ""
echo "4️⃣ Testing SQL injection blocking..."
if ALB_DNS=$(aws elbv2 describe-load-balancers \
    --names $ALB_NAME \
    --query 'LoadBalancers[0].DNSName' \
    --output text --region $REGION 2>/dev/null); then
    echo "   ALB DNS: $ALB_DNS"
    
    # Test SQL injection blocking
    if RESPONSE=$(curl -s -k -o /dev/null -w "%{http_code}" \
        "https://$ALB_DNS/health?id=1%27%20OR%201=1--" 2>/dev/null); then
        if [ "$RESPONSE" = "403" ]; then
            echo "✅ SQL injection correctly blocked (HTTP 403)"
        else
            echo "⚠️  SQL injection not blocked (HTTP $RESPONSE)"
            echo "   This may indicate a WAF configuration issue"
        fi
    else
        echo "ℹ️  Could not test SQL injection blocking (ALB may not be accessible)"
    fi
else
    echo "ℹ️  Could not retrieve ALB DNS name"
fi

echo ""
echo "🎉 Security Smoke Test permissions verified!"
echo ""
echo "✅ All required permissions are working:"
echo "   • elasticloadbalancing:DescribeLoadBalancers"
echo "   • wafv2:GetWebACLForResource"
echo ""
echo "🚀 The GitHub Actions Security Smoke Test should now pass!" 