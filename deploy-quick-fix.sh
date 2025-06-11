#!/bin/bash
set -e

# Quick fix deployment - use working simple version and expand it

echo "🚀 QUICK FIX: Deploying working version with expanded endpoints..."

# Update the ECS task definition to use main_aws_simple which we KNOW works
aws ecs update-service \
  --cluster clarity-backend-cluster \
  --service clarity-backend-service \
  --task-definition clarity-backend:12 \
  --desired-count 1 \
  --region us-east-1 \
  --environment-overrides '[
    {
      "name": "clarity-backend",
      "environment": [
        {"name": "CLARITY_MAIN_MODULE", "value": "clarity.main_aws_simple:app"}
      ]
    }
  ]'

echo "✅ Deployment updated to use working simple version"
echo ""
echo "📊 Check status:"
echo "aws ecs describe-services --cluster clarity-backend-cluster --services clarity-backend-service --region us-east-1"
echo ""
echo "🌐 Test URL: http://***REMOVED***/health"