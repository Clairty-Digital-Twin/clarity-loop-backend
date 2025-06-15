# CLARITY Backend Deployment Guide

## 🚨 CRITICAL: Platform Requirements

**ALWAYS BUILD DOCKER IMAGES FOR linux/amd64 PLATFORM FOR AWS ECS DEPLOYMENT**

AWS ECS Fargate ONLY supports `linux/amd64` platform. Building for any other platform will cause deployment failures.

## Current Production Configuration

- **ECS Cluster**: `clarity-backend-cluster`
- **Service**: `clarity-backend-service`
- **ALB URL**: https://clarity-alb-1762715656.us-east-1.elb.amazonaws.com
- **Health Check**: https://clarity.novamindnyc.com/health
- **ECR Repository**: `124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend`
- **Latest Image**: `production-final` (also tagged as `latest`)

## Deployment Process

### Quick Deploy (Existing Image)

To deploy the current production image:

```bash
cd ops/
./deploy.sh
```

### Build and Deploy (New Code)

To build a new image and deploy:

```bash
cd ops/
./deploy.sh --build
```

This will:
1. Build a new Docker image (linux/amd64 platform)
2. Tag with timestamp (e.g., `v20240615-093000`)
3. Push to ECR
4. Update ECS task definition
5. Deploy to ECS service
6. Wait for deployment to complete
7. Verify health endpoint

## Monitoring Deployment

Check deployment status:
```bash
aws ecs describe-services \
    --cluster clarity-backend-cluster \
    --services clarity-backend-service \
    --region us-east-1 \
    --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}' \
    --output table
```

Check health:
```bash
curl -s https://clarity.novamindnyc.com/health | jq .
```

## Rollback

If deployment fails, ECS will automatically rollback to the previous working version.

To manually rollback:
```bash
aws ecs update-service \
    --cluster clarity-backend-cluster \
    --service clarity-backend-service \
    --task-definition clarity-backend:42 \
    --region us-east-1
```

## Resource Management

### ECR Lifecycle Policy
- Automatically keeps only the last 3 images
- Older images are expired after 1 day

### CloudWatch Logs
- Log group: `/ecs/clarity-backend`
- Retention: 30 days (configured separately)

## Cost Optimization

Current monthly costs:
- Fargate Task (4 vCPU, 16GB): ~$120/month
- ALB: ~$20/month
- ECR Storage: <$1/month
- CloudWatch Logs: ~$5/month
- **Total**: ~$146/month

## Troubleshooting

### Task Failed to Start
Usually caused by:
1. Wrong Docker platform (must be linux/amd64)
2. Secret retrieval errors (check IAM permissions)
3. Image not found in ECR

Check task failure:
```bash
TASK_ARN=$(aws ecs list-tasks --cluster clarity-backend-cluster --service-name clarity-backend-service --desired-status STOPPED --region us-east-1 --query 'taskArns[0]' --output text)
aws ecs describe-tasks --cluster clarity-backend-cluster --tasks $TASK_ARN --region us-east-1
```

### Secrets Access
The task execution role must have access to Secrets Manager:
- Role: `ecsTaskExecutionRole`
- Required policy: Access to `secretsmanager:GetSecretValue` for `arn:aws:secretsmanager:us-east-1:124355672559:secret:clarity/*`

## Important Notes

1. **NEVER** forget the `--platform linux/amd64` flag when building Docker images
2. **ALWAYS** verify health endpoint after deployment
3. **ALWAYS** use the `ops/deploy.sh` script for consistency
4. Secrets in Secrets Manager must be in JSON format, not plain text

## Contact

For deployment issues, check:
1. CloudWatch Logs: `/ecs/clarity-backend`
2. ECS Task failure reasons
3. ALB Target Group health checks