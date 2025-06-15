# CLARITY Backend Operations

## 🚨 CRITICAL DEPLOYMENT REQUIREMENTS

**ALWAYS BUILD FOR linux/amd64 PLATFORM**
- AWS ECS Fargate ONLY supports `linux/amd64`
- Use: `docker buildx build --platform linux/amd64`
- Never use default platform on Apple Silicon Macs

## Quick Start Deployment

```bash
# Deploy existing image
./ops/deploy.sh

# Build and deploy new code
./ops/deploy.sh --build
```

## Infrastructure Configuration

### AWS Resources
- **ECR Repository**: `124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend`
- **ECS Cluster**: `clarity-backend-cluster`
- **ECS Service**: `clarity-backend-service`
- **Load Balancer**: https://clarity.novamindnyc.com
- **S3 ML Models**: `clarity-ml-models-124355672559`

### IAM Roles
- **Task Execution Role**: `ecsTaskExecutionRole` (with S3 read access)
- **Task Role**: `clarity-backend-task-role` (for runtime S3 access)

## Deployment Script Features

The `deploy.sh` script handles:
1. ✅ Platform validation (linux/amd64)
2. ✅ Git commit-based tagging
3. ✅ ECR image verification
4. ✅ Task definition with proper IAM roles
5. ✅ Service update with health checks
6. ✅ Smoke test validation

## Common Issues & Solutions

### Task Fails with Exit Code 1
**Cause**: Missing S3 permissions for ML models
**Fix**: Ensure task role has `AmazonS3ReadOnlyAccess`

### Task Stuck in PENDING
**Cause**: Platform mismatch or networking issues
**Fix**: Verify image is built for linux/amd64

### Build Hangs on Apple Silicon
**Fix**: Ensure Docker Desktop has 32GB RAM allocated

## Monitoring

```bash
# Check service status
aws ecs describe-services \
  --cluster clarity-backend-cluster \
  --services clarity-backend-service \
  --region us-east-1

# View logs
aws logs tail /ecs/clarity-backend --follow --region us-east-1

# Check running tasks
aws ecs list-tasks \
  --cluster clarity-backend-cluster \
  --service-name clarity-backend-service \
  --region us-east-1
```

## Emergency Rollback

```bash
# Get previous task definition
PREV_TD=$(aws ecs describe-service \
  --cluster clarity-backend-cluster \
  --service clarity-backend-service \
  --query 'service.taskDefinition' \
  --output text | awk -F: '{print $NF-1}')

# Rollback
aws ecs update-service \
  --cluster clarity-backend-cluster \
  --service clarity-backend-service \
  --task-definition clarity-backend:$PREV_TD \
  --force-new-deployment
```