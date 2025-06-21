# CLARITY Deployment Guide

## Quick Deploy (Manual)

For quick manual deployments:

```bash
./deploy-quick.sh
```

This will:
1. Build the Docker image for linux/amd64
2. Push to ECR with git commit SHA tag
3. Deploy to ECS using the deploy.sh script

## Automated Deployment (CI/CD)

Deployments happen automatically when you push to `main`:

```bash
git add .
git commit -m "Your changes"
git push origin main
```

GitHub Actions will:
1. Build optimized image with caching
2. Push to ECR
3. Update ECS service
4. Wait for health checks

## Deployment Architecture

```
GitHub (push to main)
    ↓
GitHub Actions (.github/workflows/deploy.yml)
    ↓
Docker Build (linux/amd64)
    ↓
ECR Push (with cache layers)
    ↓
ECS Task Definition Update
    ↓
Service Rolling Update
    ↓
Health Checks Pass
    ↓
✅ Live on AWS
```

## Manual Deployment Steps

If you need more control:

```bash
# 1. Build image
docker build --platform linux/amd64 -t clarity-backend:latest .

# 2. Tag for ECR
docker tag clarity-backend:latest 124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend:latest

# 3. Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 124355672559.dkr.ecr.us-east-1.amazonaws.com

# 4. Push image
docker push 124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend:latest

# 5. Deploy
ops/deploy.sh
```

## Rollback

To rollback to a previous version:

```bash
# List previous task definitions
aws ecs list-task-definitions --family clarity-backend --sort DESC --max-items 10

# Update service to previous version
aws ecs update-service \
  --cluster clarity-backend-cluster \
  --service clarity-backend-service \
  --task-definition clarity-backend:119  # Previous version number
```

## Monitoring

Check deployment status:

```bash
# Service status
aws ecs describe-services --cluster clarity-backend-cluster --services clarity-backend-service

# Recent logs
aws logs tail /ecs/clarity-backend --since 5m --follow

# Health endpoint
curl https://api.clarity.health/health
```

## Image Size Note

Current image is ~3.5GB due to ML dependencies:
- PyTorch: 1.6GB
- NVIDIA CUDA: 2.7GB
- Other ML libs: ~500MB

To reduce size in the future:
1. Use CPU-only PyTorch: `torch==2.7.1+cpu`
2. Create separate ML service
3. Use model servers like TorchServe

## Troubleshooting

### Build Failures
- Check Docker daemon is running
- Ensure you're on stable internet (large dependencies)
- Try with `--no-cache` flag

### Push Failures
- Refresh ECR credentials: `aws ecr get-login-password`
- Check AWS credentials are valid
- Verify ECR repository exists

### Deployment Failures
- Check ECS task logs in CloudWatch
- Verify secrets in Secrets Manager
- Check IAM permissions for task role
- Ensure health checks are passing

### Common Issues

1. **"IMAGE_PLACEHOLDER" error**
   - The deploy.sh script handles this automatically
   - If manual, ensure task definition has correct image

2. **Secret not found**
   - Check secret ARNs in task definition match Secrets Manager
   - Verify task execution role has `secretsmanager:GetSecretValue`

3. **Health check failures**
   - Check application logs for startup errors
   - Verify security groups allow health check traffic
   - Ensure app responds on `/health` endpoint