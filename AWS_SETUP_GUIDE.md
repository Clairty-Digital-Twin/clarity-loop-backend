# AWS Setup Guide for CLARITY Backend

## Prerequisites

1. AWS Account
2. AWS CLI installed (✓ Already installed)
3. Docker running locally (✓ Verified)

## Step 1: Configure AWS CLI

Run the following command and provide your credentials:

```bash
aws configure
```

Enter:
- AWS Access Key ID: [Your key from IAM]
- AWS Secret Access Key: [Your secret]
- Default region name: us-east-1 (or your preferred region)
- Default output format: json

## Step 2: Create ECR Repository

Once AWS is configured, we'll create an ECR repository for our Docker images:

```bash
# Create ECR repository
aws ecr create-repository --repository-name clarity-backend --region us-east-1

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com
```

## Step 3: Push Docker Image

```bash
# Tag the image
docker tag clarity-backend:aws-test [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/clarity-backend:latest

# Push to ECR
docker push [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/clarity-backend:latest
```

## Step 4: Create ECS Task Definition

We'll create a task definition file `ecs-task-definition.json` that specifies:
- Container configuration
- Resource allocation (CPU/Memory)
- Environment variables
- IAM roles

## Step 5: Create ECS Service

Using AWS Fargate for serverless container hosting:
- Create ECS cluster
- Create service with auto-scaling
- Configure load balancer

## Step 6: Set Up Networking

- Create VPC (or use default)
- Configure security groups
- Set up Application Load Balancer

## Required IAM Permissions

Your AWS user needs these permissions:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonECS_FullAccess`
- `IAMFullAccess`
- `AmazonVPCFullAccess`
- `SecretsManagerReadWrite`
- `AmazonDynamoDBFullAccess`
- `AmazonS3FullAccess`

## Environment Variables for Production

These will be stored in AWS Secrets Manager:
- Database connections
- API keys
- Authentication secrets
- ML model locations

## Next Steps

1. Configure AWS CLI with your credentials
2. Run the deployment script (to be created)
3. Monitor deployment in AWS Console
4. Set up CloudWatch alarms

## Cost Optimization Tips

- Use Fargate Spot for non-critical workloads (70% savings)
- Enable auto-scaling based on CPU/Memory metrics
- Use S3 lifecycle policies for log rotation
- Consider Reserved Capacity for predictable workloads