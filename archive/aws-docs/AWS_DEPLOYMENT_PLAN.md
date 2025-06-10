# AWS Deployment Plan - CLARITY Backend

## Overview

This plan outlines the migration from Google Cloud Platform to Amazon Web Services, focusing on containerized deployment using ECS Fargate with proper handling of ML workloads.

## Phase 1: Local Development & Testing

### 1.1 Fix Package Structure
The current issue with `clarity.models` not being found is due to circular imports during package initialization. We need to fix this first.

### 1.2 Create AWS-Compatible Dockerfile
- Multi-stage build for optimization
- Proper Python package installation
- Health check endpoints

### 1.3 Local Testing
- Test with docker-compose
- Verify all imports work correctly
- Run basic health checks

## Phase 2: AWS Infrastructure Setup

### 2.1 Core Services
- **ECS Fargate**: Serverless container hosting
- **ECR**: Container registry
- **ALB**: Application Load Balancer
- **Route 53**: DNS management

### 2.2 Data Services
- **DynamoDB**: NoSQL database (replacing Firestore)
- **S3**: Object storage (replacing Cloud Storage)
- **SQS/SNS**: Message queuing (replacing Pub/Sub)

### 2.3 Security & Configuration
- **Cognito**: User authentication (replacing Firebase)
- **Secrets Manager**: Sensitive configuration
- **IAM**: Role-based access control
- **VPC**: Network isolation

### 2.4 ML Services
- **EC2 GPU Instances**: For transformer models
- **SageMaker**: Model serving
- **Batch**: Batch processing jobs

## Phase 3: Code Migration

### 3.1 Create Abstraction Layer
Instead of directly replacing GCP services, create an abstraction layer:

```python
# ports/cloud_provider.py
class ICloudProvider(Protocol):
    async def get_secret(self, name: str) -> str: ...
    async def upload_file(self, bucket: str, key: str, data: bytes) -> str: ...
    # etc.

# aws/provider.py
class AWSProvider(ICloudProvider):
    # AWS implementations using boto3
```

### 3.2 Service Replacements
1. **Storage Layer**
   - Create DynamoDB repository implementing IHealthDataRepository
   - S3 client for file storage

2. **Authentication**
   - Cognito provider implementing IAuthProvider
   - JWT token validation

3. **Messaging**
   - SQS/SNS implementation for pub/sub patterns

## Phase 4: Deployment Configuration

### 4.1 Docker Configuration
- Optimized multi-stage Dockerfile
- .dockerignore for build efficiency
- Health check implementation

### 4.2 ECS Task Definition
- Container specifications
- Resource allocation (CPU/Memory)
- Environment variables
- IAM task role

### 4.3 Infrastructure as Code
- Use AWS CDK or Terraform
- Define all resources programmatically
- Version control infrastructure

## Phase 5: CI/CD Pipeline

### 5.1 GitHub Actions / AWS CodePipeline
- Build on push to main
- Run tests
- Build and push to ECR
- Deploy to ECS

### 5.2 Monitoring & Logging
- CloudWatch logs
- X-Ray for tracing
- CloudWatch metrics
- Alarms and notifications

## Immediate Next Steps

1. **Install AWS CLI**
   ```bash
   brew install awscli
   ```

2. **Fix the package import issue**
   - Modify models/__init__.py to use lazy imports
   - Create proper Dockerfile

3. **Create local docker-compose for testing**

4. **Set up AWS account and credentials**

5. **Create ECR repository**

6. **Deploy to ECS Fargate**

## Cost Optimization

- Use Fargate Spot for non-critical workloads
- Implement auto-scaling based on metrics
- Use S3 lifecycle policies
- Reserved capacity for predictable workloads

## Security Considerations

- All secrets in AWS Secrets Manager
- VPC with private subnets for containers
- Security groups with minimal access
- Enable AWS GuardDuty
- Use AWS WAF for API protection