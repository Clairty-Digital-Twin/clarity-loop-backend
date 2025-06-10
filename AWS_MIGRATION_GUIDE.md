# AWS Migration Guide - Clarity Backend

## Overview

This guide documents the migration from Google Cloud Platform (Firebase/Firestore) to AWS services. The migration involves replacing:

- **Firebase Authentication** → AWS Cognito or API Keys
- **Firestore** → DynamoDB
- **Google Cloud Pub/Sub** → AWS SQS/SNS (removed for MVP)
- **Vertex AI** → Direct Gemini API calls
- **Cloud Storage** → S3 (removed for MVP)

## Current Status

### ✅ Completed
1. Created simplified AWS-compatible backend (`main_aws_simple.py`)
2. Removed all Firebase/GCP dependencies
3. Implemented 3 working endpoints:
   - `GET /health` - Health check
   - `POST /api/v1/data` - Store health data in DynamoDB
   - `POST /api/v1/insights` - Generate insights using Gemini API
   - `GET /api/v1/data/{user_id}` - Retrieve user data
4. Simple API key authentication
5. Docker configuration for AWS deployment
6. Local development setup with DynamoDB Local

### 🚧 TODO
1. Set up AWS infrastructure (DynamoDB tables, VPC, ECS cluster)
2. Configure AWS Secrets Manager for API keys
3. Implement AWS Cognito for production authentication
4. Add S3 for file storage (when needed)
5. Re-implement Pub/Sub functionality with SQS/SNS (when needed)
6. Migrate existing Firebase data to DynamoDB

## Quick Start

### Local Development

1. **Set up environment variables:**
```bash
cp .env.aws .env
# Edit .env and add your GEMINI_API_KEY
```

2. **Run with Docker Compose (includes local DynamoDB):**
```bash
docker-compose -f docker-compose.aws.yml up
```

3. **Test the endpoints:**
```bash
python test_aws_simple.py
```

### AWS Deployment

1. **Prerequisites:**
   - AWS CLI configured
   - ECR repository created
   - ECS cluster set up
   - DynamoDB table created
   - Secrets Manager configured with GEMINI_API_KEY

2. **Deploy to ECS:**
```bash
./deploy-aws-simple.sh
```

## API Documentation

### Authentication
All endpoints (except `/health`) require an API key in the `X-API-Key` header.

### Endpoints

#### Health Check
```bash
GET /health
```

#### Store Health Data
```bash
POST /api/v1/data
Headers: X-API-Key: your-api-key
Body:
{
  "user_id": "user123",
  "data_type": "heart_rate",
  "value": 72.5,
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {
    "device": "apple_watch"
  }
}
```

#### Get User Data
```bash
GET /api/v1/data/{user_id}?limit=50
Headers: X-API-Key: your-api-key
```

#### Generate Insights
```bash
POST /api/v1/insights
Headers: X-API-Key: your-api-key
Body:
{
  "user_id": "user123",
  "query": "How can I improve my sleep?",
  "include_recent_data": true
}
```

## Architecture Changes

### Before (GCP)
```
Client → Firebase Auth → FastAPI → Firestore
                                 → Vertex AI
                                 → Pub/Sub
```

### After (AWS)
```
Client → API Key Auth → FastAPI → DynamoDB
                               → Gemini API (direct)
```

### Removed Features (for MVP)
- Real-time WebSocket connections
- Complex ML pipelines (PAT, Fusion Transformer)
- Pub/Sub message queuing
- File upload/storage
- User management

## Migration Steps

### Phase 1: Basic Functionality ✅
- Create simplified backend with core endpoints
- Replace Firebase Auth with API keys
- Replace Firestore with DynamoDB
- Keep Gemini API integration

### Phase 2: AWS Infrastructure (Current)
- Set up production DynamoDB tables
- Configure VPC and security groups
- Deploy to ECS Fargate
- Set up Application Load Balancer

### Phase 3: Enhanced Features
- Implement AWS Cognito authentication
- Add S3 for file storage
- Implement SQS/SNS for async processing
- Re-enable ML pipelines

### Phase 4: Data Migration
- Export existing Firestore data
- Transform data format for DynamoDB
- Import historical data
- Verify data integrity

## Configuration

### Environment Variables
```bash
# Core Settings
ENVIRONMENT=development|production
AWS_REGION=us-east-1
DYNAMODB_TABLE=clarity-health-data

# Authentication
CLARITY_API_KEY=your-api-key

# Gemini API
GEMINI_API_KEY=your-gemini-key

# Local Development
DYNAMODB_ENDPOINT=http://localhost:8000  # For DynamoDB Local
```

### AWS Resources Needed
1. **DynamoDB Table**: `clarity-health-data`
   - Partition Key: `user_id` (String)
   - Sort Key: `data_id` (String)
   - Billing Mode: Pay-per-request

2. **ECS Fargate**:
   - Task Definition: 256 CPU, 512 Memory
   - Service: 1 task minimum
   - Auto-scaling: Based on CPU/Memory

3. **Secrets Manager**:
   - `clarity/gemini-api-key`
   - `clarity/api-keys` (for production)

4. **IAM Roles**:
   - ECS Task Role (DynamoDB access)
   - ECS Execution Role (Secrets Manager, CloudWatch)

## Troubleshooting

### Common Issues

1. **DynamoDB Connection Error**
   - Check AWS credentials
   - Verify table exists
   - Check IAM permissions

2. **Gemini API Error**
   - Verify GEMINI_API_KEY is set
   - Check API quota limits
   - Ensure proper error handling

3. **ECS Deployment Failure**
   - Check CloudWatch logs
   - Verify security group rules
   - Ensure health check passes

## Next Steps

1. **Production Readiness**:
   - Set up monitoring (CloudWatch)
   - Configure auto-scaling
   - Implement rate limiting
   - Add comprehensive logging

2. **Security Enhancements**:
   - Rotate API keys regularly
   - Implement AWS Cognito
   - Enable VPC endpoints
   - Configure WAF rules

3. **Performance Optimization**:
   - Enable DynamoDB caching
   - Optimize container size
   - Implement connection pooling
   - Add CloudFront CDN

## Cost Estimation

### Monthly Costs (Estimated)
- ECS Fargate (1 task): ~$10
- DynamoDB (pay-per-request): ~$5-50
- Application Load Balancer: ~$20
- Data Transfer: ~$5-20
- **Total**: ~$40-100/month

## Rollback Plan

If issues arise:
1. Keep original GCP version running
2. Use Route53 for gradual traffic shift
3. Monitor error rates and performance
4. Quick rollback via DNS change