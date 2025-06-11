# CLARITY Digital Twin Platform

## 🚀 AWS-Native Health AI Platform

CLARITY is a revolutionary AI-powered mental health platform that uses wearable device data to provide personalized insights and interventions. Built with Clean Architecture principles and deployed on AWS infrastructure.

### ✅ Current Status: **Production Ready on AWS**

- **Backend API**: `https://your-domain.com/api` (configured via environment)
- **Architecture**: Clean Architecture with Dependency Injection
- **Infrastructure**: AWS ECS Fargate with Auto-scaling
- **AI Integration**: Google Gemini for health insights

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   iOS/Flutter   │────▶│   AWS ALB       │────▶│  ECS Fargate    │
│   Mobile App    │     │  Load Balancer  │     │   Containers    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                              ┌───────────────────────────┴────────────┐
                              │                                        │
                    ┌─────────▼─────────┐                   ┌─────────▼─────────┐
                    │   AWS Cognito     │                   │    DynamoDB       │
                    │  Authentication   │                   │   Health Data     │
                    └───────────────────┘                   └───────────────────┘
                                                                      │
                    ┌─────────────────┐                     ┌─────────▼─────────┐
                    │   Gemini API    │◀────────────────────│   FastAPI App     │
                    │   AI Insights   │                     │  Python Backend   │
                    └─────────────────┘                     └───────────────────┘
```

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/clarity-loop-backend.git
cd clarity-loop-backend

# Install dependencies
make install

# Copy AWS environment configuration
cp .env.aws .env

# Run locally with AWS services
make dev

# Run tests
make test
```

### Production Deployment

```bash
# Deploy to AWS ECS
./deploy-to-aws.sh

# Monitor deployment
./ops/monitor-deployment.sh
```

## 🔑 API Endpoints

### Health Check
```bash
curl https://your-domain.com/api/health
```

### Authentication
- `POST /api/v1/auth/signup` - User registration
- `POST /api/v1/auth/login` - User login
- `GET /api/v1/user/profile` - Get user profile

### Health Data
- `POST /api/v1/health-data` - Upload health metrics
- `GET /api/v1/health-data` - Query health history

### AI Insights
- `POST /api/v1/insights` - Generate personalized health insights

## 🛠️ Technology Stack

### Backend
- **Language**: Python 3.11
- **Framework**: FastAPI
- **Architecture**: Clean Architecture + Dependency Injection
- **API**: RESTful + WebSocket support

### AWS Services
- **Compute**: ECS Fargate (serverless containers)
- **Load Balancing**: Application Load Balancer
- **Authentication**: AWS Cognito
- **Database**: DynamoDB (NoSQL)
- **Storage**: S3 (file storage)
- **Messaging**: SQS/SNS (async processing)
- **Monitoring**: CloudWatch

### AI/ML
- **Insights**: Google Gemini API
- **Analysis**: Pretrained Actigraphy Transformer (PAT)
- **Processing**: Custom ML pipelines for health metrics

## 📋 Environment Variables

Create a `.env` file for local development:

```bash
# Environment
ENVIRONMENT=development
USE_AWS_SERVICES=true

# AWS Configuration
AWS_REGION=us-east-1
COGNITO_USER_POOL_ID=your-pool-id
COGNITO_CLIENT_ID=your-client-id

# Database
DYNAMODB_TABLE_PREFIX=clarity

# AI Services
GEMINI_API_KEY=your-gemini-key
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test categories
make test-unit        # Unit tests only
make test-integration # Integration tests
make test-ml         # ML pipeline tests

# Check code quality
make lint
make typecheck
```

## 🚀 Deployment

The application is deployed on AWS ECS Fargate with:

- Auto-scaling based on CPU/memory usage
- Health checks via ALB
- Rolling deployments with zero downtime
- CloudWatch logging and monitoring

### Manual Deployment

```bash
# Build and push Docker image
docker build -t clarity-backend .
docker tag clarity-backend:latest [your-ecr-registry]/clarity-backend:latest
docker push [your-ecr-registry]/clarity-backend:latest

# Update ECS service
aws ecs update-service --cluster [your-cluster] --service clarity-backend --force-new-deployment
```

## 📈 Monitoring

- **Health Dashboard**: CloudWatch metrics
- **Logs**: CloudWatch Logs for all containers
- **Alerts**: SNS notifications for critical issues
- **Tracing**: X-Ray for distributed tracing (coming soon)

## 🔒 Security

- HIPAA-compliant infrastructure
- End-to-end encryption
- JWT-based authentication
- Role-based access control
- Audit logging for all data access

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with ❤️ by the CLARITY team
- Powered by AWS and Google Gemini
- Special thanks to all contributors

---

**Note**: This is a production system handling sensitive health data. Please ensure all contributions maintain HIPAA compliance and security best practices.