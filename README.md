# CLARITY Digital Twin Backend

A FastAPI-based health data processing platform that analyzes Apple Health data using AI models to generate health insights.

## Overview

CLARITY processes wearable device data through machine learning models to extract health patterns and generate natural language insights. The system uses a Pretrained Actigraphy Transformer (PAT) for movement analysis and Google Gemini for generating conversational health explanations.

**Tech Stack**: Python 3.11+, FastAPI, AWS (DynamoDB/Cognito/ECS), PyTorch, Transformers

## Architecture

```
Apple Health Data → FastAPI → PAT Analysis → Gemini AI → Health Insights
                     ↓
                AWS DynamoDB Storage
```

**Core Components**:
- **Data Ingestion**: HealthKit JSON processing and validation
- **ML Pipeline**: PAT transformer for temporal pattern analysis  
- **AI Integration**: Gemini for natural language insight generation
- **Real-time API**: WebSocket support for live data streaming
- **AWS Infrastructure**: Production-ready cloud deployment

## Quick Start

```bash
# Setup
git clone https://github.com/your-org/clarity-loop-backend.git
cd clarity-loop-backend
make install

# Configure environment
cp .env.example .env  # Add your API keys

# Run locally
make dev

# Verify
curl http://localhost:8000/health
```

## API Overview

**44 total endpoints** across 7 main areas:

- **Authentication** (`/api/v1/auth/*`) - 7 endpoints, AWS Cognito integration
- **Health Data** (`/api/v1/health-data/*`) - 10 endpoints, data CRUD operations
- **HealthKit** (`/api/v1/healthkit/*`) - 4 endpoints, Apple integration
- **AI Insights** (`/api/v1/insights/*`) - 6 endpoints, Gemini analysis
- **PAT Analysis** (`/api/v1/pat/*`) - 5 endpoints, transformer inference
- **WebSocket** (`/api/v1/ws/*`) - 3 endpoints, real-time connections  
- **System** (`/health`, `/metrics`) - 9 endpoints, monitoring

### Example Usage

```bash
# Upload health data
POST /api/v1/healthkit/upload
{
  "data": [/* HealthKit JSON export */]
}

# Get AI analysis
POST /api/v1/insights/generate
{
  "user_id": "123",
  "type": "sleep_analysis"
}
```

## Data Processing Pipeline

1. **HealthKit Ingestion**: JSON validation and schema conversion
2. **Temporal Processing**: 1-minute intervals, outlier detection
3. **PAT Analysis**: 7-day activity patterns → sleep predictions
4. **Gemini Processing**: Health data + context → natural language
5. **Storage**: DynamoDB with audit logging

### Supported Health Metrics
- **Activity**: Steps, distance, calories, exercise minutes
- **Sleep**: Duration, efficiency, stages, disruptions
- **Cardiovascular**: Heart rate, HRV, blood pressure
- **Respiratory**: Breathing rate, SpO2
- **Mental Health**: Mood tracking, stress indicators

## AI Models

### PAT (Pretrained Actigraphy Transformer)
- **Purpose**: Sleep and circadian rhythm analysis from movement data
- **Input**: 10,080-point vectors (7 days × 1-minute intervals)
- **Architecture**: Transformer with temporal positional encoding
- **License**: CC BY-4.0 (Dartmouth College, Jacobson Lab)

### Google Gemini Integration
- **Purpose**: Natural language health insights
- **Input**: Processed health metrics + user context
- **Output**: Conversational explanations and recommendations

## Production Deployment

**AWS ECS Fargate** with:
- Auto-scaling based on CPU/memory
- Zero-downtime rolling deployments
- CloudWatch logging and monitoring
- Prometheus metrics collection

```bash
# Deploy to AWS
./deploy.sh production

# Monitor logs
aws logs tail /aws/ecs/clarity-backend --follow
```

## Development

```bash
# Install dependencies
make install

# Run tests (810 total, 807 passing)
make test

# Code quality
make lint      # Ruff linting
make typecheck # MyPy validation
make format    # Black formatting

# Coverage report (currently 57%, target 85%)
make coverage
```

## Current Status

- **Tests**: 807/810 passing (99.6% success rate)
- **Coverage**: 57% (increasing to 85% target)
- **API**: 44 endpoints, all core functionality complete
- **AWS Migration**: Complete, production-ready infrastructure
- **AI Models**: PAT and Gemini integration functional

## Security & Compliance

- **Authentication**: JWT via AWS Cognito
- **Encryption**: End-to-end for all health data
- **HIPAA Ready**: AWS infrastructure with audit logging
- **Privacy**: User data control, configurable retention

## Documentation

- **[System Overview](docs/01-overview.md)** - Architecture details
- **[API Reference](docs/02-api-reference.md)** - Complete endpoint docs
- **[AI Models](docs/03-ai-models.md)** - ML pipeline documentation
- **[Deployment Guide](docs/operations/deployment.md)** - Production setup

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

**Third-party**: PAT models under CC BY-4.0 - see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

---

**CLARITY Digital Twin Backend** - Health data processing with AI-powered insights