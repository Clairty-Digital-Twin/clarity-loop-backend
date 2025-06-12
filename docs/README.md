# CLARITY Digital Twin Platform - Documentation

Technical documentation for the CLARITY health data processing backend.

## Overview

CLARITY is a FastAPI-based platform that processes Apple Health data using machine learning models to extract health patterns and generate natural language insights. The system combines a Pretrained Actigraphy Transformer (PAT) for temporal analysis with Google Gemini for conversational health explanations.

## Quick Navigation

### Core Documentation

1. **[System Overview](01-overview.md)** 
   - Technical architecture and components
   - Data flow and processing pipeline
   - Performance characteristics and metrics
   - Current development status

2. **[API Reference](02-api-reference.md)**
   - Complete endpoint documentation (44 endpoints)
   - Authentication and authorization
   - Request/response schemas
   - Error handling and status codes

3. **[AI Models & Machine Learning](03-ai-models.md)**
   - PAT (Pretrained Actigraphy Transformer) details
   - Google Gemini integration
   - Model performance and validation
   - Deployment and monitoring

4. **[Apple HealthKit Integration](integrations/healthkit.md)**
   - Data processing pipeline
   - Supported metrics and data types
   - Performance benchmarks
   - Troubleshooting guide

## System Architecture

```
Apple Health Data → FastAPI Backend → ML Analysis → Health Insights
                      ↓
                 AWS Infrastructure
```

**Core Stack**: Python 3.11+, FastAPI, AWS (DynamoDB/Cognito/ECS), PyTorch, Transformers

### Key Components
- **Data Ingestion**: HealthKit JSON processing and validation
- **PAT Analysis**: 7-day temporal pattern analysis using transformers
- **Gemini Integration**: Natural language insight generation
- **Real-time API**: WebSocket support for live data streaming
- **AWS Infrastructure**: Production-ready cloud deployment

## Performance Metrics

| Component | Status |
|-----------|--------|
| **Tests** | 807/810 passing (99.6%) |
| **Coverage** | 57% (target: 85%) |
| **API Endpoints** | 44 total, sub-second response times |
| **Infrastructure** | AWS ECS with auto-scaling |
| **AI Models** | PAT + Gemini integration operational |

## Development Setup

```bash
# Setup
git clone https://github.com/your-org/clarity-loop-backend.git
cd clarity-loop-backend
make install

# Configure
cp .env.example .env  # Add API keys

# Run
make dev
curl http://localhost:8000/health
```

## Production Deployment

**AWS ECS Fargate** with:
- Auto-scaling containers
- Zero-downtime deployments
- CloudWatch monitoring
- Prometheus metrics

```bash
./deploy.sh production
aws logs tail /aws/ecs/clarity-backend --follow
```

## API Overview

**44 endpoints** across 7 areas:

- **Authentication** (7) - AWS Cognito integration
- **Health Data** (10) - CRUD operations for health metrics
- **HealthKit** (4) - Apple Health data processing
- **AI Insights** (6) - Gemini analysis and chat
- **PAT Analysis** (5) - Transformer model inference
- **WebSocket** (3) - Real-time data streaming
- **System** (9) - Health checks and monitoring

## AI Models

### PAT (Pretrained Actigraphy Transformer)
- **Input**: 10,080-point vectors (7 days × 1-minute intervals)
- **Purpose**: Sleep and circadian rhythm analysis
- **License**: CC BY-4.0 (Dartmouth College)

### Google Gemini
- **Purpose**: Natural language health insights
- **Input**: Processed health metrics + context
- **Output**: Conversational explanations

## Security & Compliance

- **Authentication**: JWT via AWS Cognito
- **Encryption**: End-to-end for health data
- **HIPAA Ready**: AWS infrastructure with audit logging
- **Privacy**: User-controlled data retention

## Documentation Structure

### Core Platform
- [System Overview](01-overview.md) - Technical architecture
- [API Reference](02-api-reference.md) - Complete endpoint docs
- [AI Models](03-ai-models.md) - ML pipeline details

### Integrations
- [Apple HealthKit](integrations/healthkit.md) - Data integration
- [Authentication](integrations/auth.md) - AWS Cognito setup
- [WebSockets](integrations/websockets.md) - Real-time connections

### Operations
- [Deployment](operations/deployment.md) - Production setup
- [Monitoring](operations/monitoring.md) - System health
- [Security](operations/security.md) - HIPAA compliance

## Current Status

- **Backend**: Production-ready with comprehensive test suite
- **AWS Migration**: Complete infrastructure deployment
- **AI Integration**: PAT and Gemini models operational
- **Coverage**: Increasing from 57% to 85% target
- **Documentation**: Comprehensive technical docs

## License

Apache License 2.0. PAT models under CC BY-4.0.

---

*Updated: January 2024* 