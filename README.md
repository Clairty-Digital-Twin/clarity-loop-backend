# Clarity Loop Backend

Enterprise-grade async-first backend for HealthKit wellness applications with AI-powered health insights.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Native-blue.svg)](https://cloud.google.com)
[![HIPAA](https://img.shields.io/badge/HIPAA-Compliant-red.svg)](https://www.hhs.gov/hipaa)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://astral.sh/uv) package manager
- Google Cloud SDK
- Firebase CLI
- Docker Desktop

### Installation

```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone https://github.com/your-org/clarity-loop-backend.git
cd clarity-loop-backend

# Install dependencies with uv
uv sync --extra dev

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration

# Run development server
uv run uvicorn src.clarity.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Development

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individual services
docker-compose up api
docker-compose up ml-processor
```

## 🏗️ Architecture Overview

### Async-First Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   iOS/watchOS   │───▶│   FastAPI       │───▶│  Cloud Run      │
│   HealthKit     │    │   Gateway       │    │  Microservices  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Pub/Sub       │    │  ML Pipeline    │
                       │   Queue         │    │  (Actigraphy)   │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Firestore     │◀───│  Gemini 2.5 Pro │
                       │   Real-time DB  │    │  AI Insights    │
                       └─────────────────┘    └─────────────────┘
```

### Core Services

- **API Gateway** (FastAPI + Cloud Run) - REST endpoints with authentication
- **ML Processor** (Actigraphy Transformer) - Health data analytics service  
- **AI Insights** (Gemini 2.5 Pro) - Natural language health recommendations
- **Auth Service** (Firebase) - Identity management and access control
- **Data Pipeline** (Pub/Sub + Firestore) - Async processing and storage

## 🔐 Security & Compliance

### HIPAA-Inspired Security

- **End-to-end encryption** with Google Cloud KMS
- **Zero-trust architecture** with least privilege access
- **Audit logging** for all health data operations
- **Data segregation** by user with access controls
- **Secure networking** via VPC and private endpoints

### Authentication Flow

```python
# Example: Secure health data upload
@app.post("/api/v1/health-data/upload")
@require_permission(Permission.WRITE_OWN_DATA)
async def upload_health_data(
    data: SecureHealthDataInput,
    current_user: User = Depends(get_current_user)
):
    # Validate, encrypt, and process health data
    processing_id = await health_processor.process_async(
        user_id=current_user.id,
        data=data
    )
    return {"processing_id": processing_id, "status": "accepted"}
```

## 🤖 AI-Powered Features

### Health Data Chat

Users can interact with their health data through natural language:

**User**: *"How has my sleep quality changed this month?"*

**AI Response**: *"Your sleep quality has improved by 15% this month. You're averaging 7.2 hours per night with 85% deep sleep efficiency. Key improvements: consistent bedtime routine and reduced late-evening screen time."*

### ML Pipeline

1. **Data Ingestion** - Real-time HealthKit data processing
2. **Actigraphy Analysis** - Sleep stages, activity patterns, circadian rhythm
3. **Trend Analysis** - Weekly/monthly health pattern recognition  
4. **AI Insights** - Personalized recommendations via Gemini 2.5 Pro
5. **Real-time Delivery** - Push insights to iOS app via Firestore

## 📊 API Documentation

### Health Data Endpoints

```http
POST   /api/v1/health-data/upload     # Upload health data
GET    /api/v1/health-data/export     # Export user data  
DELETE /api/v1/health-data/purge      # Delete user data

GET    /api/v1/insights/daily         # Daily health summary
GET    /api/v1/insights/weekly        # Weekly trends
POST   /api/v1/insights/chat          # Chat with health AI

GET    /api/v1/user/profile           # User profile
PUT    /api/v1/user/preferences       # Update preferences
```

### Example Request

```bash
curl -X POST "https://api.clarityloop.com/api/v1/health-data/upload" \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "heart_rate",
    "values": [
      {"value": 72, "timestamp": "2024-01-01T12:00:00Z"},
      {"value": 74, "timestamp": "2024-01-01T12:01:00Z"}
    ],
    "source": "apple_watch"
  }'
```

## 🛠️ Development

### Project Structure

```
clarity-loop-backend/
├── src/clarity/               # Main application code
│   ├── api/                   # FastAPI routes and endpoints  
│   ├── ml/                    # ML models and processing
│   ├── auth/                  # Authentication and authorization
│   ├── models/                # Pydantic data models
│   └── services/              # Business logic services
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── e2e/                   # End-to-end tests
├── docs/                      # Documentation
│   ├── architecture/          # System architecture docs
│   ├── api/                   # API documentation
│   └── development/           # Development guides
├── infrastructure/            # Terraform IaC
├── scripts/                   # Deployment and utility scripts
└── pyproject.toml            # Modern Python project config
```

### Development Workflow

```bash
# Start development environment
uv sync --extra dev
uv run pre-commit install

# Run tests
uv run pytest                  # All tests
uv run pytest tests/unit      # Unit tests only
uv run pytest -m integration  # Integration tests

# Code quality
uv run ruff check src/         # Linting
uv run black src/              # Code formatting  
uv run mypy src/               # Type checking

# Run development server
uv run uvicorn src.clarity.main:app --reload
```

## 🚢 Deployment

### Production Deployment

```bash
# Deploy to Google Cloud Run
gcloud run deploy clarity-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Deploy ML services
gcloud run deploy ml-processor \
  --source ./ml-service \
  --platform managed \
  --region us-central1
```

### Environment Configuration

- **Local**: SQLite + Firebase Emulator
- **Staging**: Cloud SQL + Firebase Auth
- **Production**: Firestore + Full Google Cloud stack

## 📈 Monitoring & Observability

### Health Checks

- **Liveness**: `/health/live` - Basic application health
- **Readiness**: `/health/ready` - Ready to serve requests
- **Detailed**: `/health/detailed` - Full dependency status

### Metrics & Logging

- **Structured logging** with Google Cloud Logging
- **Distributed tracing** via OpenTelemetry
- **Custom metrics** for business KPIs
- **SLA monitoring** with 99.9% uptime target

## 📚 Documentation

### Complete Documentation Suite

- **[Setup Guide](docs/development/setup.md)** - Environment setup and installation
- **[API Reference](docs/api/)** - Complete API documentation
- **[Architecture](docs/architecture/)** - System design and patterns
- **[Security](docs/development/security.md)** - Security implementation
- **[Testing](docs/development/testing.md)** - Testing strategy and examples
- **[Deployment](docs/development/deployment.md)** - Deployment procedures
- **[Monitoring](docs/development/monitoring.md)** - Observability and alerting

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Standards

- **Test Coverage**: Minimum 80% overall, 95% for critical paths
- **Code Style**: Black + Ruff for formatting and linting
- **Type Safety**: Full mypy compliance
- **Documentation**: All public APIs documented
- **Security**: Security review for all health data changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Production API**: <https://api.clarityloop.com>
- **Documentation**: <https://docs.clarityloop.com>  
- **Status Page**: <https://status.clarityloop.com>
- **Support**: <support@clarityloop.com>

---

**Built with ❤️ for healthcare innovation**
