# CLARITY Digital Twin Platform Backend

A digital twin platform for psychiatric care and mental health monitoring that processes health data from Apple HealthKit and generates AI-powered insights using machine learning models.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

CLARITY processes real-world health data from Apple HealthKit and generates AI-powered insights using state-of-the-art machine learning models. The platform integrates the Pretrained Actigraphy Transformer (PAT) for sleep pattern analysis and Google Gemini for natural language health insights generation.

### Key Features

**AI/ML Pipeline**

- PAT (Pretrained Actigraphy Transformer) for sleep pattern analysis and circadian rhythm detection
- Google Gemini integration for natural language health insights generation
- Proxy actigraphy conversion from Apple Watch step data to clinical-grade actigraphy

**Health Data Integration**

- Apple HealthKit integration (Heart Rate, HRV, Steps, Sleep, Respiratory Rate)
- Real-time data validation and processing
- HIPAA-compliant secure storage with encryption

**Architecture**

- Clean Architecture with SOLID principles
- Async-first design with FastAPI
- Microservices-ready with Google Cloud Platform
- Production monitoring and observability

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Google Cloud Project (for production)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/The-Obstacle-Is-The-Way/clarity-loop-backend.git
cd clarity-loop-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment configuration
cp .env.example .env
# Edit .env with your configuration

# Run the development server
make dev
```

The API will be available at <http://localhost:8000> with interactive docs at <http://localhost:8000/docs>

### Docker Development

```bash
# Start all services (API + emulators)
make dev-docker

# Run tests
make test

# Check code quality
make lint
```

### Quick Demo

```bash
# Start the full platform with monitoring
bash quick_demo.sh

# Test the API
curl http://localhost:8000/health

# Run API test suite
python scripts/api_test_suite.py
```

This will start the following services:

| Service | URL | Purpose |
|---------|-----|---------|
| Main API | [localhost:8000](http://localhost:8000) | FastAPI backend |
| API Docs | [localhost:8000/docs](http://localhost:8000/docs) | Interactive OpenAPI documentation |
| Grafana | [localhost:3000](http://localhost:3000) | Monitoring dashboards (admin/admin) |
| Prometheus | [localhost:9090](http://localhost:9090) | Metrics collection |
| Jupyter Lab | [localhost:8888](http://localhost:8888) | ML model exploration |
| Firestore UI | [localhost:4000](http://localhost:4000) | Database administration |

## API Overview

### Core Endpoints

| Endpoint | Description | Authentication |
|----------|-------------|----------------|
| `POST /api/v1/auth/register` | User registration | Public |
| `POST /api/v1/auth/login` | User authentication | Public |
| `POST /api/v1/health-data/upload` | Upload health metrics | Firebase JWT |
| `GET /api/v1/health-data/` | Retrieve health data | Firebase JWT |
| `POST /api/v1/pat/analyze-step-data` | PAT actigraphy analysis | Firebase JWT |
| `POST /api/v1/insights/generate` | Generate AI health insights | Firebase JWT |

### Example Usage

```python
import httpx

# Upload Apple HealthKit data
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/health-data/upload",
        headers={"Authorization": f"Bearer {firebase_token}"},
        json={
            "user_id": "user_123",
            "data_type": "heart_rate",
            "measurements": [
                {
                    "timestamp": "2025-01-15T10:30:00Z",
                    "value": 72.5,
                    "unit": "bpm"
                }
            ],
            "source": "apple_watch"
        }
    )
```

## AI/ML Pipeline

### PAT Model Integration

The platform integrates the Pretrained Actigraphy Transformer (PAT) for sleep analysis:

```python
# Analyze step data with PAT transformer
POST /api/v1/pat/analyze-step-data
{
    "user_id": "user_123",
    "step_data": [
        {"timestamp": "2025-01-15T00:00:00Z", "step_count": 0},
        {"timestamp": "2025-01-15T00:01:00Z", "step_count": 5}
    ]
}
```

### Gemini Insights Generation

```python
# Generate natural language health insights
POST /api/v1/insights/generate
{
    "user_id": "user_123",
    "analysis_results": {
        "sleep_efficiency": 0.85,
        "circadian_score": 0.72,
        "heart_rate_avg": 68.5
    },
    "question": "How is my overall health this week?"
}
```

## Architecture

### Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│ Frameworks & Drivers (FastAPI, GCP, Firebase)           │
├─────────────────────────────────────────────────────────┤
│ Interface Adapters (Controllers, DTOs, Gateways)        │
├─────────────────────────────────────────────────────────┤
│ Application Services (Use Cases, Business Rules)        │
├─────────────────────────────────────────────────────────┤
│ Domain Entities (Health Data, User, Analysis)           │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend Core**

- FastAPI - Modern, async Python web framework
- Pydantic - Data validation and serialization
- PyTorch - ML model inference engine

**AI/ML**

- PAT (Pretrained Actigraphy Transformer) - Sleep analysis
- Google Gemini - Health insights generation
- scikit-learn, pandas - Data processing

**Infrastructure**

- Google Cloud Platform - Cloud hosting and services
- Firestore - NoSQL database with real-time sync
- Firebase Auth - User authentication and authorization
- Pub/Sub - Asynchronous message processing
- Cloud Storage - Secure file storage

**Development & Monitoring**

- pytest - Testing framework
- Black, Ruff - Code formatting and linting
- Prometheus - Metrics collection
- Grafana - Monitoring dashboards

## Security & Compliance

### HIPAA Compliance

- End-to-end encryption for health data
- Audit logging for all data access
- User data isolation and access controls
- Secure cloud infrastructure with Google Cloud BAA

### Authentication

- Firebase Authentication with JWT tokens
- Role-based access control (RBAC)
- Rate limiting and request validation
- Secure API key management

## Testing

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/api/          # API endpoint tests

# Test coverage report
make coverage
```

**Test Categories**

- Unit Tests - Business logic and entities
- Integration Tests - Service layer interactions
- API Tests - HTTP endpoint functionality
- ML Tests - Model inference and data processing

## Performance & Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Service-specific health
curl http://localhost:8000/api/v1/health-data/health
curl http://localhost:8000/api/v1/pat/health
```

### Monitoring Features

- Prometheus metrics collection
- Grafana dashboards for visualization
- Structured logging with correlation IDs
- Performance profiling and bottleneck detection

## Deployment

### Local Development

```bash
make dev-docker  # Full stack with emulators
```

### Production (Google Cloud Run)

```bash
# Build and deploy
make docker-build
make deploy-production
```

### Environment Configuration

```bash
# Required environment variables
GOOGLE_CLOUD_PROJECT=your-project-id
FIREBASE_PROJECT_ID=your-firebase-project
FIRESTORE_DATABASE_ID=(default)

# Optional for development
FIRESTORE_EMULATOR_HOST=localhost:8080
PUBSUB_EMULATOR_HOST=localhost:8085
```

## Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive OpenAPI docs
- [Architecture Guide](docs/architecture/) - Detailed system design
- [Apple HealthKit Integration](docs/integrations/healthkit.md) - Mobile app integration
- [Development Guide](docs/development/) - Local development setup
- [Deployment Guide](docs/operations/) - Production deployment

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following our coding standards
4. Run tests: `make test`
5. Run linting: `make lint`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Code Standards

- Follow Clean Architecture principles
- Maintain test coverage above 80%
- Use type hints and docstrings
- Follow Black code formatting
- Pass all linting checks

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

### AI Foundation Models for Wearable Movement Data

This platform integrates the Pretrained Actigraphy Transformer (PAT), an open-source foundation model for time-series wearable movement data developed by the Jacobson Lab at Dartmouth College.

**Citation:**

```
Ruan, Franklin Y., Zhang, Aiwei, Oh, Jenny, Jin, SouYoung, and Jacobson, Nicholas C. 
"AI Foundation Models for Wearable Movement Data in Mental Health Research." 
arXiv:2411.15240 (2024). https://doi.org/10.48550/arXiv.2411.15240
```

**Repository:** [njacobsonlab/Pretrained-Actigraphy-Transformer](https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer)  
**License:** CC-BY-4.0  
**Corresponding Author:** Franklin Ruan (<franklin.y.ruan.24@dartmouth.edu>)

### Additional Acknowledgments

- Google Gemini - Advanced language model for health insights
- Apple HealthKit - Comprehensive health data platform
- Clean Architecture - Robert C. Martin's architectural principles
