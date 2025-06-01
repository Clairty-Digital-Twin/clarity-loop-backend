# Development Documentation

This directory contains comprehensive development guides and documentation for building, testing, and deploying the Clarity Loop Backend.

## Documentation Structure

### Core Development Guides

- **[Setup & Installation](./setup.md)** - Development environment setup and local installation
- **[Development Workflow](./workflow.md)** - Git workflow, branching strategy, and development practices
- **[Testing Strategy](./testing.md)** - Unit testing, integration testing, and test automation
- **[Deployment Guide](./deployment.md)** - Google Cloud deployment and CI/CD pipeline
- **[Contributing Guidelines](./contributing.md)** - Code style, review process, and contribution standards

### Technical References

- **[API Development](./api-development.md)** - FastAPI development patterns and best practices
- **[Database Management](./database.md)** - Firestore operations, indexing, and data migration
- **[ML Pipeline Development](./ml-pipeline.md)** - Actigraphy Transformer integration and ML ops
- **[Security Implementation](./security.md)** - Security controls implementation and testing
- **[Performance Optimization](./performance.md)** - Performance monitoring and optimization techniques

### Tools and Utilities

- **[Local Development Tools](./tools.md)** - Development utilities and helper scripts
- **[Debugging Guide](./debugging.md)** - Debugging techniques and troubleshooting
- **[Monitoring & Observability](./monitoring.md)** - Local and production monitoring setup

## Quick Start for New Developers

### Prerequisites

- **Python**: 3.9+ with pip and virtualenv
- **Google Cloud SDK**: Latest version with authentication
- **Docker**: For containerized development and testing
- **Firebase CLI**: For authentication and database emulators
- **Git**: Version control with SSH key setup

### Initial Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/your-org/clarity-loop-backend.git
cd clarity-loop-backend

# 2. Run setup script
./scripts/setup-dev.sh

# 3. Start development environment
make dev-start

# 4. Verify installation
make test-quick
```

### Development Environment Overview

#### Local Development Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Development Environment                  │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Server (localhost:8000)                          │
│  ├── Auto-reload enabled                                   │
│  ├── Debug logging                                         │
│  └── Hot-reload on file changes                           │
│                                                            │
│  Firebase Emulator Suite (localhost:4000)                 │
│  ├── Authentication Emulator (localhost:9099)            │
│  ├── Firestore Emulator (localhost:8080)                 │
│  └── Functions Emulator (localhost:5001)                 │
│                                                            │
│  ML Development Environment                                │
│  ├── Jupyter Notebooks (localhost:8888)                  │
│  ├── Model serving endpoint (localhost:8001)             │
│  └── Tensor flow serving (localhost:8501)                │
│                                                            │
│  Monitoring & Debugging                                   │
│  ├── API Documentation (localhost:8000/docs)             │
│  ├── Health Dashboard (localhost:3000)                   │
│  └── Log Aggregation (localhost:5601)                    │
└─────────────────────────────────────────────────────────────┘
```

#### Development Workflow

1. **Feature Development**: Branch from `develop`, implement feature, write tests
2. **Local Testing**: Run unit tests, integration tests, and manual testing
3. **Code Review**: Submit PR, address feedback, ensure CI passes
4. **Staging Deployment**: Automatic deployment to staging environment
5. **Production Release**: Manual approval for production deployment

## Technology Stack Details

### Backend Framework

- **FastAPI**: Async Python web framework with automatic API documentation
- **Pydantic**: Data validation and serialization with type hints
- **Asyncio**: Asynchronous programming for high concurrency
- **Uvicorn**: ASGI server for production deployment

### Cloud Infrastructure

- **Google Cloud Run**: Serverless container platform for API services
- **Cloud Firestore**: NoSQL document database for user data
- **Cloud Storage**: Object storage for files and ML model artifacts
- **Cloud Pub/Sub**: Message queue for asynchronous processing
- **Vertex AI**: ML model serving and training platform

### Authentication & Security

- **Firebase Authentication**: User identity and authentication
- **Cloud IAM**: Service-to-service authentication and authorization
- **Cloud KMS**: Encryption key management
- **Cloud Security Command Center**: Security monitoring and alerts

### ML & Analytics

- **TensorFlow**: ML model development and training
- **Scikit-learn**: Classical ML algorithms and preprocessing
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Google AI Platform**: Model training and hyperparameter tuning

### Development Tools

- **Poetry**: Python dependency management and packaging
- **Black**: Code formatting
- **Flake8**: Code linting and style checking
- **pytest**: Unit and integration testing framework
- **mypy**: Static type checking

## Development Standards

### Code Quality Standards

- **Type Hints**: All functions must include type annotations
- **Documentation**: Comprehensive docstrings for all public APIs
- **Test Coverage**: Minimum 80% code coverage required
- **Code Style**: Black formatting with line length 88 characters
- **Security**: All user inputs validated, no hardcoded secrets

### Performance Requirements

- **API Response Time**: P95 < 500ms for simple queries
- **Database Queries**: Efficient indexing, avoid N+1 queries
- **Memory Usage**: < 512MB per container instance
- **Concurrent Users**: Support 1000+ concurrent users per instance

### Security Requirements

- **Input Validation**: All inputs validated with Pydantic models
- **Authentication**: All endpoints require valid Firebase tokens
- **Authorization**: Role-based access control implemented
- **Data Encryption**: All sensitive data encrypted at rest and in transit
- **Audit Logging**: All data access and modifications logged

## Environment Configuration

### Local Development

```yaml
# .env.local
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Firebase Configuration
FIREBASE_PROJECT_ID=clarity-loop-dev
FIREBASE_USE_EMULATOR=true
FIRESTORE_EMULATOR_HOST=localhost:8080
FIREBASE_AUTH_EMULATOR_HOST=localhost:9099

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=clarity-loop-dev
GOOGLE_APPLICATION_CREDENTIALS=./service-account-dev.json

# Database Configuration
DATABASE_URL=firestore://clarity-loop-dev
DATABASE_POOL_SIZE=10

# ML Configuration
ML_MODEL_ENDPOINT=http://localhost:8001
ACTIGRAPHY_MODEL_VERSION=2.1.0
GEMINI_API_ENDPOINT=mock

# Security Configuration
JWT_SECRET_KEY=dev-secret-key-change-in-production
ENCRYPTION_KEY=dev-encryption-key-32-characters
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

### Staging Environment

```yaml
# .env.staging
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# Firebase Configuration
FIREBASE_PROJECT_ID=clarity-loop-staging
FIREBASE_USE_EMULATOR=false

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=clarity-loop-staging
GOOGLE_APPLICATION_CREDENTIALS=/secrets/service-account.json

# Database Configuration
DATABASE_URL=firestore://clarity-loop-staging
DATABASE_POOL_SIZE=20

# ML Configuration
ML_MODEL_ENDPOINT=https://ml-staging.clarityloop.com
ACTIGRAPHY_MODEL_VERSION=2.1.0
GEMINI_API_ENDPOINT=https://generativelanguage.googleapis.com

# Security Configuration
JWT_SECRET_KEY=${SECRET_MANAGER_JWT_KEY}
ENCRYPTION_KEY=${SECRET_MANAGER_ENCRYPTION_KEY}
CORS_ORIGINS=["https://staging.clarityloop.com"]
```

## Common Development Tasks

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-api           # API endpoint tests
make test-ml            # ML pipeline tests

# Run tests with coverage
make test-coverage

# Run tests in watch mode
make test-watch
```

### Database Operations

```bash
# Start Firestore emulator
make db-start

# Seed test data
make db-seed

# Run migrations
make db-migrate

# Backup local data
make db-backup

# Reset database
make db-reset
```

### ML Development

```bash
# Start ML development environment
make ml-dev-start

# Train model locally
make ml-train-local

# Serve model locally
make ml-serve-local

# Run model evaluation
make ml-evaluate

# Deploy model to staging
make ml-deploy-staging
```

### Code Quality Checks

```bash
# Format code
make format

# Lint code
make lint

# Type checking
make type-check

# Security scan
make security-scan

# Full quality check
make quality-check
```

## Debugging and Troubleshooting

### Common Issues and Solutions

#### Issue: Import Errors

```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements-dev.txt
```

#### Issue: Firebase Emulator Connection

```bash
# Solution: Restart Firebase emulators
firebase emulators:kill
firebase emulators:start --import=./firebase-export
```

#### Issue: ML Model Loading

```bash
# Solution: Check model path and version
export ACTIGRAPHY_MODEL_PATH=/path/to/model
python scripts/verify-model.py
```

### Debug Mode Configuration

```python
# main.py - Enable debug mode
import logging
from fastapi import FastAPI

if os.getenv("DEBUG", "false").lower() == "true":
    logging.basicConfig(level=logging.DEBUG)
    app = FastAPI(debug=True, docs_url="/docs", redoc_url="/redoc")
else:
    app = FastAPI(docs_url=None, redoc_url=None)
```

## Performance Profiling

### Local Performance Testing

```bash
# API load testing
make load-test-local

# Database performance testing
make db-perf-test

# ML inference benchmarking
make ml-benchmark

# Memory profiling
make profile-memory
```

### Monitoring Setup

```bash
# Start monitoring stack
make monitoring-start

# View metrics dashboard
open http://localhost:3000

# Check application logs
make logs-tail

# Health check
make health-check
```

## Getting Help

### Resources

- **API Documentation**: <http://localhost:8000/docs> (when running locally)
- **Architecture Docs**: `docs/architecture/`
- **Troubleshooting Guide**: `docs/development/debugging.md`
- **Team Wiki**: [Internal Wiki Link]

### Support Channels

- **Slack**: #clarity-backend-dev
- **Email**: <backend-team@clarityloop.com>
- **Office Hours**: Tuesday/Thursday 2-3 PM PST

### Onboarding Checklist

- [ ] Development environment setup completed
- [ ] All tests passing locally
- [ ] Access to required Google Cloud projects
- [ ] Firebase project permissions configured
- [ ] Slack channels joined
- [ ] First PR submitted and merged
- [ ] Code review process completed
- [ ] Documentation contribution made

This development documentation provides a comprehensive foundation for efficient backend development with clear standards, workflows, and support resources.
