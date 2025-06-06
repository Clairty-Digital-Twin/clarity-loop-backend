# CLAUDE.md

This file provides guidance to Claude Code (claude.AI/code) when working with code in this repository.

## Development Commands

### Core Development Tasks

```bash
# Install all dependencies (Python + Node.js)
make install

# Start development server with hot reload
make dev

# Run full test suite with coverage
make test

# Type checking
make typecheck

# Code linting and formatting
make lint
make lint-fix

# Run specific test types
make test-unit           # Unit tests only
make test-integration    # Integration tests
make test-ml            # ML model tests

# Docker development
make dev-docker         # Start with Docker Compose

# Clean build artifacts
make clean
```

### Key Build Commands

- **Install**: `pip install -e ".[dev]"` for editable development install
- **Test**: `pytest -v --cov=clarity --cov-report=term-missing --cov-report=html`
- **Lint**: `ruff check . && black --check . && mypy src/clarity/`
- **Format**: `black . && ruff check . --fix`
- **Type Check**: `mypy src/clarity/ --strict`

### Single Test Execution

```bash
# Run specific test file
pytest tests/unit/test_health_data_service.py -v

# Run specific test method
pytest tests/ml/test_pat_service.py::test_analyze_step_data -v

# Run tests with specific markers
pytest -m "unit" -v
pytest -m "integration" -v
pytest -m "ml" -v
```

## Architecture Overview

CLARITY is a **FastAPI-based health AI platform** built using **Clean Architecture principles** with dependency injection. The system processes wearable health data through machine learning pipelines to generate clinical insights.

### Core Architecture Patterns

- **Clean Architecture**: Domain entities at center, dependencies flow inward
- **Dependency Injection**: IoC container manages all service dependencies  
- **Factory Pattern**: Services created through container factories
- **Repository Pattern**: Data access abstracted through ports/adapters

### Application Structure

```
src/clarity/
├── main.py                 # FastAPI app entry point
├── core/
│   ├── container.py        # DI container - composition root
│   ├── config.py          # Environment-based configuration
│   └── exceptions.py      # RFC 7807 Problem Details
├── api/v1/                # REST API endpoints
├── auth/                  # Firebase/Mock authentication
├── ml/                    # AI/ML pipeline components
├── models/               # Pydantic data models
├── ports/                # Interface definitions (Clean Architecture)
├── services/             # Business logic services
└── storage/              # Data persistence layer
```

### Key Components

**ML Pipeline:**

- **PAT (Pretrained Actigraphy Transformer)**: Sleep/behavioral analysis from movement data
- **Fusion Transformer**: Multi-modal health data integration  
- **Specialized Processors**: Sleep, cardiovascular, respiratory, activity analysis
- **Gemini Integration**: Natural language health insights

**Infrastructure:**

- **Firebase Auth**: JWT-based authentication with graceful mock fallback
- **Firestore**: Health data persistence with mock repository for development
- **Google Cloud**: Vertex AI, Secret Manager, Storage integration
- **Prometheus**: Metrics and monitoring

### Dependency Injection Flow

The `DependencyContainer` in `src/clarity/core/container.py` is the composition root:

1. **Configuration Provider**: Environment-based settings management
2. **Auth Provider**: Firebase or Mock authentication based on environment  
3. **Health Data Repository**: Firestore or Mock storage based on configuration
4. **Service Dependencies**: Automatically injected into API routes

**Environment Behaviors:**

- **Development**: Uses mock services, graceful degradation enabled
- **Testing**: Mock authentication, minimal logging  
- **Production**: Strict validation, real services required

### Service Initialization

The application uses **graceful degradation** with timeout protection:

- Auth provider initialization: 8s timeout with mock fallback
- Repository initialization: 8s timeout with mock fallback  
- Startup failure: Falls back to minimal mock functionality

### API Route Structure

Routes are configured with dependency injection in `container.py`:

- `/health` - Application health check (no auth)
- `/api/v1/auth/*` - User authentication endpoints
- `/api/v1/health-data/*` - Health data upload/retrieval (Firebase JWT required)
- `/api/v1/pat/*` - PAT analysis endpoints (Firebase JWT required)
- `/api/v1/insights/*` - Gemini AI insights (Firebase JWT required)
- `/metrics` - Prometheus monitoring metrics

### Testing Strategy

The codebase uses **pytest** with comprehensive test categories:

- **Unit tests** (`tests/unit/`): Fast, isolated component tests
- **Integration tests** (`tests/integration/`): Cross-component functionality  
- **API tests** (`tests/api/`): End-to-end API endpoint testing
- **ML tests** (`tests/ml/`): Machine learning pipeline validation

Test markers help organize execution:

```bash
pytest -m "unit"         # Fast unit tests only
pytest -m "integration"  # Integration tests  
pytest -m "ml"          # ML pipeline tests
pytest -m "requires_gcp" # Tests needing Google Cloud
```

### Environment Configuration

The system uses **Pydantic Settings** with environment-specific validation:

- Development: Permissive, detailed logging, mock fallbacks enabled
- Testing: Mock services, minimal logging, authentication disabled
- Production: Strict validation, real services required, optimized performance

Critical environment variables:

- `ENVIRONMENT` - Sets behavior mode (development/testing/production)
- `FIREBASE_PROJECT_ID` - Firebase project for authentication
- `GCP_PROJECT_ID` - Google Cloud project for services
- `SKIP_EXTERNAL_SERVICES` - Forces mock services (useful for development)

### Code Quality Standards

- **Type Safety**: Strict mypy enforcement with type hints
- **Code Style**: Black formatting + Ruff linting
- **Test Coverage**: Target >80% coverage with HTML reports
- **Security**: Bandit security scanning, HIPAA-compliant logging
- **Clean Architecture**: Dependencies flow inward, ports/adapters pattern

When modifying code, always run the quality gates:

```bash
make lint      # Code quality checks
make test      # Full test suite  
make typecheck # Type safety validation
```
