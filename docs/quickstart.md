# Developer Quick-Start Guide

Get from `git clone` to running the full Clarity Loop Backend stack in under 10 minutes.

## Prerequisites

### Required Tools

```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Docker Desktop
# Download from: https://docker.com/products/docker-desktop

# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash
```

### System Requirements

- **Python**: 3.11+ (managed by uv)
- **Node.js**: 18+ (for Firebase emulator)
- **Docker**: Latest version
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB free space

## One-Command Setup

```bash
# Clone and setup entire development environment
git clone git@github.com:The-Obstacle-Is-The-Way/clarity-loop-backend.git
cd clarity-loop-backend
make dev-setup
```

## Manual Setup (if needed)

### 1. Python Environment

```bash
# Create virtual environment with uv
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
uv pip install -e .
uv pip install -r requirements-dev.txt
```

### 2. Google Cloud Setup

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# Set project (replace with your project ID)
gcloud config set project clarity-loop-production

# Install Firebase CLI
npm install -g firebase-tools
firebase login
```

### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your values
nano .env
```

### 4. Development Services

```bash
# Start all development services
docker-compose up -d

# Or individually:
make firestore-emulator    # Firestore emulator
make pubsub-emulator      # Pub/Sub emulator  
make storage-emulator     # Cloud Storage emulator
```

## Environment Variables

Create `.env` file with these required variables:

```bash
# === Google Cloud Configuration ===
GOOGLE_CLOUD_PROJECT=clarity-loop-development
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json

# === Firebase Configuration ===
FIREBASE_PROJECT_ID=clarity-loop-development
FIRESTORE_EMULATOR_HOST=localhost:8080
PUBSUB_EMULATOR_HOST=localhost:8085

# === API Configuration ===
API_HOST=localhost
API_PORT=8000
API_ENVIRONMENT=development
DEBUG=true

# === Security ===
SECRET_KEY=your-super-secret-development-key-here
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]

# === ML Service Configuration ===
VERTEX_AI_LOCATION=us-central1
PAT_MODEL_VERSION=small
GEMINI_MODEL=gemini-2.0-flash-exp

# === Database ===
DATABASE_URL=firestore://localhost:8080
REDIS_URL=redis://localhost:6379

# === Monitoring (Development) ===
LOG_LEVEL=debug
ENABLE_TRACING=true
METRICS_ENABLED=true
```

## Verification Steps

### 1. Run Health Checks

```bash
# Check all services
make health-check

# Individual checks
curl http://localhost:8000/health          # API Gateway
curl http://localhost:8001/health          # ML Service
curl http://localhost:8080/health          # Firestore emulator
```

### 2. Run Test Suite

```bash
# Full test suite
make test

# Individual test types
make test-unit           # Unit tests
make test-integration    # Integration tests
make test-e2e           # End-to-end tests
```

### 3. Load Sample Data

```bash
# Upload sample HealthKit data
make seed-data

# Verify in Firestore emulator UI
open http://localhost:4000
```

## Development Tools

### Code Quality (Pre-configured)

```bash
# Format code
make format              # ruff format + isort

# Lint code
make lint               # ruff check + mypy

# Run pre-commit hooks
pre-commit run --all-files
```

### Testing Tools

```bash
# Run tests with coverage
make test-coverage

# Run specific test file
pytest tests/test_healthkit.py -v

# Run with debugging
pytest tests/test_ml_service.py -s --pdb
```

### Development Server

```bash
# Start API gateway with hot reload
make dev-api

# Start ML service with hot reload  
make dev-ml

# Start all services
make dev-all
```

## Common Development Workflows

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/healthkit-integration

# Start development environment
make dev-all

# Make changes, test iteratively
make test-unit
make lint
make format

# Integration test
make test-integration

# Commit when ready
git add .
git commit -m "feat: implement HealthKit data validation"
```

### 2. Testing Changes

```bash
# Quick feedback loop
make test-changed        # Only test changed files
make lint-changed        # Only lint changed files

# Full validation before commit
make validate           # format + lint + test + coverage
```

### 3. Debugging Issues

```bash
# Start with debug logging
DEBUG=true LOG_LEVEL=debug make dev-all

# Check service logs
docker-compose logs -f api-gateway
docker-compose logs -f ml-service

# Connect to services
docker-compose exec api-gateway bash
```

## Makefile Commands

```bash
# Setup and Installation
make install             # Install Python dependencies
make install-dev         # Install development dependencies
make dev-setup          # Complete development setup

# Development
make dev                # Start development servers
make dev-api            # Start API gateway only
make dev-ml             # Start ML service only

# Code Quality
make format             # Format code (ruff + isort)
make lint               # Lint code (ruff + mypy)
make type-check         # Type checking only
make validate           # Full validation pipeline

# Testing
make test               # Run all tests
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-e2e          # End-to-end tests only
make test-coverage     # Tests with coverage report

# Database and Services
make db-migrate         # Run database migrations
make db-seed           # Load sample data
make services-up       # Start Docker services
make services-down     # Stop Docker services

# Deployment
make build             # Build Docker images
make deploy-dev        # Deploy to development
make deploy-prod       # Deploy to production

# Utilities
make clean             # Clean build artifacts
make health-check      # Check all services
make logs              # View service logs
make shell             # Interactive shell
```

## Troubleshooting

### Python Environment Issues

```bash
# Reset virtual environment
rm -rf .venv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### Google Cloud Authentication

```bash
# Re-authenticate
gcloud auth revoke --all
gcloud auth login
gcloud auth application-default login

# Verify credentials
gcloud auth list
gcloud config list
```

### Docker Service Issues

```bash
# Reset Docker services
docker-compose down -v
docker-compose up -d

# Check service status
docker-compose ps
docker-compose logs SERVICE_NAME
```

### Port Conflicts

```bash
# Check what's using ports
lsof -i :8000  # API Gateway
lsof -i :8001  # ML Service
lsof -i :8080  # Firestore emulator

# Kill processes if needed
kill -9 PID
```

## IDE Setup

### VS Code (Recommended)

Install these extensions:

- Python (Microsoft)
- Pylance (Microsoft)
- Ruff (Astral Software)
- Docker (Microsoft)
- Firebase (Firebase)

Workspace settings (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.formatting.provider": "none",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "ruff.enable": true,
  "ruff.organizeImports": true
}
```

### PyCharm

1. Open project in PyCharm
2. Configure Python interpreter: `.venv/bin/python`
3. Install plugins: Docker, Google Cloud Tools
4. Configure ruff as external tool

## Next Steps

After successful setup:

1. **Read the Architecture**: Review `docs/blueprint.md`
2. **Study the APIs**: Check `docs/api/` directory
3. **Understand ML Pipeline**: Read `docs/development/ml-pipeline.md`
4. **Review Test Strategy**: See `docs/development/testing.md`
5. **Check Security Guidelines**: Review `docs/development/security.md`

## Getting Help

- **Documentation**: All docs in `docs/` directory
- **Issues**: GitHub Issues tab
- **Slack**: #clarity-loop-backend channel
- **Code Reviews**: Follow PR template

---

**Target**: From `git clone` to passing tests in **< 10 minutes**  
**Stack**: FastAPI + Google Cloud + Modern Python tooling  
**Goal**: Friction-free development with autonomous agent support
