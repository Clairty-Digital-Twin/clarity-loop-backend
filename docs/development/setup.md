# Setup & Installation Guide

This guide provides step-by-step instructions for setting up the Clarity Loop Backend development environment on macOS, Linux, and Windows.

## Prerequisites

### System Requirements
- **Operating System**: macOS 10.15+, Ubuntu 18.04+, or Windows 10+
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 20GB available disk space
- **Network**: Stable internet connection for cloud services

### Required Software

#### 1. Python 3.9+
```bash
# macOS (using Homebrew)
brew install python@3.9

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.9 python3.9-pip python3.9-venv

# Windows (using Chocolatey)
choco install python --version=3.9.0

# Verify installation
python3 --version  # Should show 3.9.x or higher
```

#### 2. Google Cloud SDK
```bash
# macOS
brew install --cask google-cloud-sdk

# Ubuntu/Debian
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-sdk

# Windows
# Download and install from: https://cloud.google.com/sdk/docs/install

# Verify installation
gcloud --version
```

#### 3. Firebase CLI
```bash
# Install via npm (requires Node.js)
npm install -g firebase-tools

# Alternative: Download binary
curl -sL https://firebase.tools | bash

# Verify installation
firebase --version
```

#### 4. Docker Desktop
```bash
# macOS
brew install --cask docker

# Ubuntu
sudo apt-get install docker.io docker-compose

# Windows
# Download Docker Desktop from: https://www.docker.com/products/docker-desktop

# Verify installation
docker --version
docker-compose --version
```

#### 5. Git
```bash
# macOS
brew install git

# Ubuntu/Debian
sudo apt-get install git

# Windows
# Download from: https://git-scm.com/download/win

# Verify installation
git --version
```

## Project Setup

### 1. Clone Repository
```bash
# Clone the repository
git clone https://github.com/your-org/clarity-loop-backend.git
cd clarity-loop-backend

# Verify clone
ls -la  # Should show project files
```

### 2. Python Environment Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install Poetry (optional but recommended)
pip install poetry

# Install dependencies
# Using pip
pip install -r requirements-dev.txt

# Or using Poetry
poetry install --with dev
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.template .env.local

# Edit environment variables
nano .env.local  # or your preferred editor
```

#### Environment Variables Configuration
```bash
# .env.local
# =============================================
# ENVIRONMENT CONFIGURATION
# =============================================
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
HOST=localhost
PORT=8000

# =============================================
# FIREBASE CONFIGURATION
# =============================================
FIREBASE_PROJECT_ID=clarity-loop-dev
FIREBASE_USE_EMULATOR=true
FIRESTORE_EMULATOR_HOST=localhost:8080
FIREBASE_AUTH_EMULATOR_HOST=localhost:9099
FIREBASE_STORAGE_EMULATOR_HOST=localhost:9199

# Firebase Admin SDK (for server-side operations)
FIREBASE_ADMIN_CREDENTIALS_PATH=./config/firebase-admin-dev.json
FIREBASE_WEB_API_KEY=your-web-api-key-here

# =============================================
# GOOGLE CLOUD CONFIGURATION
# =============================================
GOOGLE_CLOUD_PROJECT=clarity-loop-dev
GOOGLE_APPLICATION_CREDENTIALS=./config/service-account-dev.json
GOOGLE_CLOUD_REGION=us-central1

# Cloud Storage
CLOUD_STORAGE_BUCKET=clarity-loop-dev-storage
CLOUD_STORAGE_ML_BUCKET=clarity-loop-dev-ml-models

# Pub/Sub Topics
PUBSUB_HEALTH_DATA_TOPIC=health-data-processing
PUBSUB_INSIGHTS_TOPIC=insights-generation
PUBSUB_USER_EVENTS_TOPIC=user-events

# =============================================
# DATABASE CONFIGURATION
# =============================================
DATABASE_TYPE=firestore
DATABASE_PROJECT_ID=clarity-loop-dev
DATABASE_TIMEOUT=30
DATABASE_RETRY_ATTEMPTS=3

# Connection pooling
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# =============================================
# ML & AI CONFIGURATION
# =============================================
# Actigraphy Transformer Service
ACTIGRAPHY_SERVICE_URL=http://localhost:8001
ACTIGRAPHY_MODEL_VERSION=v2.1.0
ACTIGRAPHY_BATCH_SIZE=32
ACTIGRAPHY_TIMEOUT=60

# Gemini AI Configuration
GEMINI_API_ENDPOINT=https://generativelanguage.googleapis.com
GEMINI_MODEL_NAME=gemini-2.0-flash-exp
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=1000

# Vertex AI Configuration
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_STAGING_BUCKET=clarity-loop-vertex-staging

# =============================================
# SECURITY CONFIGURATION
# =============================================
# JWT Configuration
JWT_SECRET_KEY=dev-secret-key-32-chars-minimum
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
JWT_REFRESH_EXPIRATION_DAYS=30

# Encryption
ENCRYPTION_KEY=dev-encryption-key-exactly-32-chars
ENCRYPTION_ALGORITHM=AES-256-GCM

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
CORS_ALLOW_HEADERS=["*"]

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST_SIZE=20

# =============================================
# MONITORING & OBSERVABILITY
# =============================================
# Logging
LOG_FORMAT=json
LOG_CORRELATION_ID_HEADER=X-Correlation-ID

# Metrics
METRICS_ENABLED=true
METRICS_PORT=9090
METRICS_PATH=/metrics

# Health Checks
HEALTH_CHECK_TIMEOUT=10
HEALTH_CHECK_INTERVAL=30

# Distributed Tracing
JAEGER_AGENT_HOST=localhost
JAEGER_AGENT_PORT=6831
JAEGER_SAMPLER_RATE=0.1

# =============================================
# DEVELOPMENT TOOLS
# =============================================
# API Documentation
API_DOCS_ENABLED=true
API_DOCS_PATH=/docs
API_REDOC_PATH=/redoc

# Hot Reload
RELOAD_ON_CHANGE=true
RELOAD_DIRS=["./src", "./tests"]

# Testing
TEST_DATABASE_URL=firestore://clarity-loop-test
TEST_PARALLEL_WORKERS=4
```

### 4. Google Cloud Authentication
```bash
# Authenticate with Google Cloud
gcloud auth login

# Set default project
gcloud config set project clarity-loop-dev

# Enable required APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  firestore.googleapis.com \
  pubsub.googleapis.com \
  storage.googleapis.com \
  aiplatform.googleapis.com \
  secretmanager.googleapis.com

# Create service account for development
gcloud iam service-accounts create clarity-dev-service \
  --display-name="Clarity Development Service Account"

# Generate service account key
gcloud iam service-accounts keys create ./config/service-account-dev.json \
  --iam-account=clarity-dev-service@clarity-loop-dev.iam.gserviceaccount.com

# Grant necessary permissions
gcloud projects add-iam-policy-binding clarity-loop-dev \
  --member="serviceAccount:clarity-dev-service@clarity-loop-dev.iam.gserviceaccount.com" \
  --role="roles/datastore.user"

gcloud projects add-iam-policy-binding clarity-loop-dev \
  --member="serviceAccount:clarity-dev-service@clarity-loop-dev.iam.gserviceaccount.com" \
  --role="roles/pubsub.editor"

gcloud projects add-iam-policy-binding clarity-loop-dev \
  --member="serviceAccount:clarity-dev-service@clarity-loop-dev.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
```

### 5. Firebase Setup
```bash
# Login to Firebase
firebase login

# Set default project
firebase use clarity-loop-dev

# Download Firebase config
firebase apps:sdkconfig web > ./config/firebase-config.json

# Generate Firebase Admin SDK credentials
# Go to Firebase Console > Project Settings > Service Accounts
# Generate new private key and save as ./config/firebase-admin-dev.json

# Initialize Firebase emulators
firebase init emulators
# Select: Authentication, Firestore, Storage, Functions
# Accept default ports or customize as needed
```

### 6. Docker Environment (Optional)
```bash
# Build development Docker image
docker build -f Dockerfile.dev -t clarity-backend-dev .

# Create Docker network
docker network create clarity-network

# Start supporting services
docker-compose -f docker-compose.dev.yml up -d

# Verify services are running
docker-compose -f docker-compose.dev.yml ps
```

## Development Tools Setup

### 1. Code Quality Tools
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Install development dependencies
pip install -r requirements-dev.txt

# Configure IDE (VS Code example)
# Install extensions:
# - Python
# - Pylance
# - Black Formatter
# - Python Docstring Generator
# - GitLens
```

### 2. Testing Framework
```bash
# Install pytest and plugins
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Create pytest configuration
cat > pytest.ini << EOF
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = 
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    -v
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    api: API endpoint tests
    ml: Machine learning tests
    slow: Slow running tests
EOF
```

### 3. ML Development Environment
```bash
# Install ML dependencies
pip install tensorflow scikit-learn pandas numpy jupyter

# Set up Jupyter Notebook
jupyter notebook --generate-config

# Create ML development workspace
mkdir -p notebooks/{data-exploration,model-development,evaluation}

# Install TensorBoard
pip install tensorboard

# Start TensorBoard (optional)
tensorboard --logdir=./logs/tensorboard --port=6006
```

## Verification and Testing

### 1. Environment Verification
```bash
# Run environment check script
python scripts/check-environment.py

# Expected output:
# ✓ Python 3.9+ installed
# ✓ Google Cloud SDK authenticated
# ✓ Firebase CLI configured
# ✓ Docker running
# ✓ Environment variables set
# ✓ Dependencies installed
```

### 2. Service Health Checks
```bash
# Start Firebase emulators
firebase emulators:start --import=./firebase-export --export-on-exit

# In another terminal, start the FastAPI server
python -m uvicorn src.main:app --reload --host=localhost --port=8000

# Verify services are responding
curl http://localhost:8000/health
curl http://localhost:4000  # Firebase Emulator UI
curl http://localhost:8080  # Firestore Emulator
curl http://localhost:9099  # Auth Emulator
```

### 3. Run Test Suite
```bash
# Run quick tests
make test-quick

# Run full test suite
make test

# Expected output: All tests should pass
# Coverage should be > 80%
```

### 4. API Documentation Access
```bash
# Start server (if not already running)
uvicorn src.main:app --reload

# Access interactive API docs
open http://localhost:8000/docs     # Swagger UI
open http://localhost:8000/redoc    # ReDoc
```

## Common Setup Issues and Solutions

### Issue 1: Python Version Conflicts
```bash
# Problem: Multiple Python versions causing conflicts
# Solution: Use pyenv for Python version management

# Install pyenv
curl https://pyenv.run | bash

# Install Python 3.9
pyenv install 3.9.18
pyenv local 3.9.18

# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
```

### Issue 2: Google Cloud Authentication
```bash
# Problem: Authentication errors with Google Cloud
# Solution: Re-authenticate and check permissions

# Clear existing credentials
gcloud auth revoke --all

# Re-authenticate
gcloud auth login
gcloud auth application-default login

# Verify authentication
gcloud auth list
gcloud config list
```

### Issue 3: Firebase Emulator Issues
```bash
# Problem: Firebase emulators not starting
# Solution: Check ports and restart

# Check what's using the ports
lsof -i :4000  # Emulator UI
lsof -i :8080  # Firestore
lsof -i :9099  # Auth

# Kill processes if needed
kill -9 <PID>

# Clear emulator data and restart
firebase emulators:kill
rm -rf .firebase/emulators
firebase emulators:start --import=./firebase-export
```

### Issue 4: Dependency Installation Issues
```bash
# Problem: Package installation failures
# Solution: Update pip and use specific package versions

# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Clear pip cache
pip cache purge

# Install with no cache
pip install --no-cache-dir -r requirements-dev.txt

# If specific packages fail, install individually
pip install tensorflow==2.13.0  # Use specific versions
```

### Issue 5: Docker Issues
```bash
# Problem: Docker containers not starting
# Solution: Check Docker daemon and resources

# Check Docker status
docker system info

# Prune unused resources
docker system prune -a

# Check available resources
docker system df

# Restart Docker Desktop (macOS/Windows)
# Or restart Docker daemon (Linux)
sudo systemctl restart docker
```

## IDE Configuration

### VS Code Setup
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": ["tests"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".mypy_cache": true,
    "node_modules": true
  },
  "python.envFile": "${workspaceFolder}/.env.local"
}
```

### PyCharm Setup
1. Open project in PyCharm
2. Go to Settings > Project > Python Interpreter
3. Select existing virtual environment: `./venv/bin/python`
4. Configure run configurations for FastAPI server
5. Enable pytest as test runner
6. Configure code style with Black formatter

## Next Steps

After completing the setup:

1. **Read the Development Workflow**: `docs/development/workflow.md`
2. **Run Your First Test**: `make test-quick`
3. **Start Development Server**: `make dev-start`
4. **Explore API Documentation**: `http://localhost:8000/docs`
5. **Join Team Communication**: Slack channels and team meetings
6. **Review Architecture Documentation**: `docs/architecture/`

## Getting Help

If you encounter issues during setup:

1. **Check Troubleshooting Guide**: `docs/development/debugging.md`
2. **Review Environment Check**: `python scripts/check-environment.py`
3. **Ask Team on Slack**: #clarity-backend-dev
4. **Create GitHub Issue**: For persistent problems
5. **Schedule Pair Programming**: With senior team member

The setup process should take 30-60 minutes for first-time setup. Once complete, you'll have a fully functional development environment ready for backend development.
