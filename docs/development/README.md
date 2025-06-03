# Development Guide

**UPDATED:** December 6, 2025 - Based on actual codebase implementation

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Google Cloud SDK (for deployment)
- Firebase project configured

### Local Development Setup

1. **Clone and Setup Environment**

   ```bash
   git clone <repository-url>
   cd clarity-loop-backend
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure Environment**

   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your configuration:
   # - Firebase credentials
   # - Google Cloud project settings
   # - API keys for Gemini AI
   ```

3. **Start Development Services**

   ```bash
   # Start with Docker Compose (recommended)
   docker-compose up -d
   
   # OR run locally
   python main.py
   ```

4. **Verify Setup**

   ```bash
   # Check health endpoint
   curl http://localhost:8000/health
   
   # View API docs
   open http://localhost:8000/docs
   ```

## Project Structure

```
clarity-loop-backend/
├── src/clarity/              # Main application code
│   ├── api/v1/              # API endpoints
│   ├── services/            # Business logic
│   ├── core/                # Domain models
│   ├── storage/             # Data persistence
│   ├── ml/                  # ML components
│   └── integrations/        # External services
├── tests/                   # Test suite (729 tests)
├── models/                  # ML model files
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Local development
├── Dockerfile              # Production container
└── main.py                 # Application entry point
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes to code
# Write tests for new functionality
# Ensure all tests pass
pytest tests/

# Check code quality
ruff check .
mypy src/

# Commit and push
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 2. Testing

**Run Full Test Suite:**

```bash
# All tests
pytest

# With coverage report
pytest --cov=src --cov-report=html

# Specific test files
pytest tests/api/v1/test_health_data.py
```

**Current Test Status:**

- ✅ **729 tests passing**
- ⚠️ **59.28% code coverage** (target: 85%)
- 🎯 **Key areas needing coverage**: API error scenarios, async processing

### 3. Code Quality

**Linting & Formatting:**

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check --fix .

# Type checking
mypy src/
```

**Code Standards:**

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Maintain test coverage above 85%

### 4. Database & Storage

**Firestore Development:**

```bash
# Use Firestore emulator for testing
gcloud emulators firestore start

# Set environment for emulator
export FIRESTORE_EMULATOR_HOST=localhost:8080
```

**Cloud Storage Development:**

```bash
# Use local storage emulator
docker run -p 9199:9199 google/cloud-sdk:alpine gcloud beta emulators storage start --host-port=0.0.0.0:9199
```

## Key Development Files

### Configuration

- `src/clarity/core/config.py` - Application configuration
- `.env` - Environment variables (not in repo)
- `docker-compose.yml` - Local development services

### Core Components

- `src/clarity/api/v1/` - REST API endpoints
- `src/clarity/services/` - Business logic services
- `src/clarity/storage/` - Data access layer
- `src/clarity/ml/` - Machine learning components

### Testing

- `tests/api/` - API endpoint tests
- `tests/services/` - Service layer tests
- `tests/integration/` - Full integration tests
- `tests/ml/` - ML component tests

## API Development

### Adding New Endpoints

1. **Create API Router**

   ```python
   # src/clarity/api/v1/new_feature.py
   from fastapi import APIRouter, Depends
   from src.clarity.auth import verify_firebase_token
   
   router = APIRouter(prefix="/new-feature", tags=["new-feature"])
   
   @router.get("/")
   async def get_feature_data(
       current_user: dict = Depends(verify_firebase_token)
   ):
       user_id = current_user["uid"]
       # Implementation here
       return {"data": "example"}
   ```

2. **Add to Main Application**

   ```python
   # src/clarity/api/v1/__init__.py
   from .new_feature import router as new_feature_router
   
   # Add to router list
   ```

3. **Write Tests**

   ```python
   # tests/api/v1/test_new_feature.py
   import pytest
   from httpx import AsyncClient
   
   @pytest.mark.asyncio
   async def test_get_feature_data(client: AsyncClient, auth_headers):
       response = await client.get("/api/v1/new-feature/", headers=auth_headers)
       assert response.status_code == 200
   ```

### Authentication in Endpoints

All protected endpoints should use Firebase token verification:

```python
from fastapi import Depends
from src.clarity.auth import verify_firebase_token

@router.get("/protected-endpoint")
async def protected_endpoint(
    current_user: dict = Depends(verify_firebase_token)
):
    user_id = current_user["uid"]
    # User is authenticated and user_id is available
```

## Machine Learning Development

### Adding New ML Models

1. **Add Model File**

   ```
   models/
   └── your_model/
       ├── model_weights.h5
       ├── config.json
       └── README.md
   ```

2. **Create Processor**

   ```python
   # src/clarity/ml/processors/your_model.py
   class YourModelProcessor:
       def __init__(self):
           # Load model
           pass
       
       async def process(self, data):
           # Model inference
           pass
   ```

3. **Add Service Integration**

   ```python
   # src/clarity/services/ai/your_model_service.py
   from src.clarity.ml.processors.your_model import YourModelProcessor
   
   class YourModelService:
       def __init__(self):
           self.processor = YourModelProcessor()
   ```

### PAT Model Development

The Pretrained Actigraphy Transformer is already integrated:

- **Model Location**: `models/pat/PAT-M_29k_weights.h5`
- **Processor**: `src/clarity/ml/processors/pat_processor.py`
- **API**: `src/clarity/api/v1/pat.py`
- **Tests**: `tests/ml/test_pat_processor.py` (89% coverage)

## Environment Variables

Required environment variables:

```bash
# Firebase Configuration
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
FIREBASE_PROJECT_ID=your-firebase-project

# Google Cloud
GOOGLE_CLOUD_PROJECT=your-gcp-project
GOOGLE_CLOUD_STORAGE_BUCKET=your-storage-bucket

# AI Services
GOOGLE_AI_API_KEY=your-gemini-api-key

# Application
ENVIRONMENT=development
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
```

## Debugging

### Local Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debugger
python -m debugpy --listen 5678 --wait-for-client main.py
```

### API Debugging

- **FastAPI Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Metrics**: `http://localhost:8000/metrics`

### Database Debugging

```python
# Enable Firestore debug logging
import logging
logging.getLogger('google.cloud.firestore').setLevel(logging.DEBUG)
```

## Performance Optimization

### Current Performance

- ✅ **FastAPI**: High-performance async framework
- ✅ **Async Processing**: Pub/Sub for heavy operations
- ✅ **Model Caching**: ML models loaded once
- ⚠️ **Monitoring**: Basic metrics available

### Optimization Areas

1. **API Response Times**: Currently adequate, can improve with caching
2. **ML Inference**: Batch processing for multiple requests
3. **Database Queries**: Add query optimization
4. **Memory Usage**: Monitor model memory footprint

## Deployment

### Local Testing

```bash
# Build and test Docker image
docker build -t clarity-backend .
docker run -p 8000:8000 clarity-backend
```

### Production Deployment

```bash
# Deploy to Google Cloud Run
gcloud run deploy clarity-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Documentation

### API Documentation

- **Auto-generated**: Available at `/docs` endpoint
- **Manual Docs**: `docs/api/` directory
- **Keep Updated**: Documentation should match implementation

### Code Documentation

- Use comprehensive docstrings
- Include type hints
- Document complex business logic
- Update README files when adding features

## Troubleshooting

### Common Issues

**Import Errors:**

```bash
# Ensure PYTHONPATH includes src/
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

**Firebase Auth Issues:**

```bash
# Verify service account credentials
gcloud auth application-default login
```

**Model Loading Issues:**

```bash
# Check model file exists
ls -la models/pat/PAT-M_29k_weights.h5

# Verify TensorFlow installation
python -c "import tensorflow; print(tensorflow.__version__)"
```

### Getting Help

1. **Check Documentation**: Start with this guide and API docs
2. **Review Tests**: Tests serve as usage examples
3. **Check Logs**: Application logs provide detailed error information
4. **Use Debugger**: Step through code to understand behavior
