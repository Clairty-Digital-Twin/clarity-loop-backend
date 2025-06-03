# CLARITY Quick Start Guide

**UPDATED:** December 6, 2025 - Based on actual implementation

Get the CLARITY Digital Twin Platform running locally in under 10 minutes.

## 🚀 Prerequisites

- **Python 3.12+** installed
- **Docker & Docker Compose** installed  
- **Git** installed
- **Google Cloud account** (free tier works)
- **Firebase project** (free tier works)

## ⚡ 5-Minute Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd clarity-loop-backend
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings:
# FIREBASE_PROJECT_ID=your-firebase-project
# GOOGLE_CLOUD_PROJECT=your-gcp-project  
# GOOGLE_AI_API_KEY=your-gemini-api-key
```

### 4. Start Application

```bash
# Option A: Docker (Recommended)
docker-compose up -d

# Option B: Local Python
python main.py
```

### 5. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

## 🎯 First API Calls

### Register a User

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "securepassword123"
  }'
```

### Login and Get Token

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com", 
    "password": "securepassword123"
  }'
```

### Upload Health Data

```bash
# Use token from login response
TOKEN="your-jwt-token-here"

curl -X POST "http://localhost:8000/api/v1/health-data/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "firebase-uid-123",
    "metrics": [
      {
        "type": "heart_rate",
        "value": 72.5,
        "unit": "bpm",
        "timestamp": "2025-01-15T10:30:00Z",
        "source": "apple_watch"
      }
    ],
    "upload_source": "api_test"
  }'
```

### Generate AI Insights

```bash
curl -X POST "http://localhost:8000/api/v1/insights/generate" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "firebase-uid-123",
    "data_sources": ["recent_health_data"],
    "analysis_type": "comprehensive"
  }'
```

## 🔍 Interactive API Testing

**Best way to explore the API:**

1. **Open Swagger UI**: `http://localhost:8000/docs`
2. **Click "Authorize"** button
3. **Enter your JWT token** from login
4. **Try out endpoints** interactively

## 📊 Verify ML Models

### Check PAT Model

```bash
curl "http://localhost:8000/api/v1/pat/analyze"
```

Should return PAT model info and capabilities.

### Test AI Insights

Use the insights endpoint above - it integrates:

- PAT model for sleep analysis
- Gemini AI for natural language insights
- Real health data processing

## 🧪 Run Tests

```bash
# Full test suite (729 tests)
pytest

# With coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Specific component tests
pytest tests/api/v1/test_health_data.py
pytest tests/ml/test_pat_processor.py
```

## 🔧 Development Setup

### Code Quality Tools

```bash
# Linting
ruff check .

# Type checking  
mypy src/

# Auto-formatting
ruff check --fix .
```

### Database Setup (Local Development)

```bash
# Start Firestore emulator
gcloud emulators firestore start

# In another terminal, set emulator environment
export FIRESTORE_EMULATOR_HOST=localhost:8080
python main.py
```

## 📱 What's Working

### ✅ Core Features

- **Authentication**: Firebase-based user management
- **Health Data**: Upload, storage, retrieval with pagination
- **PAT Analysis**: Real Dartmouth weights, 89% test coverage
- **AI Insights**: Gemini 2.5 Pro integration, natural language outputs
- **Async Processing**: Background analysis with Pub/Sub
- **Real-time Updates**: Firestore listeners

### ✅ Production Ready

- **729 tests passing** (59.28% coverage)
- **Clean architecture** with dependency injection
- **Type safety** with comprehensive mypy checking
- **Security** with Firebase Auth + JWT tokens
- **Scalability** designed for Google Cloud deployment

### ⚠️ Areas for Improvement

- **Test coverage**: 59% → 85% target (main focus area)
- **Error handling**: Some edge cases need more coverage
- **Documentation**: Some API docs were outdated (now fixed)

## 🚀 Next Steps

### For Development

1. **Increase test coverage** - focus on API error scenarios
2. **Add monitoring** - enhance metrics and alerting  
3. **Performance tuning** - optimize ML inference times
4. **New features** - sleep processor module

### For Production

1. **Set up Google Cloud project**
2. **Configure Firebase project**
3. **Deploy to Cloud Run**
4. **Set up monitoring dashboard**

## 🆘 Troubleshooting

### Common Issues

**Import Errors:**

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

**Firebase Authentication Issues:**

```bash
# Verify credentials
gcloud auth application-default login

# Check service account file
ls -la path/to/service-account.json
```

**Model Loading Issues:**

```bash
# Verify PAT model exists
ls -la models/pat/PAT-M_29k_weights.h5

# Check TensorFlow
python -c "import tensorflow; print(tensorflow.__version__)"
```

**Port Already in Use:**

```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use different port
API_PORT=8001 python main.py
```

## 📚 Learn More

- **API Documentation**: `http://localhost:8000/docs`
- **Architecture Guide**: `docs/architecture/README.md`
- **Development Guide**: `docs/development/README.md`
- **Deployment Guide**: `docs/development/deployment.md`

## 💡 Pro Tips

1. **Use Docker Compose** for consistent development environment
2. **Check `/health` endpoint** to verify all services are running
3. **Monitor logs** with `docker-compose logs -f` for debugging
4. **Use Swagger UI** for interactive API testing
5. **Run tests frequently** to catch issues early

Ready to build the future of digital health! 🏥✨
