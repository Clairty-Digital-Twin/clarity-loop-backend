# API Documentation

**UPDATED:** December 6, 2025 - **NOW ACCURATE** based on actual codebase implementation

This directory contains API documentation for the CLARITY Digital Twin Platform backend.

## 🎯 Base URL

- **Development**: `http://localhost:8000`
- **Production**: Your deployed Cloud Run URL

## 🔐 Authentication

All protected endpoints require Firebase JWT token:
```
Authorization: Bearer <firebase-jwt-token>
```

## 📋 Available API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login

### Health Data  
- `POST /api/v1/health-data/upload` - Upload health metrics
- `GET /api/v1/health-data/` - List health data (paginated)
- `GET /api/v1/health-data/processing/{id}` - Check processing status
- `DELETE /api/v1/health-data/{id}` - Delete health data

### AI Insights
- `POST /api/v1/insights/generate` - Generate AI health insights
- `GET /api/v1/insights/{insight_id}` - Get cached insight

### PAT Analysis
- `GET /api/v1/pat/analyze` - PAT model analysis
- `POST /api/v1/pat/batch-analyze` - Batch PAT analysis

### System
- `GET /health` - Root health check
- `GET /metrics` - Prometheus metrics

## 📖 Interactive Documentation

**FastAPI automatically generates interactive API documentation:**

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These are always up-to-date with the actual implementation.

## 📝 Detailed Endpoint Documentation

See individual files in this directory:

- [authentication.md](./authentication.md) - Auth endpoints
- [health-data.md](./health-data.md) - Health data management
- [insights.md](./insights.md) - AI insights generation

## ⚠️ Important Notes

1. **URL Prefix**: All endpoints use `/api/v1/` prefix
2. **Real Implementation**: This documentation matches the actual code
3. **Auto-Generated Docs**: Use `/docs` endpoint for most current specs
4. **Testing**: All endpoints have corresponding tests in `tests/api/`