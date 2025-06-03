# CLARITY Digital Twin Platform - Demo Guide

A digital twin platform for psychiatric care that processes Apple HealthKit data and generates AI-powered insights using machine learning models.

## Overview

CLARITY integrates real health data processing with advanced AI models to provide clinical insights for psychiatric care. The platform uses the Pretrained Actigraphy Transformer (PAT) for sleep analysis and Google Gemini for generating natural language health insights.

**Core Components:**
- FastAPI backend with async processing
- PyTorch ML inference pipeline
- Google Cloud infrastructure
- Clean Architecture implementation
- Comprehensive monitoring and observability

## Quick Demo Setup

### Start the Platform

```bash
git clone <repository>
cd clarity-loop-backend
./scripts/demo_deployment.sh
```

### Available Services

| Service | URL | Purpose |
|---------|-----|---------|
| API Backend | http://localhost:8080 | Main FastAPI application |
| API Documentation | http://localhost:8080/docs | Interactive OpenAPI explorer |
| Grafana | http://localhost:3000 | Monitoring dashboards (admin/admin) |
| Prometheus | http://localhost:9090 | Metrics collection |

### Test the API Endpoints

```bash
# Health check
curl http://localhost:8080/health

# Test AI insights generation
curl -X POST http://localhost:8080/api/v1/insights/generate \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_results": {
      "heart_rate_avg": 72,
      "sleep_efficiency": 0.85,
      "circadian_score": 0.72
    },
    "question": "How is my sleep affecting my health?"
  }'

# Test PAT model sleep analysis
curl -X POST http://localhost:8080/api/v1/pat/analyze-step-data \
  -H "Content-Type: application/json" \
  -d '{
    "step_counts": [0, 0, 0, 5, 12, 8, 0, 0],
    "timestamps": ["2025-01-15T22:00:00Z", "2025-01-15T23:00:00Z"]
  }'
```

## Architecture Overview

```
┌─────────────────────────────────────────┐
│  FastAPI Backend                        │
│  ├─ Authentication (Firebase)           │
│  ├─ Health Data API                     │
│  ├─ PAT Sleep Analysis                  │
│  └─ Gemini Insights Generation          │
├─────────────────────────────────────────┤
│  AI/ML Pipeline                         │
│  ├─ PAT (Sleep Analysis)                │
│  └─ Gemini (Health Insights)            │
├─────────────────────────────────────────┤
│  Data Layer                             │
│  ├─ Firestore (Health Records)          │
│  └─ Apple HealthKit Integration         │
├─────────────────────────────────────────┤
│  Monitoring                             │
│  ├─ Prometheus + Grafana                │
│  └─ Health Checks                       │
└─────────────────────────────────────────┘
```

## Key Features

**AI Models**
- **PAT**: Pretrained Actigraphy Transformer from Jacobson Lab (Dartmouth) for sleep analysis
- **Gemini**: Google's language model for generating health insights from data

**Health Data Processing**
- Apple HealthKit integration for biometric data (heart rate, HRV, steps, sleep)
- Proxy actigraphy conversion (converts step data into sleep analysis)
- HIPAA-compliant storage and processing

**Production Features**
- Async FastAPI with comprehensive error handling
- Firebase authentication with role-based permissions
- Monitoring with Prometheus and Grafana
- Docker containerization
- Comprehensive test suite with type safety

## Demo Workflow

1. **Health Checks**: Verify all services are operational
2. **Data Upload**: Submit HealthKit data via API endpoints
3. **PAT Analysis**: Process step data for sleep insights
4. **Gemini Insights**: Generate natural language health summaries
5. **Monitoring**: Review metrics and system performance

## Technical Highlights

- **Type Safety**: Full MyPy compliance across the codebase
- **Clean Architecture**: SOLID principles with proper dependency injection
- **ML Integration**: Production-ready AI models with real inference
- **Security**: Firebase authentication, encrypted storage, audit logging
- **Observability**: Comprehensive metrics, health checks, and structured logging

## Test Suite

```bash
# Run the comprehensive API test suite
python scripts/api_test_suite.py
```

This validates all endpoints and verifies the platform functionality.

## Research Attribution

This platform leverages the Pretrained Actigraphy Transformer (PAT) open-source foundation model:

**Paper:** Ruan, F.Y., Zhang, A., Oh, J., Jin, S., & Jacobson, N.C. (2024). "AI Foundation Models for Wearable Movement Data in Mental Health Research." *arXiv:2411.15240*. https://doi.org/10.48550/arXiv.2411.15240

**Repository:** https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer (CC-BY-4.0 License)