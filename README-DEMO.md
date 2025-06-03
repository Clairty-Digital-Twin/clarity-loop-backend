# CLARITY Digital Twin Platform - Demo

> **AI-powered psychiatric health analytics that actually works**

A legit health platform that ingests Apple HealthKit data and runs it through some serious AI models to generate clinical insights. Built with proper architecture, not just another hackathon project.

## What This Actually Does

**Real Health AI Pipeline:**

- Pulls biometric data from Apple HealthKit (heart rate, sleep, steps, etc.)
- Runs it through PAT (Pretrained Actigraphy Transformer) for sleep analysis
- Generates natural language insights with Google Gemini
- Stores everything securely with HIPAA compliance

**Tech Stack That Doesn't Suck:**

- FastAPI with async everything
- PyTorch for ML inference
- Google Cloud with proper security
- Clean Architecture (no spaghetti code)
- 100% type safety (zero MyPy errors)

## Quick Demo

### Start Everything

```bash
git clone <repository>
cd clarity-loop-backend
./scripts/demo_deployment.sh
```

### What You Get

- **API**: <http://localhost:8080> - Main backend
- **Docs**: <http://localhost:8080/docs> - Interactive API explorer
- **Monitoring**: <http://localhost:9090> - Prometheus metrics
- **Dashboards**: <http://localhost:3000> - Grafana (admin/admin)

### Test the AI

```bash
# Health insights from Gemini
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

# Sleep analysis with PAT model
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
│  ├─ Auth (Firebase)                     │
│  ├─ Health Data API                     │
│  ├─ PAT Analysis                        │
│  └─ Gemini Insights                     │
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

**AI Models:**

- **PAT**: [Pretrained Actigraphy Transformer](https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer) from Jacobson Lab (Dartmouth) for sleep analysis
- **Gemini**: Google's LLM for generating health insights from data

**Real Health Data:**

- Apple HealthKit integration for heart rate, HRV, steps, sleep
- Proxy actigraphy conversion (turns step data into sleep analysis)
- HIPAA-compliant storage and processing

**Production Ready:**

- Async FastAPI with proper error handling
- Firebase auth with role-based permissions
- Monitoring with Prometheus/Grafana
- Docker containerization
- Comprehensive test suite

## Demo Flow

1. **Health Checks**: Verify all services are running
2. **Upload Data**: Send HealthKit data via API
3. **PAT Analysis**: Get sleep insights from step data
4. **Gemini Insights**: Generate natural language health summaries
5. **Monitor**: Check metrics and performance

## Tech Highlights

- **Type Safety**: Zero MyPy errors across the entire codebase
- **Clean Architecture**: Proper dependency injection and SOLID principles
- **ML Integration**: Real AI models, not just mock responses
- **Security**: Firebase auth, encrypted storage, audit logging
- **Observability**: Metrics, health checks, structured logging

## Test Suite

```bash
python scripts/api_test_suite.py
```

Runs comprehensive tests against all endpoints to verify everything works.

---

## 🙏 **Research Attribution**

This platform leverages the **Pretrained Actigraphy Transformer (PAT)** open-source foundation model:

**Paper:** Ruan, F.Y., Zhang, A., Oh, J., Jin, S., & Jacobson, N.C. (2024). "AI Foundation Models for Wearable Movement Data in Mental Health Research." *arXiv:2411.15240*. <https://doi.org/10.48550/arXiv.2411.15240>

**Repository:** <https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer> (CC-BY-4.0 License)

---

**Built for advancing psychiatric care through practical AI applications**
