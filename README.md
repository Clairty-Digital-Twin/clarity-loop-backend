# CLARITY Digital Twin Platform

> **Your Apple Health data becomes a conversation with an AI psychiatrist**

Transform wearable data into actionable mental health insights using state-of-the-art AI. CLARITY processes Apple Watch data through advanced transformer models to create a "digital twin" of your health patterns.

## What This Does

**Input**: Your Apple Health data (heart rate, sleep, activity, etc.)  
**Processing**: Pretrained Actigraphy Transformer (PAT) + Google Gemini AI  
**Output**: Personalized mental health insights you can chat with  

```
Apple Watch → HealthKit Data → AI Analysis → Conversational Health Insights
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Apple Health   │───▶│   FastAPI       │───▶│  AWS DynamoDB   │
│  Data Upload    │    │   Backend       │    │  Storage        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
              ┌────────▼────────┐ ┌──────▼──────┐
              │ PAT Transformer │ │ Gemini AI   │
              │ Sleep/Activity  │ │ Chat Engine │
              └─────────────────┘ └─────────────┘
```

**Tech Stack**: Python FastAPI, AWS (DynamoDB/Cognito/ECS), Google Gemini, PyTorch Transformers

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/clarity-loop-backend.git
cd clarity-loop-backend
make install

# Local development
cp .env.example .env  # Configure your API keys
make dev

# Test it works
curl http://localhost:8000/health
```

## API Endpoints

### Core Flow
```bash
# 1. Upload Apple Health data
POST /api/v1/healthkit/upload
{
  "data": [/* HealthKit JSON export */]
}

# 2. Get AI analysis
POST /api/v1/insights/generate
{
  "user_id": "123",
  "type": "mental_health"
}

# 3. Chat about your health
POST /api/v1/insights/chat
{
  "message": "Why am I sleeping poorly?",
  "context": "sleep_patterns"
}
```

### All Endpoints (44 total)
- **Auth**: `/api/v1/auth/*` (7 endpoints) - Cognito-based authentication
- **Health Data**: `/api/v1/health-data/*` (10 endpoints) - Raw data management  
- **HealthKit**: `/api/v1/healthkit/*` (4 endpoints) - Apple integration
- **AI Insights**: `/api/v1/insights/*` (6 endpoints) - Gemini chat & analysis
- **PAT Analysis**: `/api/v1/pat/*` (5 endpoints) - Transformer model inference
- **WebSocket**: `/api/v1/ws/*` (3 endpoints) - Real-time connections
- **Monitoring**: `/api/v1/metrics/*` + `/health` (5 endpoints) - System health

## Data Pipeline

**Apple Health → Preprocessing → AI Models → Insights**

1. **Data Ingestion**: HealthKit JSON → Validated schemas
2. **Time Series Processing**: 1-minute intervals, outlier removal, normalization  
3. **PAT Analysis**: Activity patterns → Sleep quality predictions
4. **Gemini Insights**: Health data + context → Natural language explanations
5. **User Interface**: Chat-based Q&A about your health trends

### Supported Health Metrics
- **Cardiovascular**: Heart rate, HRV, blood pressure, ECG
- **Sleep**: Stages, efficiency, disturbances  
- **Activity**: Steps, calories, exercise, movement patterns
- **Mental Health**: Mood, stress, energy levels
- **Respiratory**: Breathing rate, SpO2

## AI Models

### PAT (Pretrained Actigraphy Transformer)
- **Purpose**: Sleep and circadian rhythm analysis from movement data
- **Input**: 7-day activity vectors (10,080 time points)  
- **Output**: Sleep quality scores, circadian disruption detection
- **Architecture**: Transformer with positional encoding for temporal patterns

### Google Gemini Integration  
- **Purpose**: Natural language health insights and chat
- **Context**: Your processed health data + conversation history
- **Capabilities**: Trend analysis, recommendations, Q&A about patterns

## Production Deployment

**AWS ECS Fargate** with auto-scaling, health checks, zero-downtime deployments

```bash
# Deploy to AWS
./deploy.sh production

# Monitor
aws logs tail /aws/ecs/clarity-backend --follow
```

**Infrastructure**: 
- **Compute**: ECS Fargate containers
- **Database**: DynamoDB (health data) + Cognito (auth)
- **Storage**: S3 (raw uploads) 
- **Monitoring**: CloudWatch + Prometheus metrics

## Development

```bash
# Setup
make install          # Install dependencies
make test            # Run tests (807/810 passing)
make lint            # Code quality checks
make typecheck       # MyPy validation

# Local services
make dev             # Run with hot reload
make test-integration  # Test against real AWS services
```

## Security & Compliance

- **HIPAA-ready infrastructure** on AWS
- **End-to-end encryption** for all health data
- **JWT authentication** via AWS Cognito
- **Audit logging** for all data access
- **Role-based permissions** for API access

## Status

✅ **Backend Complete**: 807/810 tests passing (99.6%)  
✅ **AWS Migration**: Fully deployed on production infrastructure  
⚠️ **Test Coverage**: 56% (targeting 85% for production)  
🔄 **Active Development**: Adding new AI models and chat features

## What Makes This Different

This isn't just another health app. CLARITY creates a **digital psychiatric twin** by:

1. **Deep Analysis**: Beyond step counting - analyzing circadian rhythms, sleep architecture, HRV patterns
2. **Conversational AI**: Chat with your health data using Gemini's language understanding  
3. **Temporal Intelligence**: PAT model understands time-series patterns humans miss
4. **Clinical Relevance**: Metrics that matter for mental health, not just fitness

## Contributing

Built for developers who want to push the boundaries of health AI.

```bash
git checkout -b feature/amazing-ai-model
# Build something incredible
git commit -m "feat: revolutionary health insight algorithm"
```

## License

MIT License - Build the future of health AI

---

**CLARITY Digital Twin Platform** - Making Apple Health data intelligent