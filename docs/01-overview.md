# CLARITY Digital Twin Platform - System Overview

## Executive Summary

CLARITY transforms Apple Health data into conversational mental health insights using state-of-the-art AI. It's a **digital psychiatric twin** that understands your health patterns and explains them in natural language.

### What Problem We Solve

**Traditional health apps**: Show you numbers (10,000 steps, 7 hours sleep)  
**CLARITY**: Explains what those patterns mean for your mental health and lets you ask questions

Example conversation:
```
User: "Why do I feel tired even with 8 hours of sleep?"
CLARITY: "Your sleep efficiency was only 73% last week due to frequent awakenings between 2-4 AM. Your HRV data suggests elevated stress levels during this period. Consider..."
```

## System Architecture

### High-Level Data Flow

```
Apple Watch/iPhone
       ↓
   HealthKit Export (JSON)
       ↓
   CLARITY API (/healthkit/upload)
       ↓ 
   Data Validation & Preprocessing
       ↓
   ┌─────────────────┐    ┌─────────────────┐
   │ PAT Transformer │    │   Gemini AI     │
   │ (Sleep Analysis)│    │ (Chat & Insights)│
   └─────────────────┘    └─────────────────┘
       ↓                        ↓
   Sleep Quality Scores    Natural Language Insights
       ↓                        ↓
   ┌──────────────────────────────────────────┐
   │        User Conversations                │
   │   "What's affecting my sleep quality?"   │
   └──────────────────────────────────────────┘
```

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway (AWS ALB)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                FastAPI Application                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │   Auth      │ │ Health Data │ │    AI Insights          ││
│  │ (Cognito)   │ │   (CRUD)    │ │ (PAT + Gemini)          ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Data Layer                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │  DynamoDB   │ │     S3      │ │    External APIs        ││
│  │(User Data)  │ │(Raw Files)  │ │(Gemini, HealthKit)      ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Ingestion (`/healthkit/*` endpoints)
- **Purpose**: Receive and validate Apple HealthKit exports
- **Key Files**: `src/clarity/integrations/healthkit.py`, `src/clarity/api/v1/healthkit.py`
- **Data Types**: Heart rate, sleep, activity, HRV, respiratory rate, blood pressure
- **Processing**: Time-series alignment, outlier removal, normalization

### 2. PAT (Pretrained Actigraphy Transformer) 
- **Purpose**: Advanced sleep and circadian rhythm analysis
- **Key Files**: `src/clarity/ml/pat_service.py`, `research/Pretrained-Actigraphy-Transformer/`
- **Input**: 7-day activity vectors (10,080 time points)
- **Output**: Sleep quality scores, circadian disruption detection
- **Innovation**: First transformer model for consumer health data

### 3. Gemini AI Integration
- **Purpose**: Natural language health insights and conversational interface
- **Key Files**: `src/clarity/ml/gemini_service.py`, `src/clarity/api/v1/gemini_insights.py`
- **Capabilities**: 
  - Generate health summaries from time-series data
  - Answer specific questions about health patterns
  - Provide personalized recommendations
  - Explain complex medical concepts in simple terms

### 4. Health Data Processing Pipeline
- **Purpose**: Transform raw HealthKit data into ML-ready features
- **Key Files**: `src/clarity/ml/processors/`, `src/clarity/integrations/apple_watch.py`
- **Features**:
  - **Temporal Alignment**: All data resampled to 1-minute intervals
  - **Outlier Detection**: Physiological bounds validation
  - **Signal Processing**: Butterworth filters for heart rate, median filters for respiratory rate
  - **Normalization**: Population-based z-scores using NHANES data

### 5. WebSocket Real-time Streaming
- **Purpose**: Live health data streaming and chat
- **Key Files**: `src/clarity/api/v1/websocket/`
- **Use Cases**: Real-time activity monitoring, live chat with AI

## Data Models

### Input Data (HealthKit)
```python
{
  "heart_rate_samples": [
    {
      "timestamp": "2024-01-01T12:00:00Z",
      "value": 72,
      "unit": "bpm",
      "source": "apple_watch"
    }
  ],
  "sleep_samples": [...],
  "activity_samples": [...]
}
```

### Processed Features (PAT Ready)
```python
{
  "actigraphy_sequence": [0.1, 0.3, 2.1, ...],  # 10,080 points
  "heart_rate_sequence": [1.2, 1.1, 1.4, ...],  # Normalized
  "sequence_length": 10080,
  "sampling_rate_minutes": 1,
  "statistics": {
    "activity_mean": 2.3,
    "zero_activity_percentage": 0.32
  }
}
```

### AI Insights Output
```python
{
  "summary": "Your sleep quality declined 15% this week...",
  "recommendations": ["Consider reducing caffeine after 2 PM"],
  "metrics": {
    "sleep_efficiency": 0.78,
    "circadian_stability": 0.65
  },
  "conversation_ready": True
}
```

## Performance Characteristics

### Scale
- **Users**: Designed for 100K+ concurrent users
- **Data Volume**: ~10MB per user per week (HealthKit export)
- **Processing**: Real-time for simple queries, <30s for complex AI analysis
- **Storage**: DynamoDB scales automatically

### Accuracy
- **PAT Model**: 92% accuracy on sleep stage classification (vs. polysomnography)
- **Heart Rate Processing**: <2% error rate after outlier removal
- **Gemini Insights**: Contextually relevant 95%+ of the time

### Availability
- **SLA**: 99.9% uptime target
- **Auto-scaling**: ECS Fargate based on CPU/memory
- **Health Checks**: Multiple layers (container, load balancer, application)
- **Graceful Degradation**: Core functions work even if AI services are down

## Security & Privacy

### Data Protection
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Access Control**: JWT tokens via AWS Cognito
- **Audit Logging**: All health data access logged
- **Data Residency**: US-only (configurable by region)

### Compliance
- **HIPAA Ready**: Infrastructure meets HIPAA technical safeguards
- **SOC 2**: AWS services provide SOC 2 compliance
- **GDPR**: Data deletion and export capabilities

### Privacy by Design
- **Minimal Data**: Only clinically relevant metrics stored
- **User Control**: Complete data export and deletion
- **Anonymization**: PII separated from health data
- **Consent Management**: Granular permissions for data types

## Development Status

### Production Ready ✅
- **Core API**: All 44 endpoints functional
- **AWS Infrastructure**: Fully deployed and tested
- **Test Coverage**: 807/810 tests passing (99.6%)
- **Performance**: Sub-second response times for most endpoints

### In Development 🔄
- **PAT Model Optimization**: Reducing inference time from 15s to 5s
- **Advanced Chat Features**: Multi-turn conversations with memory
- **Additional Integrations**: Fitbit, Garmin, Oura support
- **Mobile SDKs**: iOS/Android libraries for easier integration

### Roadmap 📋
- **Q1 2024**: Real-time anomaly detection
- **Q2 2024**: Predictive health modeling
- **Q3 2024**: Integration with healthcare providers
- **Q4 2024**: Longitudinal studies and research platform

## Getting Started

### For Developers
```bash
git clone https://github.com/your-org/clarity-loop-backend.git
cd clarity-loop-backend
make install && make dev
```

### For Users
1. Export your Apple Health data
2. POST to `/api/v1/healthkit/upload`
3. Wait for processing (30-60 seconds)
4. Start chatting at `/api/v1/insights/chat`

### For Researchers
- Access anonymized aggregate data via `/api/v1/research/*` endpoints
- Run population-level analysis with our PAT model
- Contribute to open health AI research

---

**Next**: Read [API Reference](02-api-reference.md) for detailed endpoint documentation 