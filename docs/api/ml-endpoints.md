# ML API Endpoints

Comprehensive API documentation for machine learning endpoints in the Clarity Loop Backend, featuring Pretrained Actigraphy Transformer (PAT) integration and AI-powered health insights.

## Model Weights & Deployment

### PAT Model Weights Location

The Pretrained Actigraphy Transformer models are stored in the project's `/models` directory:

```
models/
├── PAT-L_29k_weights.h5    # Large model (7.6MB) - Best accuracy
├── PAT-M_29k_weights.h5    # Medium model (3.8MB) - Balanced performance
└── PAT-S_29k_weights.h5    # Small model (1.1MB) - Fastest inference
```

### Model Selection Criteria

| Model Size | Parameters | Inference Time | Best Use Case |
|------------|------------|----------------|---------------|
| **PAT-L**  | 1.99M      | ~800ms        | Research, batch processing |
| **PAT-M**  | 1.00M      | ~450ms        | Production default |
| **PAT-S**  | 285K       | ~200ms        | Real-time, mobile |

### Deployment Pipeline

1. **Development**: Models loaded from `/models/*.h5` files
2. **Production**: Models deployed to Google Cloud AI Platform
3. **Fallback**: Local model serving via TensorFlow Serving

**Note**: The models in `/models/` are the official Dartmouth College weights (29,307 participants) as referenced in the PAT research paper (arXiv:2411.15240).

## Overview

The ML API provides endpoints for health data analysis, actigraphy feature extraction, and AI-powered insight generation. All endpoints follow RESTful principles and support both synchronous and asynchronous processing patterns.

## Authentication

All ML endpoints require Firebase Authentication tokens with appropriate health data permissions.

```http
Authorization: Bearer <firebase_id_token>
X-User-Role: patient|clinician|researcher
```

## Base URL

```
Production: https://api.clarity-loop.com/v1/ml
Development: https://dev-api.clarity-loop.com/v1/ml
Local: http://localhost:8000/v1/ml
```

## Endpoints

### 1. Health Data Analysis

#### POST /analyze/actigraphy

Analyzes health data using the Pretrained Actigraphy Transformer to extract meaningful features and patterns.

**Request Body:**

```json
{
  "user_id": "string",
  "data_type": "steps|heart_rate|sleep|activity",
  "timeframe": "1week|1month|3months",
  "data": {
    "values": [number],
    "timestamps": ["ISO8601"],
    "source": "apple_watch|iphone|manual",
    "metadata": {
      "device_model": "string",
      "os_version": "string",
      "app_version": "string"
    }
  },
  "analysis_options": {
    "include_trends": true,
    "include_patterns": true,
    "include_anomalies": true,
    "model_size": "small|medium|large"
  }
}
```

**Response (Success - 200):**

```json
{
  "request_id": "string",
  "user_id": "string",
  "analysis_timestamp": "ISO8601",
  "processing_time_ms": 450,
  "features": {
    "sleep_efficiency": 0.85,
    "circadian_rhythm_strength": 0.72,
    "activity_fragmentation": 0.23,
    "rest_activity_ratio": 2.1,
    "sleep_onset_variability": 0.4,
    "wake_after_sleep_onset": 12.5
  },
  "patterns": {
    "sleep_patterns": {
      "avg_sleep_duration": 7.2,
      "sleep_onset_time": "23:15",
      "wake_time": "06:30",
      "sleep_consistency_score": 0.78
    },
    "activity_patterns": {
      "daily_step_avg": 8542,
      "activity_peaks": ["09:00", "17:30"],
      "sedentary_periods": 4.2,
      "activity_consistency": 0.65
    },
    "circadian_patterns": {
      "rhythm_amplitude": 0.68,
      "phase_shift_minutes": -15,
      "stability_score": 0.82
    }
  },
  "trends": {
    "sleep_trend": "improving",
    "activity_trend": "stable",
    "circadian_trend": "declining",
    "confidence_scores": {
      "sleep": 0.89,
      "activity": 0.76,
      "circadian": 0.83
    }
  },
  "anomalies": [
    {
      "type": "sleep_disruption",
      "timestamp": "ISO8601",
      "severity": "moderate",
      "description": "Unusual sleep fragmentation detected",
      "confidence": 0.78
    }
  ],
  "metadata": {
    "model_version": "pat-v1.2.0",
    "model_size": "medium",
    "data_quality_score": 0.92,
    "processing_node": "us-central1-ml-001"
  }
}
```

**Response (Processing - 202):**

```json
{
  "request_id": "string",
  "status": "processing",
  "estimated_completion": "ISO8601",
  "progress_url": "/v1/ml/status/{request_id}"
}
```

**Error Responses:**

```json
// 400 Bad Request
{
  "error": "validation_error",
  "message": "Invalid data format or missing required fields",
  "details": {
    "field": "data.timestamps",
    "issue": "Timestamps must be in ISO8601 format"
  }
}

// 429 Too Many Requests
{
  "error": "rate_limit_exceeded",
  "message": "ML analysis rate limit exceeded",
  "retry_after": 300,
  "current_limit": "10 requests per minute"
}
```

#### GET /analyze/status/{request_id}

Check the status of an asynchronous analysis request.

**Response:**

```json
{
  "request_id": "string",
  "status": "queued|processing|completed|failed",
  "progress_percentage": 75,
  "stage": "feature_extraction|ai_processing|insight_generation",
  "estimated_completion": "ISO8601",
  "result_url": "/v1/ml/results/{request_id}",
  "created_at": "ISO8601",
  "updated_at": "ISO8601"
}
```

### 2. AI Insights Generation

#### POST /insights/generate

Generate natural language health insights using Gemini AI based on actigraphy features.

**Request Body:**

```json
{
  "user_id": "string",
  "insight_type": "general|sleep|activity|circadian|personalized",
  "context": {
    "time_period": "1week|1month|3months",
    "user_goals": ["better_sleep", "more_activity", "stress_reduction"],
    "health_conditions": ["string"],
    "lifestyle_factors": {
      "work_schedule": "regular|shift|flexible",
      "exercise_frequency": "daily|weekly|occasional|none",
      "stress_level": "low|moderate|high"
    }
  },
  "actigraphy_features": {
    "sleep_efficiency": 0.85,
    "circadian_rhythm_strength": 0.72,
    "activity_fragmentation": 0.23,
    "rest_activity_ratio": 2.1
  },
  "personalization": {
    "tone": "professional|friendly|casual",
    "detail_level": "summary|detailed|comprehensive",
    "include_recommendations": true,
    "language": "en|es|fr|de"
  }
}
```

**Response:**

```json
{
  "insight_id": "string",
  "user_id": "string",
  "generated_at": "ISO8601",
  "insight_type": "general",
  "summary": "Your sleep patterns show excellent consistency with a healthy 85% sleep efficiency. Your circadian rhythm appears well-regulated, suggesting good alignment with natural light cycles.",
  "detailed_analysis": {
    "sleep_insights": {
      "title": "Sleep Quality Assessment",
      "content": "Your sleep efficiency of 85% indicates excellent sleep quality...",
      "key_points": [
        "Consistent 7.2-hour sleep duration",
        "Regular bedtime routine established",
        "Minimal sleep fragmentation"
      ],
      "trend": "improving",
      "confidence": 0.89
    },
    "activity_insights": {
      "title": "Activity Pattern Analysis",
      "content": "Your daily activity shows a healthy pattern with good rest-activity balance...",
      "key_points": [
        "Average 8,542 steps per day",
        "Good activity distribution throughout day",
        "Appropriate sedentary periods"
      ],
      "trend": "stable",
      "confidence": 0.76
    },
    "circadian_insights": {
      "title": "Circadian Rhythm Health",
      "content": "Your circadian rhythm strength of 72% suggests good biological clock function...",
      "key_points": [
        "Well-timed activity peaks",
        "Consistent sleep-wake cycle",
        "Good light exposure patterns"
      ],
      "trend": "stable",
      "confidence": 0.83
    }
  },
  "recommendations": [
    {
      "category": "sleep",
      "priority": "high",
      "title": "Maintain Sleep Consistency",
      "description": "Continue your excellent sleep schedule to maintain high sleep efficiency",
      "actionable_steps": [
        "Keep consistent bedtime and wake time",
        "Maintain current pre-sleep routine",
        "Monitor any changes in sleep quality"
      ],
      "expected_impact": "high",
      "timeframe": "ongoing"
    },
    {
      "category": "activity",
      "priority": "medium",
      "title": "Optimize Activity Timing",
      "description": "Consider adding brief activity sessions to enhance circadian rhythm",
      "actionable_steps": [
        "Take 10-minute walks after meals",
        "Add morning light exposure",
        "Consider gentle evening stretching"
      ],
      "expected_impact": "medium",
      "timeframe": "2-4 weeks"
    }
  ],
  "health_score": {
    "overall": 82,
    "sleep": 89,
    "activity": 76,
    "circadian": 78,
    "trend": "improving"
  },
  "metadata": {
    "model_version": "gemini-2.5-pro-preview-05-06",
    "processing_time_ms": 1200,
    "language": "en",
    "personalization_applied": true,
    "model_capabilities": {
      "input_types": ["text", "code", "images", "audio", "video"],
      "max_input_tokens": 1048576,
      "max_output_tokens": 65535,
      "features": [
        "grounding_with_google_search",
        "code_execution",
        "system_instructions",
        "controlled_generation",
        "function_calling",
        "context_caching",
        "vertex_ai_rag_engine"
      ]
    }
  }
}
```

### 3. Real-time Monitoring

#### POST /monitor/realtime

Set up real-time monitoring for continuous health data analysis.

**Request Body:**

```json
{
  "user_id": "string",
  "monitoring_config": {
    "data_types": ["steps", "heart_rate", "sleep"],
    "analysis_frequency": "hourly|daily|weekly",
    "alert_thresholds": {
      "sleep_efficiency_min": 0.7,
      "activity_deviation_max": 0.3,
      "circadian_disruption_threshold": 0.5
    },
    "notification_preferences": {
      "push_notifications": true,
      "email_summary": "daily",
      "priority_alerts": true
    }
  }
}
```

**Response:**

```json
{
  "monitoring_id": "string",
  "status": "active",
  "created_at": "ISO8601",
  "next_analysis": "ISO8601",
  "webhook_url": "string",
  "config": {
    "data_types": ["steps", "heart_rate", "sleep"],
    "analysis_frequency": "daily",
    "alert_thresholds": { /* ... */ }
  }
}
```

#### GET /monitor/{monitoring_id}/status

Get real-time monitoring status and recent alerts.

**Response:**

```json
{
  "monitoring_id": "string",
  "status": "active|paused|stopped",
  "last_analysis": "ISO8601",
  "next_analysis": "ISO8601",
  "recent_alerts": [
    {
      "alert_id": "string",
      "type": "sleep_disruption",
      "severity": "moderate",
      "triggered_at": "ISO8601",
      "message": "Sleep efficiency dropped below threshold",
      "acknowledged": false
    }
  ],
  "health_metrics_summary": {
    "last_24h": {
      "sleep_efficiency": 0.75,
      "activity_level": 0.68,
      "circadian_strength": 0.82
    },
    "trend_7d": "declining",
    "anomalies_detected": 2
  }
}
```

### 4. Model Management

#### GET /models/available

List available ML models and their configurations.

**Response:**

```json
{
  "models": [
    {
      "name": "pat-small",
      "version": "1.2.0",
      "description": "Pretrained Actigraphy Transformer - Small",
      "parameters": "1.2M",
      "inference_time_ms": 200,
      "accuracy_metrics": {
        "sleep_detection": 0.87,
        "circadian_analysis": 0.81
      },
      "supported_data_types": ["steps", "activity"],
      "memory_requirements": "256MB",
      "gpu_required": false
    },
    {
      "name": "pat-medium",
      "version": "1.2.0",
      "description": "Pretrained Actigraphy Transformer - Medium",
      "parameters": "3.5M",
      "inference_time_ms": 450,
      "accuracy_metrics": {
        "sleep_detection": 0.92,
        "circadian_analysis": 0.87
      },
      "supported_data_types": ["steps", "heart_rate", "sleep", "activity"],
      "memory_requirements": "512MB",
      "gpu_required": true
    }
  ],
  "default_model": "pat-medium",
  "model_selection_criteria": {
    "accuracy_priority": "pat-large",
    "speed_priority": "pat-small",
    "balanced": "pat-medium"
  }
}
```

#### GET /models/{model_name}/health

Check ML model health and performance metrics.

**Response:**

```json
{
  "model_name": "pat-medium",
  "status": "healthy",
  "last_health_check": "ISO8601",
  "performance_metrics": {
    "avg_latency_ms": 445,
    "throughput_per_minute": 120,
    "error_rate_24h": 0.02,
    "memory_usage_mb": 487,
    "gpu_utilization": 0.65
  },
  "data_drift_status": {
    "detected": false,
    "last_check": "ISO8601",
    "drift_score": 0.12,
    "threshold": 0.3
  },
  "model_accuracy": {
    "recent_validation": {
      "timestamp": "ISO8601",
      "sleep_detection": 0.91,
      "activity_classification": 0.88,
      "circadian_analysis": 0.86
    },
    "trend": "stable"
  }
}
```

### 5. Batch Processing

#### POST /batch/analyze

Submit batch analysis request for multiple users or large datasets.

**Request Body:**

```json
{
  "batch_id": "string",
  "analysis_type": "actigraphy|insights|trends",
  "data_sources": [
    {
      "user_id": "string",
      "data_location": "gs://bucket/path/file.json",
      "data_type": "steps",
      "timeframe": "1month"
    }
  ],
  "analysis_config": {
    "model_size": "medium",
    "include_insights": true,
    "output_format": "json|csv|parquet",
    "notification_webhook": "https://your-app.com/webhook"
  },
  "priority": "low|normal|high"
}
```

**Response:**

```json
{
  "batch_id": "string",
  "status": "queued",
  "estimated_completion": "ISO8601",
  "total_items": 100,
  "progress_url": "/v1/ml/batch/{batch_id}/status",
  "result_location": "gs://clarity-results/batch-{batch_id}/",
  "cost_estimate_usd": 12.50
}
```

## Actigraphy Transformer I/O Contract (PAT-L 2025)

### 1. Input tensor

- **Shape**: `[B, 10 080]` where 10 080 = minutes in 1 week (pp 18-21)
- **Datatype**: `float32` – standardized (z-score) accelerometer magnitude.  
- **Standardisation rule**: z-score _per NHANES year_ **before** patching; during fine-tune, compute μ/σ on **train split only** to prevent leakage (pp 22-24).  
- **Padding / variable length**: If shorter than 10 080, left-pad with value `0` (already zero-mean); if longer, slide a 1-week window with stride = patch size (18 min).

### 2. Patch embedding

- **Patch size S** = 18 min ⇒ **N tokens** = 560 per week (pp 18-21).  
- Linear projection to **D = 128** (identical for S/M/L) – matches source code default.  
- Optionally replace with 1-D conv patchizer; models get suffix "`-conv`" (pp 18-21).

### 3. Backbone variants

| ID | #Layers | D (embed) | MLP hidden | #Heads | Params |
|----|---------|-----------|-----------|--------|--------|
| PAT-S | 3 | 128 | 256 | 4 | **285 K** (p 15) |
| PAT-M | 6 | 128 | 512 | 4 | **1.00 M** (p 15) |
| PAT-L | 12 | 128 | 512 | 8 | **1.99 M** (p 15) |

### 4. Positional encoding

- Fixed sine/cosine tables, added **twice**: once before encoder, once before decoder (pp 18-21).

### 5. Pre-training recipe (masked auto-encoder)

- **Mask ratio** = 0.90, random without replacement (pp 20, 31-36).  
- **Encoder** sees only un-masked tokens.  
- **Decoder** receives encoder tokens **plus** learned `[MASK]` embeddings; 2 transformer layers.  
- **Loss**: MSE over **all 10 080 minutes** – not just masked positions (pp 20, 31-36).  
- **Optimizer**: AdamW, lr = 1e-4, β=(0.9,0.95), weight-decay = 0.05, batch = 256, epochs = 400 (source code default).

### 6. Fine-tuning head

- `nn.Linear(D,1)` + sigmoid.  End-to-end un-frozen (FT); LP offers no gain (see Table 3) (pp 20, 31-36).

### 7. Explainability endpoint `/explain`

- Returns JSON:

```json
{
  "layer": 12,
  "attention": [H, N, N]  // summed-key then averaged over heads
}
```

- Generation rule: extract last transformer block, sum attention over keys per query, average heads, min-max normalise (pp 11-13).

### 8. Performance envelope

- Pre-train PAT-L ≤ 6 h on single A100; inference fits 2 vCPU / 4 GB RAM --concurrency 1 on Cloud Run (p 16).

## Training Do-Nots

> ⚠️ **Critical Training Guidelines** - Avoid these patterns to prevent performance degradation:

- **Don't** lower the mask ratio below 0.75 – AUC drops ≈ 3–6 pp (pp 31-36).
- **Don't** apply Savitzky-Golay smoothing – hurts AUC by ~3 pp (pp 31-36).
- **Don't** compute loss only on masked patches – collapses to AUC ≈ 0.54 (pp 31-36).

## Rate Limits

| Endpoint | Rate Limit | Window |
|----------|------------|---------|
| `/analyze/actigraphy` | 10 requests | 1 minute |
| `/insights/generate` | 5 requests | 1 minute |
| `/monitor/realtime` | 2 requests | 1 minute |
| `/batch/analyze` | 1 request | 5 minutes |

## Error Codes

| Code | Description | Action |
|------|-------------|---------|
| `ML001` | Invalid health data format | Verify data structure and types |
| `ML002` | Model not available | Check model status or try different model |
| `ML003` | Processing timeout | Reduce data size or try batch processing |
| `ML004` | Insufficient data quality | Ensure minimum data requirements |
| `ML005` | Rate limit exceeded | Wait and retry with exponential backoff |

## Data Requirements

### Minimum Data Requirements

| Data Type | Minimum Duration | Minimum Frequency | Quality Score |
|-----------|------------------|-------------------|---------------|
| Steps | 3 days | 1 minute | > 0.7 |
| Heart Rate | 1 day | 1 minute | > 0.8 |
| Sleep | 5 nights | Per sleep session | > 0.9 |
| Activity | 1 week | 1 minute | > 0.7 |

### Data Quality Metrics

```json
{
  "data_quality_score": 0.92,
  "completeness": 0.95,
  "consistency": 0.89,
  "accuracy": 0.93,
  "timeliness": 0.97,
  "issues": [
    {
      "type": "missing_data",
      "severity": "low",
      "count": 12,
      "percentage": 0.05
    }
  ]
}
```

## Webhooks

### Event Types

- `analysis.completed` - Analysis request finished
- `insights.generated` - AI insights ready
- `monitoring.alert` - Health alert triggered
- `batch.completed` - Batch processing finished
- `model.drift_detected` - Data drift detected

### Webhook Payload Example

```json
{
  "event_type": "analysis.completed",
  "timestamp": "ISO8601",
  "data": {
    "request_id": "string",
    "user_id": "string",
    "status": "completed",
    "result_url": "/v1/ml/results/{request_id}"
  },
  "signature": "sha256=..."
}
```

## SDKs and Integration

### Python SDK Example

```python
from clarity_ml_client import ClarityMLClient

client = ClarityMLClient(
    api_key="your_api_key",
    base_url="https://api.clarity-loop.com/v1/ml"
)

# Analyze health data
result = await client.analyze_actigraphy(
    user_id="user123",
    data_type="steps",
    values=step_data,
    timestamps=timestamps
)

# Generate insights
insights = await client.generate_insights(
    user_id="user123",
    actigraphy_features=result.features,
    insight_type="general"
)
```

### JavaScript SDK Example

```javascript
import { ClarityMLClient } from '@clarity-loop/ml-client';

const client = new ClarityMLClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.clarity-loop.com/v1/ml'
});

// Analyze health data
const result = await client.analyzeActigraphy({
  userId: 'user123',
  dataType: 'steps',
  values: stepData,
  timestamps: timestamps
});

// Generate insights
const insights = await client.generateInsights({
  userId: 'user123',
  actigraphyFeatures: result.features,
  insightType: 'general'
});
```

This comprehensive API documentation provides developers with all the information needed to integrate ML capabilities into health applications using the Clarity Loop Backend's advanced PAT-based analysis engine.
