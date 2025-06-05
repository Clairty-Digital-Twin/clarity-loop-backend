# AI Insights API

**UPDATED:** June 3rd, 2025 - Based on actual implementation in `src/clarity/api/v1/insights.py`

## Overview

The AI Insights API generates natural language health insights using Google's Gemini 2.5 Pro model, combining health data analysis with contextual understanding.

## Authentication

All endpoints require Firebase JWT token:

```
Authorization: Bearer <firebase-jwt-token>
```

## Endpoints

### Generate AI Insights

```http
POST /api/v1/insights/generate
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

**Request Body:**

```json
{
  "user_id": "firebase-uid-123",
  "data_sources": [
    "recent_health_data",
    "pat_analysis",
    "sleep_patterns"
  ],
  "analysis_type": "comprehensive",
  "date_range": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-01-15T23:59:59Z"
  },
  "context": {
    "user_goals": ["improve_sleep", "increase_activity"],
    "medical_conditions": [],
    "medications": [],
    "lifestyle_factors": ["remote_work", "irregular_schedule"]
  },
  "preferences": {
    "language": "en",
    "tone": "professional",
    "detail_level": "moderate"
  }
}
```

**Analysis Types:**

- `comprehensive` - Full multi-modal health analysis (default)
- `sleep_focused` - Sleep patterns and recommendations
- `activity_focused` - Physical activity and fitness insights
- `trend_analysis` - Long-term health trend analysis
- `quick_summary` - Brief daily/weekly summary

**Response (201 Created):**

```json
{
  "insight_id": "insight-uuid-def789",
  "status": "generating",
  "estimated_completion": "2025-01-15T10:31:00Z",
  "analysis_type": "comprehensive",
  "data_points": 1247,
  "message": "Insight generation started successfully"
}
```

### Get Generated Insight

```http
GET /api/v1/insights/{insight_id}
Authorization: Bearer <firebase-jwt-token>
```

**Response (200 OK):**

```json
{
  "insight_id": "insight-uuid-def789",
  "status": "completed",
  "analysis_type": "comprehensive",
  "generated_at": "2025-01-15T10:30:45Z",
  "data_period": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-01-15T23:59:59Z"
  },
  "insights": {
    "summary": "Your health metrics show consistent improvement over the past two weeks, with notable gains in sleep quality and daily activity levels.",
    "key_findings": [
      {
        "category": "sleep",
        "finding": "Average sleep duration increased by 23 minutes",
        "significance": "high",
        "trend": "improving"
      },
      {
        "category": "activity",
        "finding": "Daily step count 15% above personal average",
        "significance": "moderate",
        "trend": "stable"
      }
    ],
    "recommendations": [
      {
        "category": "sleep",
        "action": "Maintain current bedtime routine",
        "rationale": "Your consistent sleep schedule correlates with improved REM sleep",
        "priority": "high"
      },
      {
        "category": "activity",
        "action": "Consider adding 10-minute morning walks",
        "rationale": "Could further enhance cardiovascular health indicators",
        "priority": "medium"
      }
    ],
    "health_score": {
      "overall": 78,
      "sleep": 82,
      "activity": 74,
      "recovery": 76,
      "trend": "improving"
    }
  },
  "data_quality": {
    "completeness": 0.94,
    "reliability": 0.89,
    "data_points": 1247,
    "confidence": "high"
  },
  "sources": [
    "apple_healthkit",
    "pat_model_analysis",
    "gemini_ai_processing"
  ]
}
```

**Status Values:**

- `pending` - Insight request queued
- `generating` - AI model processing data
- `completed` - Insight generated successfully
- `failed` - Generation encountered an error
- `expired` - Request timed out

### List User Insights

```http
GET /api/v1/insights/?limit=20&analysis_type=comprehensive
Authorization: Bearer <firebase-jwt-token>
```

**Query Parameters:**

- `limit` (optional): Number of insights (1-100, default: 20)
- `analysis_type` (optional): Filter by analysis type
- `start_date` (optional): Filter insights generated after date
- `status` (optional): Filter by status

**Response (200 OK):**

```json
{
  "insights": [
    {
      "insight_id": "insight-uuid-def789",
      "analysis_type": "comprehensive",
      "status": "completed",
      "generated_at": "2025-01-15T10:30:45Z",
      "health_score": 78,
      "summary": "Your health metrics show consistent improvement..."
    }
  ],
  "pagination": {
    "has_next": false,
    "total_count": 15
  }
}
```

## AI Model Integration

### Gemini 2.5 Pro Features

- **Multi-modal Analysis**: Processes numerical data + contextual information
- **Personalization**: Adapts insights to user goals and medical history
- **Natural Language**: Generates human-readable, actionable recommendations
- **Medical Knowledge**: Incorporates current health and wellness research

### Data Sources

- **PAT Model Output**: Sleep and circadian rhythm analysis
- **Health Metrics**: Heart rate, steps, activity levels
- **Historical Patterns**: Long-term trends and correlations
- **User Context**: Goals, preferences, medical conditions

## Error Handling

### Common Error Responses

**400 Bad Request:**

```json
{
  "error": "validation_error",
  "message": "Invalid date range: end date must be after start date",
  "details": {
    "field": "date_range.end",
    "code": "INVALID_DATE_RANGE"
  }
}
```

**429 Too Many Requests:**

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many insight requests. Try again in 60 seconds.",
  "retry_after": 60
}
```

**503 Service Unavailable:**

```json
{
  "error": "ai_service_unavailable",
  "message": "Gemini AI service temporarily unavailable",
  "estimated_recovery": "2025-01-15T10:35:00Z"
}
```

## Rate Limits

- **Generation**: 10 insight requests per hour per user
- **Retrieval**: 100 requests per minute per user
- **Burst**: Brief bursts of up to 5 requests per minute allowed

## Implementation Details

- **Location**: `src/clarity/api/v1/insights.py`
- **AI Model**: Gemini 2.5 Pro via Google AI SDK
- **Async Processing**: Background generation with real-time status updates
- **Caching**: Results cached in Firestore for instant retrieval
- **Validation**: Comprehensive input validation and sanitization
