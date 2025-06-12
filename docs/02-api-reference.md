# CLARITY API Reference

Complete reference for all 44 API endpoints. All endpoints return JSON and use standard HTTP status codes.

## Authentication

Most endpoints require authentication via JWT tokens from AWS Cognito.

```bash
# Get auth token
POST /api/v1/auth/login
{
  "email": "user@example.com",
  "password": "password"
}

# Use token in requests
Authorization: Bearer <jwt_token>
```

## Base URL

- **Production**: `https://api.clarity-health.com`
- **Staging**: `https://staging-api.clarity-health.com`
- **Local**: `http://localhost:8000`

---

## Authentication Endpoints

### POST `/api/v1/auth/register`

Create a new user account.

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "securePassword123",
  "first_name": "John",
  "last_name": "Doe"
}
```

**Response (201):**

```json
{
  "user_id": "uuid-here",
  "email": "user@example.com",
  "verification_required": true
}
```

### POST `/api/v1/auth/login`

Authenticate user and get access token.

**Request Body:**

```json
{
  "email": "user@example.com", 
  "password": "securePassword123"
}
```

**Response (200):**

```json
{
  "access_token": "jwt_token_here",
  "refresh_token": "refresh_token_here",
  "expires_in": 3600,
  "user_id": "uuid-here"
}
```

### POST `/api/v1/auth/logout`

Invalidate current session.

**Headers:** `Authorization: Bearer <token>`

**Response (200):**

```json
{
  "message": "Successfully logged out"
}
```

### POST `/api/v1/auth/refresh`

Get new access token using refresh token.

**Request Body:**

```json
{
  "refresh_token": "refresh_token_here"
}
```

**Response (200):**

```json
{
  "access_token": "new_jwt_token",
  "expires_in": 3600
}
```

### POST `/api/v1/auth/verify`

Verify email address.

**Request Body:**

```json
{
  "email": "user@example.com",
  "verification_code": "123456"
}
```

### POST `/api/v1/auth/reset-password`

Request password reset.

**Request Body:**

```json
{
  "email": "user@example.com"
}
```

### GET `/api/v1/auth/profile`

Get user profile information.

**Headers:** `Authorization: Bearer <token>`

**Response (200):**

```json
{
  "user_id": "uuid-here",
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "created_at": "2024-01-01T00:00:00Z",
  "health_data_connected": true
}
```

---

## Health Data Endpoints

### POST `/api/v1/health-data/upload`

Upload processed health metrics.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "user_id": "uuid-here",
  "metrics": [
    {
      "type": "heart_rate",
      "value": 72.5,
      "unit": "bpm",
      "timestamp": "2024-01-01T12:00:00Z",
      "source": "apple_watch"
    }
  ]
}
```

**Response (201):**

```json
{
  "processing_id": "proc-uuid-here",
  "status": "processing",
  "metrics_count": 1,
  "estimated_completion": "2024-01-01T12:01:00Z"
}
```

### GET `/api/v1/health-data/`

List user's health data with pagination.

**Headers:** `Authorization: Bearer <token>`

**Query Parameters:**

- `limit` (int, default: 50): Number of records to return
- `offset` (int, default: 0): Number of records to skip  
- `start_date` (ISO string): Filter by date range
- `end_date` (ISO string): Filter by date range
- `metric_type` (string): Filter by metric type

**Response (200):**

```json
{
  "total": 1250,
  "limit": 50,
  "offset": 0,
  "data": [
    {
      "id": "metric-uuid",
      "type": "heart_rate", 
      "value": 72.5,
      "timestamp": "2024-01-01T12:00:00Z",
      "source": "apple_watch"
    }
  ]
}
```

### GET `/api/v1/health-data/{processing_id}`

Get details of a specific processing job.

**Response (200):**

```json
{
  "processing_id": "proc-uuid-here",
  "status": "completed",
  "created_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:00:30Z",
  "metrics_processed": 1,
  "errors": []
}
```

### DELETE `/api/v1/health-data/{processing_id}`

Delete a processing job and associated data.

**Response (204):** No content

### GET `/api/v1/health-data/processing/{id}/status`

Get status of data processing job.

**Response (200):**

```json
{
  "id": "proc-uuid-here",
  "status": "processing",
  "progress": 0.75,
  "estimated_completion": "2024-01-01T12:01:00Z"
}
```

---

## HealthKit Integration Endpoints

### POST `/api/v1/healthkit/upload`

Upload raw HealthKit data export.

**Headers:**

- `Authorization: Bearer <token>`
- `Content-Type: application/json`

**Request Body:**

```json
{
  "user_id": "uuid-here",
  "export_date": "2024-01-01T12:00:00Z",
  "data": {
    "quantity_samples": [
      {
        "uuid": "sample-uuid",
        "type_identifier": "HKQuantityTypeIdentifierHeartRate",
        "start_date": "2024-01-01T12:00:00Z",
        "end_date": "2024-01-01T12:00:00Z", 
        "value": 72.5,
        "unit": "count/min",
        "source": "Apple Watch"
      }
    ],
    "category_samples": [
      {
        "uuid": "sleep-uuid",
        "type_identifier": "HKCategoryTypeIdentifierSleepAnalysis",
        "start_date": "2024-01-01T23:00:00Z",
        "end_date": "2024-01-02T07:00:00Z",
        "value": 1,
        "source": "iPhone"
      }
    ]
  }
}
```

**Response (202):**

```json
{
  "upload_id": "upload-uuid-here",
  "status": "processing",
  "estimated_completion": "2024-01-01T12:02:00Z",
  "samples_received": 1847
}
```

### GET `/api/v1/healthkit/status/{upload_id}`

Check status of HealthKit upload processing.

**Response (200):**

```json
{
  "upload_id": "upload-uuid-here",
  "status": "completed",
  "progress": 1.0,
  "samples_processed": 1847,
  "pat_analysis_ready": true,
  "insights_generated": true
}
```

### POST `/api/v1/healthkit/sync`

Trigger sync with HealthKit (for connected apps).

**Headers:** `Authorization: Bearer <token>`

**Response (202):**

```json
{
  "sync_id": "sync-uuid-here",
  "status": "initiated",
  "last_sync": "2024-01-01T00:00:00Z"
}
```

### GET `/api/v1/healthkit/categories`

Get supported HealthKit data categories.

**Response (200):**

```json
{
  "quantity_types": [
    {
      "identifier": "HKQuantityTypeIdentifierHeartRate",
      "display_name": "Heart Rate",
      "unit": "bpm",
      "category": "vitals"
    }
  ],
  "category_types": [
    {
      "identifier": "HKCategoryTypeIdentifierSleepAnalysis", 
      "display_name": "Sleep Analysis",
      "category": "sleep"
    }
  ]
}
```

---

## AI Insights Endpoints

### POST `/api/v1/insights/generate`

Generate health insights from user's data.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "user_id": "uuid-here",
  "type": "weekly_summary",
  "date_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-07T23:59:59Z"
  },
  "focus_areas": ["sleep", "activity", "heart_rate"]
}
```

**Response (200):**

```json
{
  "insight_id": "insight-uuid-here",
  "type": "weekly_summary",
  "summary": "Your sleep quality improved 12% this week, with particularly good deep sleep on Tuesday and Wednesday. Your heart rate variability suggests lower stress levels compared to last week.",
  "recommendations": [
    "Continue your current bedtime routine",
    "Consider increasing morning sunlight exposure"
  ],
  "metrics": {
    "sleep_efficiency": 0.84,
    "avg_resting_hr": 58,
    "hrv_trend": "+8%"
  },
  "generated_at": "2024-01-08T12:00:00Z"
}
```

### POST `/api/v1/insights/chat`

Interactive chat with health AI.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "message": "Why do I feel tired even with 8 hours of sleep?",
  "context": {
    "conversation_id": "chat-uuid-here",
    "focus_timeframe": "last_week"
  }
}
```

**Response (200):**

```json
{
  "response": "Based on your sleep data, while you're getting 8 hours in bed, your sleep efficiency is only 73%. You're experiencing frequent awakenings between 2-4 AM, which is fragmenting your deep sleep phases. Your HRV data also shows elevated stress levels during these periods.",
  "conversation_id": "chat-uuid-here", 
  "follow_up_questions": [
    "What might be causing these nighttime awakenings?",
    "How can I improve my sleep efficiency?"
  ],
  "relevant_data": {
    "avg_sleep_efficiency": 0.73,
    "awakening_frequency": 3.2,
    "hrv_trend": "declining"
  }
}
```

### GET `/api/v1/insights/summary`

Get daily/weekly health summaries.

**Headers:** `Authorization: Bearer <token>`

**Query Parameters:**

- `period` (string): "daily" | "weekly" | "monthly"
- `date` (ISO string): Date for the summary

**Response (200):**

```json
{
  "period": "weekly",
  "date": "2024-01-01",
  "summary": "A strong week for your health metrics...",
  "key_insights": [
    "Sleep quality improved significantly",
    "Heart rate variability is trending upward"
  ],
  "metrics": {
    "sleep_score": 85,
    "activity_score": 78,
    "recovery_score": 82
  }
}
```

### GET `/api/v1/insights/recommendations`

Get personalized health recommendations.

**Headers:** `Authorization: Bearer <token>`

**Response (200):**

```json
{
  "recommendations": [
    {
      "category": "sleep",
      "priority": "high",
      "title": "Optimize sleep schedule",
      "description": "Your data shows irregular bedtimes. Consistency could improve sleep quality by 15-20%.",
      "actionable_steps": [
        "Set a fixed bedtime within 30 minutes each night",
        "Create a wind-down routine starting 1 hour before bed"
      ]
    }
  ],
  "generated_at": "2024-01-01T12:00:00Z"
}
```

### GET `/api/v1/insights/trends`

Analyze health trends over time.

**Headers:** `Authorization: Bearer <token>`

**Query Parameters:**

- `metric` (string): "sleep" | "heart_rate" | "activity" | "hrv"
- `timeframe` (string): "week" | "month" | "quarter"

**Response (200):**

```json
{
  "metric": "sleep",
  "timeframe": "month",
  "trend": "improving",
  "change_percentage": 12.5,
  "analysis": "Your sleep quality has been steadily improving over the past month, with particularly strong gains in deep sleep duration.",
  "data_points": [
    {"date": "2024-01-01", "value": 7.2, "quality_score": 78},
    {"date": "2024-01-02", "value": 7.8, "quality_score": 82}
  ]
}
```

### GET `/api/v1/insights/alerts`

Get health alerts and warnings.

**Headers:** `Authorization: Bearer <token>`

**Response (200):**

```json
{
  "alerts": [
    {
      "id": "alert-uuid",
      "type": "warning",
      "category": "heart_rate",
      "message": "Your resting heart rate has been elevated for 3 consecutive days",
      "severity": "medium",
      "created_at": "2024-01-01T12:00:00Z",
      "actionable": true,
      "recommendations": ["Consider stress management techniques"]
    }
  ]
}
```

---

## PAT Analysis Endpoints

### POST `/api/v1/pat/analyze`

Run PAT transformer analysis on movement data.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "user_id": "uuid-here",
  "data_source": "healthkit_upload_id",
  "analysis_type": "sleep_quality",
  "timeframe": {
    "start": "2024-01-01T00:00:00Z", 
    "end": "2024-01-08T00:00:00Z"
  }
}
```

**Response (202):**

```json
{
  "analysis_id": "pat-analysis-uuid",
  "status": "processing",
  "estimated_completion": "2024-01-01T12:01:00Z",
  "model_version": "pat-v2.1"
}
```

### GET `/api/v1/pat/status/{analysis_id}`

Check status of PAT analysis.

**Response (200):**

```json
{
  "analysis_id": "pat-analysis-uuid",
  "status": "completed",
  "progress": 1.0,
  "started_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:00:45Z"
}
```

### GET `/api/v1/pat/results/{analysis_id}`

Get PAT analysis results.

**Response (200):**

```json
{
  "analysis_id": "pat-analysis-uuid",
  "results": {
    "sleep_quality_score": 0.78,
    "circadian_rhythm_stability": 0.85,
    "sleep_efficiency": 0.82,
    "predicted_sleep_stages": [
      {"start": "2024-01-01T23:00:00Z", "stage": "light", "confidence": 0.92},
      {"start": "2024-01-01T23:15:00Z", "stage": "deep", "confidence": 0.89}
    ],
    "anomalies": [
      {"timestamp": "2024-01-02T03:00:00Z", "type": "unusual_activity", "severity": "low"}
    ]
  },
  "model_metadata": {
    "version": "pat-v2.1",
    "confidence": 0.87,
    "data_quality": "good"
  }
}
```

### POST `/api/v1/pat/batch`

Run batch PAT analysis for multiple users/timeframes.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "analyses": [
    {
      "user_id": "uuid-1", 
      "timeframe": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-08T00:00:00Z"}
    },
    {
      "user_id": "uuid-2",
      "timeframe": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-08T00:00:00Z"}
    }
  ]
}
```

**Response (202):**

```json
{
  "batch_id": "batch-uuid",
  "analyses_queued": 2,
  "estimated_completion": "2024-01-01T12:05:00Z"
}
```

### GET `/api/v1/pat/models`

List available PAT model versions.

**Response (200):**

```json
{
  "models": [
    {
      "version": "pat-v2.1",
      "name": "Enhanced Sleep Analysis",
      "description": "Improved accuracy for sleep stage detection",
      "accuracy": 0.924,
      "is_default": true
    },
    {
      "version": "pat-v2.0", 
      "name": "Standard Sleep Analysis",
      "accuracy": 0.908,
      "is_default": false
    }
  ]
}
```

---

## Metrics & Monitoring Endpoints

### GET `/api/v1/metrics/health`

Get system health metrics.

**Response (200):**

```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "version": "0.2.0",
  "services": {
    "database": "healthy",
    "ai_models": "healthy", 
    "external_apis": "healthy"
  },
  "performance": {
    "avg_response_time_ms": 250,
    "requests_per_minute": 1200
  }
}
```

### GET `/api/v1/metrics/user/{user_id}`

Get user-specific health statistics.

**Headers:** `Authorization: Bearer <token>`

**Response (200):**

```json
{
  "user_id": "uuid-here",
  "data_summary": {
    "total_data_points": 15680,
    "days_of_data": 30,
    "last_upload": "2024-01-01T12:00:00Z"
  },
  "analysis_summary": {
    "pat_analyses_completed": 12,
    "insights_generated": 45,
    "avg_processing_time_seconds": 28
  }
}
```

### POST `/api/v1/metrics/export`

Export user health metrics.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "format": "csv",
  "date_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "metric_types": ["heart_rate", "sleep", "activity"]
}
```

**Response (200):**

```json
{
  "export_id": "export-uuid",
  "download_url": "https://s3.amazonaws.com/exports/export-uuid.csv",
  "expires_at": "2024-01-01T24:00:00Z"
}
```

### GET `/metrics`

Prometheus metrics endpoint (for monitoring).

**Response (200):**

```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",status="200"} 1234
...
```

---

## WebSocket Endpoints

### WS `/api/v1/ws`

Main WebSocket connection for real-time data.

**Connection Headers:**

```
Authorization: Bearer <jwt_token>
```

**Message Types:**

**Subscribe to health data updates:**

```json
{
  "type": "subscribe",
  "channel": "health_updates",
  "user_id": "uuid-here"
}
```

**Receive real-time health data:**

```json
{
  "type": "health_data",
  "data": {
    "heart_rate": 75,
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

**Start AI chat session:**

```json
{
  "type": "chat_message", 
  "message": "How is my sleep trending?",
  "conversation_id": "chat-uuid"
}
```

### GET `/api/v1/ws/health`

WebSocket health check endpoint.

**Response (200):**

```json
{
  "websocket_status": "operational",
  "active_connections": 245,
  "avg_message_latency_ms": 15
}
```

### GET `/api/v1/ws/rooms`

Get active WebSocket rooms/channels.

**Headers:** `Authorization: Bearer <token>`

**Response (200):**

```json
{
  "rooms": [
    {
      "id": "health_updates",
      "description": "Real-time health data stream",
      "active_users": 12
    },
    {
      "id": "ai_chat",
      "description": "Live AI conversation",
      "active_users": 8
    }
  ]
}
```

---

## System Endpoints

### GET `/health`

Root health check for load balancers.

**Response (200):**

```json
{
  "status": "healthy",
  "service": "clarity-backend-aws",
  "version": "0.2.0",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### GET `/docs`

OpenAPI documentation (Swagger UI).

### GET `/redoc`

ReDoc API documentation.

### GET `/openapi.json`

OpenAPI schema definition.

### GET `/`

Root endpoint with service information.

**Response (200):**

```json
{
  "name": "CLARITY Digital Twin Platform",
  "version": "0.2.0", 
  "status": "operational",
  "total_endpoints": 44,
  "api_docs": "/docs"
}
```

---

## Error Responses

All endpoints use standard HTTP status codes and return errors in this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": {
      "field": "email",
      "issue": "Email format is invalid"
    }
  },
  "request_id": "req-uuid-here",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Status Codes

- **200**: Success
- **201**: Created
- **202**: Accepted (processing)
- **400**: Bad Request
- **401**: Unauthorized
- **403**: Forbidden
- **404**: Not Found
- **422**: Validation Error
- **429**: Rate Limited
- **500**: Internal Server Error
- **503**: Service Unavailable

---

**Next**: Read [Data Models](03-data-models.md) for detailed schema definitions
