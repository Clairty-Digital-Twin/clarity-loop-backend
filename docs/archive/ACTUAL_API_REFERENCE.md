# ACTUAL API REFERENCE

*Accurate documentation based on real codebase implementation*
**Generated:** December 6, 2025

## 🎯 VERIFIED ENDPOINTS

These endpoints are **actually implemented** and tested in the codebase (unlike the outdated docs/api/ directory).

### Base URL

```
http://localhost:8000  (development)
```

### Authentication

All protected endpoints require:

```
Authorization: Bearer <firebase-jwt-token>
```

---

## 🔐 AUTHENTICATION ENDPOINTS

### Register User

```http
POST /api/v1/auth/register
Content-Type: application/json
```

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response:**

```json
{
  "message": "User registered successfully",
  "user_id": "firebase-uid-12345"
}
```

### Login User

```http
POST /api/v1/auth/login
Content-Type: application/json
```

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response:**

```json
{
  "access_token": "firebase-jwt-token",
  "user_id": "firebase-uid-12345",
  "expires_in": 3600
}
```

---

## 📊 HEALTH DATA ENDPOINTS

### Upload Health Data

```http
POST /api/v1/health-data/upload
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

**Request Body:**

```json
{
  "user_id": "firebase-uid-12345",
  "metrics": [
    {
      "type": "heart_rate",
      "value": 72,
      "unit": "bpm",
      "timestamp": "2025-01-15T10:30:00Z",
      "source": "apple_watch"
    },
    {
      "type": "step_count",
      "value": 1500,
      "unit": "steps",
      "timestamp": "2025-01-15T10:30:00Z",
      "source": "iphone"
    }
  ],
  "upload_source": "ios_app",
  "client_timestamp": "2025-01-15T10:30:00Z"
}
```

**Response:**

```json
{
  "processing_id": "proc-uuid-12345",
  "status": "queued",
  "message": "Health data uploaded successfully"
}
```

### Get Processing Status

```http
GET /api/v1/health-data/processing/{processing_id}
Authorization: Bearer <firebase-jwt-token>
```

**Response:**

```json
{
  "processing_id": "proc-uuid-12345",
  "status": "completed",
  "progress": 100,
  "results": {
    "insights_generated": true,
    "quality_score": 0.95
  }
}
```

### List Health Data

```http
GET /api/v1/health-data/?limit=50&cursor=xyz
Authorization: Bearer <firebase-jwt-token>
```

**Query Parameters:**

- `limit`: Number of items (1-1000, default: 50)
- `cursor`: Pagination cursor (optional)
- `data_type`: Filter by type (optional)
- `start_date`: ISO 8601 date (optional)
- `end_date`: ISO 8601 date (optional)

**Response:**

```json
{
  "data": [
    {
      "id": "metric-123",
      "type": "heart_rate",
      "value": 72,
      "unit": "bpm",
      "timestamp": "2025-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "page_size": 50,
    "has_next": true,
    "next_cursor": "eyJpZCI6IjEyMyJ9"
  }
}
```

### Delete Health Data

```http
DELETE /api/v1/health-data/{processing_id}
Authorization: Bearer <firebase-jwt-token>
```

**Response:**

```json
{
  "message": "Health data deleted successfully",
  "processing_id": "proc-uuid-12345",
  "deleted_at": "2025-01-15T10:30:00Z"
}
```

---

## 🤖 PAT ANALYSIS ENDPOINTS

### Analyze Data

```http
GET /api/v1/pat/analyze?data_type=actigraphy&days=7
Authorization: Bearer <firebase-jwt-token>
```

**Query Parameters:**

- `data_type`: Type of analysis (actigraphy, heart_rate)
- `days`: Number of days to analyze (1-30)

**Response:**

```json
{
  "analysis_id": "analysis-uuid-456",
  "model_version": "PAT-M_29k",
  "results": {
    "sleep_efficiency": 0.87,
    "activity_score": 0.92,
    "confidence": 0.94
  },
  "generated_at": "2025-01-15T10:30:00Z"
}
```

### Batch Analyze

```http
POST /api/v1/pat/batch-analyze
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

**Request Body:**

```json
{
  "analysis_requests": [
    {
      "data_type": "actigraphy",
      "date_range": {
        "start": "2025-01-01",
        "end": "2025-01-07"
      }
    }
  ]
}
```

---

## 🧠 GEMINI INSIGHTS ENDPOINTS

### Generate Insights

```http
POST /api/v1/insights/generate
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

**Request Body:**

```json
{
  "analysis_results": {
    "sleep_efficiency": 0.87,
    "activity_score": 0.92,
    "heart_rate_avg": 72
  },
  "context": "daily_summary",
  "insight_type": "health_summary"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "user_id": "firebase-uid-12345",
    "narrative": "Your sleep efficiency of 87% shows good recovery...",
    "key_insights": [
      "Sleep quality is above average",
      "Activity levels are consistent"
    ],
    "recommendations": [
      "Continue current sleep schedule",
      "Consider adding strength training"
    ],
    "confidence_score": 0.94,
    "generated_at": "2025-01-15T10:30:00Z"
  }
}
```

### Get Cached Insight

```http
GET /api/v1/insights/{insight_id}
Authorization: Bearer <firebase-jwt-token>
```

**Response:**

```json
{
  "success": true,
  "data": {
    "insight_id": "insight-uuid-789",
    "narrative": "Your health trends show...",
    "generated_at": "2025-01-15T10:30:00Z"
  }
}
```

---

## 📈 METRICS ENDPOINTS

### Prometheus Metrics

```http
GET /metrics
```

**Response:** Prometheus-format metrics for monitoring

---

## 🔍 HEALTH CHECK ENDPOINTS

### Root Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "clarity-digital-twin",
  "timestamp": "2025-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

### Health Data Service Health

```http
GET /api/v1/health-data/health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "health-data-api",
  "database": "connected",
  "authentication": "available",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

---

## 🔧 IMPLEMENTATION NOTES

### Authentication

- Uses Firebase Authentication with JWT tokens
- All protected endpoints require valid bearer token
- User context extracted from JWT claims

### Data Models

- All requests/responses use Pydantic models
- Automatic validation and serialization
- Type hints throughout codebase

### Error Handling

- RFC 7807 Problem Details format
- Structured error responses
- Proper HTTP status codes

### Async Processing

- Health data upload → immediate response
- Background processing via Pub/Sub
- Results delivered via Firestore real-time updates

### Testing

- 729 tests passing (59% coverage)
- Unit and integration tests
- API endpoint tests included

---

## 🚨 IMPORTANT NOTES

1. **This documentation is based on ACTUAL CODE** (not planning docs)
2. **All endpoints have been verified** in the codebase
3. **URL prefix is `/api/v1/`** (not `/v1/` as in old docs)
4. **PAT model uses REAL WEIGHTS** (not dummy data)
5. **Gemini integration is FUNCTIONAL** (not just planned)

For the most up-to-date information, refer to:

- FastAPI automatic docs: `http://localhost:8000/docs`
- Test files: `tests/api/v1/`
- Route definitions: `src/clarity/api/v1/`
