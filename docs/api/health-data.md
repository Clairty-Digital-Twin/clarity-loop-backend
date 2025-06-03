# Health Data API

**UPDATED:** December 6, 2025 - Based on actual implementation in `src/clarity/api/v1/health_data.py`

## Overview

The Health Data API handles upload, storage, and retrieval of health metrics from various sources including Apple HealthKit, wearable devices, and manual inputs.

## Authentication

All endpoints require Firebase JWT token:

```
Authorization: Bearer <firebase-jwt-token>
```

## Endpoints

### Upload Health Data

```http
POST /api/v1/health-data/upload
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

**Request Body:**

```json
{
  "user_id": "firebase-uid-123",
  "metrics": [
    {
      "type": "heart_rate",
      "value": 72.5,
      "unit": "bpm",
      "timestamp": "2025-01-15T10:30:00Z",
      "source": "apple_watch",
      "metadata": {
        "device_model": "Apple Watch Series 9",
        "confidence": 0.95
      }
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
  "client_timestamp": "2025-01-15T10:30:00Z",
  "sync_token": "optional-sync-identifier"
}
```

**Response (201 Created):**

```json
{
  "processing_id": "proc-uuid-abc123",
  "status": "queued",
  "message": "Health data uploaded successfully",
  "metrics_count": 2,
  "uploaded_at": "2025-01-15T10:30:00Z"
}
```

**Supported Metric Types:**

- `heart_rate` (bpm)
- `step_count` (steps)
- `sleep_analysis` (hours)
- `respiratory_rate` (breaths/min)
- `active_energy` (calories)
- `distance_walking` (meters)

### Get Processing Status

```http
GET /api/v1/health-data/processing/{processing_id}
Authorization: Bearer <firebase-jwt-token>
```

**Response (200 OK):**

```json
{
  "processing_id": "proc-uuid-abc123",
  "status": "completed",
  "progress": 100,
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:31:00Z",
  "results": {
    "insights_generated": true,
    "quality_score": 0.95,
    "metrics_processed": 2
  }
}
```

**Status Values:**

- `pending` - Upload received, processing queued
- `processing` - Data currently being analyzed  
- `completed` - Processing finished successfully
- `failed` - Processing encountered an error
- `cancelled` - Processing was cancelled

### List Health Data

```http
GET /api/v1/health-data/?limit=50&cursor=xyz
Authorization: Bearer <firebase-jwt-token>
```

**Query Parameters:**

- `limit` (optional): Number of items (1-1000, default: 50)
- `cursor` (optional): Pagination cursor
- `offset` (optional): Alternative to cursor for pagination
- `data_type` (optional): Filter by metric type
- `start_date` (optional): ISO 8601 date (e.g., 2025-01-15T00:00:00Z)
- `end_date` (optional): ISO 8601 date
- `source` (optional): Filter by data source

**Response (200 OK):**

```json
{
  "data": [
    {
      "id": "metric-uuid-456",
      "type": "heart_rate",
      "value": 72.5,
      "unit": "bpm", 
      "timestamp": "2025-01-15T10:30:00Z",
      "source": "apple_watch",
      "quality_score": 0.95
    }
  ],
  "pagination": {
    "page_size": 50,
    "has_next": true,
    "has_previous": false,
    "next_cursor": "eyJpZCI6IjQ1NiJ9"
  },
  "links": {
    "self": "https://api.clarity.health/api/v1/health-data?limit=50",
    "next": "https://api.clarity.health/api/v1/health-data?limit=50&cursor=eyJpZCI6IjQ1NiJ9"
  }
}
```

### Delete Health Data

```http
DELETE /api/v1/health-data/{processing_id}
Authorization: Bearer <firebase-jwt-token>
```

**Response (200 OK):**

```json
{
  "message": "Health data deleted successfully",
  "processing_id": "proc-uuid-abc123",
  "deleted_at": "2025-01-15T10:30:00Z"
}
```

### Health Check

```http
GET /api/v1/health-data/health
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "service": "health-data-api",
  "database": "connected",
  "authentication": "available",
  "timestamp": "2025-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

## Data Processing Flow

1. **Upload** → Health data stored in Cloud Storage
2. **Queue** → Analysis task published to Pub/Sub
3. **Process** → Background analysis (PAT model + Gemini insights)
4. **Results** → Insights saved to Firestore, real-time updates to client

## Security & Privacy

- **User Isolation**: Users can only access their own data
- **Encryption**: All data encrypted at rest and in transit
- **Audit Logging**: All data access logged for compliance
- **HIPAA Ready**: Designed with healthcare compliance in mind

## Implementation Details

- **Location**: `src/clarity/api/v1/health_data.py`
- **Storage**: Google Cloud Storage for raw data
- **Database**: Firestore for processed results
- **Async Processing**: Pub/Sub for background jobs
- **Validation**: Pydantic models with strict type checking
