# CLARITY Backend - API Contract

This document provides a detailed contract for frontend developers interacting with the CLARITY backend API. It outlines every available endpoint, its purpose, required request payloads with examples, and expected success and error responses.

**Base URL:** `https://crave-trinity-prod--clarity-backend-fastapi-app.modal.run/api/v1`

---

## 1. Authentication (`/auth`)

Handles user registration, login, and session management.

### `POST /auth/register`

Creates a new user account.

**Request Body:** `UserRegistrationRequest`

```json
{
  "email": "user@example.com",
  "password": "a-strong-password",
  "full_name": "Jane Doe"
}
```

**Success Response (201 CREATED):** `RegistrationResponse`

```json
{
  "user_id": "firebase-user-id-123",
  "email": "user@example.com",
  "full_name": "Jane Doe",
  "is_active": true,
  "created_at": "2025-06-08T12:00:00Z"
}
```

---

### `POST /auth/login`

Authenticates a user and returns access tokens.

**Request Body:** `UserLoginRequest`

```json
{
  "email": "user@example.com",
  "password": "a-strong-password"
}
```

**Success Response (200 OK):** `LoginResponse`

```json
{
  "access_token": "ey...",
  "refresh_token": "ey...",
  "token_type": "bearer",
  "user": {
    "user_id": "firebase-user-id-123",
    "email": "user@example.com",
    "full_name": "Jane Doe",
    "is_active": true
  }
}
```

---

### `POST /auth/refresh`

Refreshes an expired access token.

**Request Body:** `RefreshTokenRequest`

```json
{
  "refresh_token": "the-long-refresh-token"
}
```

**Success Response (200 OK):** `TokenResponse`

```json
{
  "access_token": "a-new-access-token...",
  "token_type": "bearer"
}
```

---

### `POST /auth/logout`

Logs a user out and revokes tokens.

**Request Body:** `RefreshTokenRequest`

```json
{
  "refresh_token": "the-long-refresh-token"
}
```

**Success Response (200 OK):**

```json
{
  "status": "logout_successful"
}
```

---

### `GET /auth/me`

Retrieves information about the currently authenticated user. Requires `Authorization: Bearer <access_token>` header.

**Success Response (200 OK):** `UserSessionResponse`

```json
{
  "user_id": "firebase-user-id-123",
  "email": "user@example.com",
  "roles": ["user"],
  "permissions": ["read:own_data", "write:own_data"]
}
```

---

## 2. Gemini Health Insights (`/insights`)

Generates AI-powered health insights. All endpoints require `Authorization: Bearer <access_token>` header.

### `POST /insights/generate`

Generates a new health insight from analysis data.

**Request Body:** `InsightGenerationRequest`

```json
{
  "analysis_results": {
    "sleep_score": 85,
    "readiness_score": 92,
    "activity_score": 78
  },
  "context": "User is training for a marathon.",
  "insight_type": "comprehensive",
  "include_recommendations": true,
  "language": "en"
}
```

**Success Response (200 OK):** `InsightGenerationResponse`

---

## 3. Health Data (`/health-data`)

Endpoints for uploading and managing health data. All endpoints require `Authorization: Bearer <access_token>` header.

### `POST /health-data/upload`

Uploads a batch of health metrics.

**Request Body:** `HealthDataUpload`

```json
{
    "user_id": "firebase-user-id-123",
    "metrics": [
        {
            "type": "heart_rate",
            "value": 72.5,
            "unit": "bpm",
            "timestamp": "2025-06-08T12:00:00Z"
        }
    ],
    "source_device": "Apple Watch Series 9"
}
```

**Success Response (201 CREATED):** `HealthDataResponse`

---

## 4. HealthKit (`/healthkit`)

Endpoints for handling raw HealthKit data uploads.

### `POST /healthkit/upload`

Uploads a bundle of HealthKit data for asynchronous processing.

**Request Body:** `HealthKitUploadRequest`

```json
{
  "user_id": "firebase-user-id-123",
  "quantity_samples": [
    {
      "identifier": "HKQuantityTypeIdentifierHeartRate",
      "type": "quantity",
      "value": 75.0,
      "unit": "count/min",
      "start_date": "2025-06-08T12:00:00Z",
      "end_date": "2025-06-08T12:01:00Z",
      "source_name": "My Apple Watch"
    }
  ],
  "category_samples": [],
  "workouts": [],
  "correlation_samples": []
}
```

**Success Response (202 ACCEPTED):** `HealthKitUploadResponse`

---

## 5. PAT Analysis (`/pat`)

Endpoints for the Pretrained Actigraphy Transformer model.

### `POST /pat/analyze-step-data`

Analyzes Apple HealthKit step data.

**Request Body:** `StepDataRequest`

```json
{
    "step_counts": [10, 12, 15, ...],
    "timestamps": ["2025-06-08T12:00:00Z", "2025-06-08T12:01:00Z", ...],
    "user_metadata": { "age_group": "25-34" }
}
```

**Success Response (200 OK):** `AnalysisResponse`

---

## 6. Monitoring (`/metrics`)

### `GET /metrics`

Exposes application metrics for Prometheus scraping. No request body needed.

---

This document should provide your frontend team with all the necessary information. Let me know if you need any more details or further clarification on any of the endpoints. 