# Clarity Loop Backend API Audit

Generated: 2025-06-17

## 1. API Endpoints Overview

### Base URL
- Development: `http://localhost:8080`
- Production: `http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com`

### API Structure
All API endpoints are prefixed with `/api/v1/`

### 1.1 Authentication Endpoints (`/api/v1/auth/`)

| Method | Endpoint | Description | Authentication Required |
|--------|----------|-------------|------------------------|
| POST | `/register` | Register new user | No |
| POST | `/login` | Login user | No |
| GET | `/me` | Get current user info | Yes (Bearer token) |
| PUT | `/me` | Update current user info | Yes (Bearer token) |
| POST | `/logout` | Logout user | Yes (Bearer token) |
| POST | `/refresh` | Refresh access token | No (Refresh token) |
| POST | `/confirm-email` | Confirm email with code | No |
| POST | `/resend-confirmation` | Resend confirmation code | No |
| POST | `/forgot-password` | Initiate password reset | No |
| POST | `/reset-password` | Reset password with code | No |
| GET | `/health` | Auth service health check | No |

### 1.2 Health Data Endpoints (`/api/v1/health-data/`)

| Method | Endpoint | Description | Authentication Required |
|--------|----------|-------------|------------------------|
| POST | `/upload` | Upload health data | Yes |
| GET | `/processing/{processing_id}` | Get processing status | Yes |
| GET | `/` or `//` | List health data (paginated) | Yes |
| DELETE | `/{processing_id}` | Delete health data | Yes |
| GET | `/health` | Health data service status | No |

### 1.3 PAT Analysis Endpoints (`/api/v1/pat/`)

| Method | Endpoint | Description | Authentication Required |
|--------|----------|-------------|------------------------|
| POST | `/analyze-step-data` | Analyze Apple HealthKit step data | Yes |
| POST | `/analyze` | Analyze direct actigraphy data | Yes |
| GET | `/analysis/{processing_id}` | Get PAT analysis results | Yes |
| GET | `/health` | PAT service health check | Yes |
| GET | `/models/info` | Get PAT model information | Yes |

### 1.4 AI Insights Endpoints (`/api/v1/insights/`)

| Method | Endpoint | Description | Authentication Required |
|--------|----------|-------------|------------------------|
| POST | `/generate` | Generate health insights | Yes |
| GET | `/{insight_id}` | Get cached insight | Yes |
| GET | `/history/{user_id}` | Get insight history | Yes |
| GET | `/status` | Service health status | Yes |

### 1.5 HealthKit Upload Endpoints (`/api/v1/healthkit/`)

| Method | Endpoint | Description | Authentication Required |
|--------|----------|-------------|------------------------|
| POST | `/upload` | Upload HealthKit data | Yes |
| GET | `/upload-status/{upload_id}` | Get upload status | Yes |

### 1.6 Metrics Endpoints (`/api/v1/metrics/`)

| Method | Endpoint | Description | Authentication Required |
|--------|----------|-------------|------------------------|
| GET | `/health` | Service health metrics | Yes |
| GET | `/system` | System performance metrics | Yes |
| GET | `/user/{user_id}` | User-specific metrics | Yes |

### 1.7 WebSocket Endpoints (`/api/v1/ws/`)

| Endpoint | Description | Authentication Required |
|----------|-------------|------------------------|
| `/chat/{room_id}` | Real-time chat with health insights | Yes (Query param: token) |
| `/health-analysis/{user_id}` | Real-time health analysis updates | Yes (Query param: token) |
| GET `/chat/stats` | Get chat statistics | No |
| GET `/chat/users/{room_id}` | Get users in room | No |

### 1.8 Test Endpoints (`/api/v1/test/`)

| Method | Endpoint | Description | Authentication Required |
|--------|----------|-------------|------------------------|
| GET | `/hello` | Basic test endpoint | No |
| POST | `/echo` | Echo test endpoint | No |
| GET | `/protected` | Protected test endpoint | Yes |
| GET | `/error` | Error handling test | No |

### 1.9 Debug Endpoints (`/api/v1/debug/`) - Development Only

| Method | Endpoint | Description | Authentication Required |
|--------|----------|-------------|------------------------|
| GET | `/config` | View current configuration | No |
| GET | `/auth-info` | View auth configuration | No |
| GET | `/health` | Comprehensive health check | No |

## 2. Authentication Flow

### 2.1 AWS Cognito Integration

The backend uses AWS Cognito for authentication with the following configuration:

```
User Pool ID: us-east-1_efXaR5EcP
Client ID: 7sm7ckrkovg78b03n1595euc71
Region: us-east-1
```

### 2.2 Authentication Middleware

- Located at: `src/clarity/middleware/auth_middleware.py`
- Automatically validates Bearer tokens from Authorization headers
- Creates/updates user records in DynamoDB
- Populates `request.state.user` with authenticated user context
- Supports both FastAPI and Modal compatibility

### 2.3 Token Management

**Access Token:**
- JWT format
- Default expiry: 3600 seconds (1 hour)
- Contains user claims (sub, email, cognito:username, etc.)

**Refresh Token:**
- Used to obtain new access tokens
- Longer expiry period
- Must be securely stored by client

### 2.4 Authentication Flow

1. **Registration:**
   - POST `/api/v1/auth/register` with email, password, names
   - Creates Cognito user
   - May require email verification
   - Returns tokens on success

2. **Login:**
   - POST `/api/v1/auth/login` with email, password
   - Authenticates against Cognito
   - Returns access and refresh tokens

3. **Token Refresh:**
   - POST `/api/v1/auth/refresh` with refresh token
   - Returns new access token

4. **Protected Requests:**
   - Include `Authorization: Bearer <access_token>` header
   - Middleware validates token and populates user context

## 3. Data Models

### 3.1 User Models

**UserContext:**
```python
{
    "user_id": str,
    "email": str | None,
    "role": UserRole,  # PATIENT, CLINICIAN, RESEARCHER, ADMIN
    "permissions": list[Permission],
    "is_verified": bool,
    "is_active": bool,
    "custom_claims": dict[str, Any],
    "created_at": datetime | None,
    "last_login": datetime | None
}
```

### 3.2 Health Data Models

**HealthDataUpload:**
```python
{
    "user_id": UUID,
    "metrics": list[HealthMetric],  # Max 100 per upload
    "upload_source": str,  # e.g., "apple_health", "fitbit"
    "client_timestamp": datetime,
    "sync_token": str | None
}
```

**HealthMetric:**
```python
{
    "metric_id": UUID,
    "metric_type": HealthMetricType,
    "biometric_data": BiometricData | None,
    "sleep_data": SleepData | None,
    "activity_data": ActivityData | None,
    "mental_health_data": MentalHealthIndicator | None,
    "device_id": str | None,
    "raw_data": dict | None,
    "metadata": dict | None,
    "created_at": datetime
}
```

**Supported Metric Types:**
- HEART_RATE
- HEART_RATE_VARIABILITY
- BLOOD_PRESSURE
- BLOOD_OXYGEN
- RESPIRATORY_RATE
- SLEEP_ANALYSIS
- ACTIVITY_LEVEL
- STRESS_INDICATORS
- MOOD_ASSESSMENT
- COGNITIVE_METRICS
- ENVIRONMENTAL
- BODY_TEMPERATURE
- BLOOD_GLUCOSE

### 3.3 PAT Analysis Models

**ActigraphyInput:**
```python
{
    "user_id": str,
    "data_points": list[ActigraphyDataPoint],
    "sampling_rate": float,  # Default: 1.0 (samples per minute)
    "duration_hours": int
}
```

**ActigraphyAnalysis:**
```python
{
    "activity_patterns": dict,
    "sleep_metrics": dict,
    "circadian_rhythm": dict,
    "feature_embeddings": list[float],
    "confidence_scores": dict
}
```

## 4. External Service Integrations

### 4.1 AWS Services

**DynamoDB:**
- Tables: clarity-health-data
- Stores health data, processing jobs, user profiles, audit logs
- Region: us-east-1

**S3:**
- Bucket: clarity-ml-models-124355672559 (ML models)
- Bucket: clarity-health-uploads (user uploads)
- Used for storing raw health data and ML models

**Cognito:**
- User authentication and authorization
- Token validation
- User management

**CloudWatch:**
- Logging and monitoring
- Metrics collection

### 4.2 AI/ML Services

**Google Gemini:**
- Model: gemini-2.0-flash-exp
- Used for health insights generation
- Natural language processing for health data

**PAT (Pretrained Actigraphy Transformer):**
- Models: PAT-S, PAT-M, PAT-L
- Actigraphy analysis
- Sleep pattern detection
- Activity classification

## 5. Request/Response Formats

### 5.1 Standard Response Format

**Success Response:**
```json
{
    "data": {...},
    "metadata": {
        "request_id": "req_insights_12345678",
        "timestamp": "2025-06-17T10:30:00Z",
        "service": "service-name",
        "version": "1.0.0",
        "processing_time_ms": 125.5
    }
}
```

**Error Response (RFC 7807 Problem Details):**
```json
{
    "type": "error_type",
    "title": "Error Title",
    "detail": "Detailed error message",
    "status": 400,
    "instance": "https://api.clarity.health/requests/12345",
    "errors": [
        {
            "field": "field_name",
            "message": "Field-specific error",
            "code": "ERROR_CODE"
        }
    ]
}
```

### 5.2 Pagination Format (HAL-style)

```json
{
    "data": [...],
    "pagination": {
        "page_size": 50,
        "has_next": true,
        "has_previous": false,
        "next_cursor": "eyJpZCI6IjEyMyJ9",
        "total_count": 150
    },
    "links": {
        "self": "https://api.clarity.health/api/v1/health-data?limit=50",
        "next": "https://api.clarity.health/api/v1/health-data?limit=50&cursor=eyJpZCI6IjEyMyJ9",
        "first": "https://api.clarity.health/api/v1/health-data?limit=50"
    }
}
```

## 6. WebSocket Features

### 6.1 Real-time Chat

**Connection:**
```
ws://localhost:8080/api/v1/ws/chat/{room_id}?token={jwt_token}
```

**Message Types:**
- MESSAGE: Regular chat message
- TYPING: Typing indicator
- HEARTBEAT: Keep-alive signal
- HEALTH_INSIGHT: Health data analysis request
- SYSTEM: System messages
- ERROR: Error messages

**Message Format:**
```json
{
    "type": "MESSAGE",
    "user_id": "user_123",
    "timestamp": "2025-06-17T10:30:00Z",
    "content": "Message content"
}
```

### 6.2 Health Analysis Updates

**Connection:**
```
ws://localhost:8080/api/v1/ws/health-analysis/{user_id}?token={jwt_token}
```

Provides real-time updates during:
- PAT analysis processing
- AI insight generation
- Health data processing

## 7. Environment Variables

### Required Environment Variables

```bash
# Environment
ENVIRONMENT=development|production|testing
DEBUG=true|false

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# AWS Cognito
COGNITO_USER_POOL_ID=us-east-1_efXaR5EcP
COGNITO_CLIENT_ID=7sm7ckrkovg78b03n1595euc71
COGNITO_REGION=us-east-1

# DynamoDB
DYNAMODB_TABLE_NAME=clarity-health-data

# S3
S3_BUCKET_NAME=clarity-health-uploads

# Gemini AI
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-1.5-flash

# Authentication
ENABLE_AUTH=true|false
SECRET_KEY=your-secret-key

# Server Configuration
HOST=127.0.0.1
PORT=8080
LOG_LEVEL=INFO

# Optional
SKIP_AWS_INIT=false
SKIP_EXTERNAL_SERVICES=false
CORS_ORIGINS=["http://localhost:3000"]
```

## 8. Security Features

### 8.1 Authentication & Authorization
- JWT-based authentication via AWS Cognito
- Role-based access control (RBAC)
- Automatic token validation middleware
- Support for MFA (Multi-Factor Authentication)

### 8.2 Data Protection
- HIPAA-compliant data handling
- Audit logging for all data operations
- Data encryption at rest (via AWS)
- Secure token storage recommendations

### 8.3 API Security
- CORS configuration
- Rate limiting on WebSocket connections
- Input validation on all endpoints
- SQL injection prevention (using DynamoDB)
- XSS protection through data sanitization

### 8.4 Compliance Features
- GDPR data deletion support
- Audit trail maintenance
- User consent tracking
- Data retention policies (30-day default)

## 9. Error Handling

### HTTP Status Codes Used
- 200: Success
- 201: Created
- 202: Accepted (e.g., email verification required)
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 409: Conflict (e.g., user already exists)
- 410: Gone (removed endpoints)
- 422: Unprocessable Entity
- 500: Internal Server Error
- 503: Service Unavailable

### Error Response Structure
All errors follow RFC 7807 Problem Details format for consistency.

## 10. Development Tools

### API Documentation
- OpenAPI/Swagger UI: `/docs`
- ReDoc: `/redoc`
- Prometheus Metrics: `/metrics`

### Health Checks
- Root health: `/health`
- Service-specific health checks available on each router

### Testing Endpoints
- Available in development/testing environments
- Located under `/api/v1/test/`
- Debug endpoints under `/api/v1/debug/`

## 11. Frontend Integration Guidelines

### 11.1 Authentication Flow
1. Register/login to get tokens
2. Store tokens securely (HttpOnly cookies recommended)
3. Include Bearer token in all authenticated requests
4. Refresh token before expiry
5. Handle 401 responses by refreshing or re-authenticating

### 11.2 Data Upload Flow
1. Authenticate user
2. Prepare health data in correct format
3. POST to `/api/v1/health-data/upload`
4. Poll processing status endpoint
5. Retrieve results when ready

### 11.3 WebSocket Integration
1. Establish connection with JWT token
2. Handle connection lifecycle
3. Implement reconnection logic
4. Process different message types
5. Send heartbeats to maintain connection

### 11.4 Error Handling
1. Parse RFC 7807 error responses
2. Display user-friendly error messages
3. Implement retry logic for transient failures
4. Log errors for debugging

## 12. Performance Considerations

### 12.1 Rate Limits
- Health data upload: Max 100 metrics per request
- WebSocket messages: Max 64KB per message
- Pagination: Default 50 items, max 1000

### 12.2 Caching
- DynamoDB caching enabled (5-minute TTL)
- Token validation caching
- PAT analysis results cached

### 12.3 Timeouts
- Default request timeout: 30 seconds
- WebSocket idle timeout: 5 minutes
- Processing job timeout: varies by job type