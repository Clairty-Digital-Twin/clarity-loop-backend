# API Documentation

This directory contains comprehensive API documentation for the Clarity Loop Backend REST API.

## API Overview

The Clarity Loop Backend provides a RESTful API designed for iOS and watchOS health applications. The API follows OpenAPI 3.0 specifications and implements async-first patterns for optimal performance.

### Base URL

- **Production**: `https://api.clarityloop.com/v1`
- **Staging**: `https://staging-api.clarityloop.com/v1`
- **Development**: `http://localhost:8000/v1`

### API Versioning

- **Current Version**: v1
- **Versioning Strategy**: URL path versioning (`/v1/`, `/v2/`)
- **Deprecation Policy**: 12-month notice before version retirement
- **Backward Compatibility**: Maintained within major versions

## Authentication

### Firebase Authentication Integration

All API endpoints require valid Firebase authentication tokens, except for public health checks and documentation.

#### Token Format

```http
Authorization: Bearer <firebase-jwt-token>
```

#### Token Validation

- **Issuer**: Firebase Auth for your project
- **Audience**: Your Firebase project ID
- **Expiration**: Tokens expire after 1 hour
- **Refresh**: Use Firebase SDK refresh mechanisms

### Custom Claims

User roles and permissions are managed through Firebase custom claims:

```json
{
  "custom_claims": {
    "roles": ["patient", "premium_user"],
    "permissions": ["read:health_data", "write:health_data"],
    "subscription_tier": "premium",
    "data_retention_period": "5_years"
  }
}
```

## Core API Principles

### 1. Async-First Design

- **Immediate Acknowledgment**: Upload endpoints return immediate success responses
- **Background Processing**: Complex operations processed asynchronously
- **Status Endpoints**: Check processing status via dedicated endpoints
- **Real-time Updates**: Clients receive updates via Firestore listeners

### 2. Error Handling

Consistent error response format across all endpoints:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid heart rate value",
    "details": {
      "field": "heart_rate",
      "value": 300,
      "constraint": "must be between 30-220"
    },
    "request_id": "req_abc123",
    "timestamp": "2024-01-20T14:30:00Z"
  }
}
```

### 3. Rate Limiting

- **Global Limit**: 1000 requests per hour per user
- **Endpoint-Specific**: Varies by endpoint complexity
- **Burst Allowance**: Short-term burst handling
- **Headers**: Rate limit info in response headers

### 4. Data Validation

- **Input Validation**: Comprehensive validation using Pydantic models
- **Health Data Ranges**: Physiologically realistic value ranges
- **Temporal Validation**: Logical timestamp relationships
- **Data Quality Scoring**: Automatic quality assessment

## API Categories

### Authentication Endpoints

- **POST** `/v1/auth/token/verify` - Verify Firebase token
- **POST** `/v1/auth/token/refresh` - Refresh authentication token
- **GET** `/v1/auth/user/profile` - Get authenticated user profile
- **DELETE** `/v1/auth/logout` - Logout and invalidate tokens

### Health Data Endpoints

- **POST** `/v1/health/data/upload` - Upload health data batch
- **GET** `/v1/health/data/sessions` - Get health data sessions
- **GET** `/v1/health/data/session/{sessionId}` - Get specific session
- **DELETE** `/v1/health/data/session/{sessionId}` - Delete session

### Insights & Analytics

- **GET** `/v1/insights/daily/{date}` - Get daily insights
- **GET** `/v1/insights/weekly/{week}` - Get weekly insights
- **GET** `/v1/insights/trends` - Get health trends
- **POST** `/v1/insights/generate` - Trigger insight generation

### User Management

- **GET** `/v1/user/profile` - Get user profile
- **PUT** `/v1/user/profile` - Update user profile
- **POST** `/v1/user/preferences` - Set user preferences
- **DELETE** `/v1/user/account` - Delete user account

### System & Monitoring

- **GET** `/v1/system/health` - System health check
- **GET** `/v1/system/status` - System status and metrics
- **GET** `/v1/system/version` - API version information

## Request/Response Patterns

### Standard Response Format

```json
{
  "success": true,
  "data": {
    // Endpoint-specific response data
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2024-01-20T14:30:00Z",
    "processing_time_ms": 150,
    "api_version": "1.0.0"
  }
}
```

### Async Processing Response

```json
{
  "success": true,
  "data": {
    "job_id": "job_abc123",
    "status": "queued",
    "estimated_completion": "2024-01-20T14:32:00Z"
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2024-01-20T14:30:00Z"
  }
}
```

### Pagination Response

```json
{
  "success": true,
  "data": {
    "items": [...],
    "pagination": {
      "page": 1,
      "per_page": 20,
      "total_items": 150,
      "total_pages": 8,
      "has_next": true,
      "next_page": 2
    }
  }
}
```

## Error Codes and Handling

### HTTP Status Codes

- **200 OK**: Successful request
- **201 Created**: Resource created successfully
- **202 Accepted**: Request accepted for async processing
- **400 Bad Request**: Invalid request format or parameters
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource conflict (e.g., duplicate data)
- **422 Unprocessable Entity**: Validation errors
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Service temporarily unavailable

### Custom Error Codes

```json
{
  "VALIDATION_ERROR": "Request validation failed",
  "AUTHENTICATION_FAILED": "Authentication token invalid",
  "INSUFFICIENT_PERMISSIONS": "User lacks required permissions",
  "RATE_LIMIT_EXCEEDED": "Request rate limit exceeded",
  "DATA_QUALITY_ERROR": "Health data quality issues detected",
  "PROCESSING_ERROR": "Background processing failed",
  "EXTERNAL_SERVICE_ERROR": "External service unavailable",
  "MAINTENANCE_MODE": "System under maintenance"
}
```

## Security Considerations

### Request Security

- **HTTPS Only**: All requests must use TLS 1.3
- **Content-Type Validation**: Strict content type enforcement
- **Request Size Limits**: Maximum payload size enforced
- **Input Sanitization**: All inputs sanitized and validated

### Response Security

- **Security Headers**: Comprehensive security headers
- **Data Masking**: Sensitive data masked in responses
- **CORS Configuration**: Strict cross-origin policies
- **Content Security Policy**: CSP headers for XSS protection

### Data Privacy

- **PII Protection**: Personal information encrypted
- **Data Minimization**: Only necessary data in responses
- **Audit Logging**: All API access logged
- **Consent Enforcement**: User consent verified for data access

## Performance Characteristics

### Response Times (P95)

- **Authentication**: < 100ms
- **Health Data Upload**: < 200ms
- **Simple Queries**: < 150ms
- **Complex Analytics**: < 500ms (sync), < 5s (async)
- **Insight Generation**: 30-120s (async)

### Throughput Limits

- **Global**: 10,000 requests/minute
- **Per User**: 1,000 requests/hour
- **Upload Endpoint**: 100 concurrent uploads
- **Query Endpoints**: 500 concurrent queries

### Caching Strategy

- **Response Caching**: 5-minute cache for read operations
- **CDN Caching**: Static content and documentation
- **Database Caching**: Redis for frequently accessed data
- **ML Model Caching**: Model results cached for 1 hour

## Client SDK Recommendations

### iOS/Swift Integration

```swift
// Example SDK usage
import ClarityLoopSDK

let client = ClarityLoopClient(
    baseURL: "https://api.clarityloop.com/v1",
    firebaseAuth: Auth.auth()
)

// Upload health data
let healthData = HealthDataBatch(/* ... */)
let result = await client.uploadHealthData(healthData)

// Get insights
let insights = await client.getDailyInsights(for: Date())
```

### Error Handling Patterns

```swift
do {
    let insights = try await client.getDailyInsights(for: date)
    // Handle success
} catch ClarityLoopError.authenticationFailed {
    // Handle auth error
} catch ClarityLoopError.rateLimitExceeded(let retryAfter) {
    // Handle rate limiting
} catch ClarityLoopError.validationError(let details) {
    // Handle validation errors
}
```

## Testing and Development

### API Testing Tools

- **Postman Collection**: Complete API collection available
- **OpenAPI Specification**: Interactive documentation
- **Test Data**: Realistic test datasets provided
- **Mock Endpoints**: Development mock server available

### Development Environment

- **Local Setup**: Docker Compose for local development
- **Test Database**: Isolated test data
- **Debug Mode**: Enhanced logging and debugging
- **Hot Reload**: Fast development iteration

## Documentation Structure

- **[Authentication](./authentication.md)** - Detailed authentication flows
- **[Health Data API](./health-data.md)** - Health data endpoints specification
- **[Insights API](./insights.md)** - AI insights and analytics endpoints
- **[User Management API](./user-management.md)** - User profile and preferences
- **[Webhooks](./webhooks.md)** - Webhook integration patterns
- **[OpenAPI Specification](./openapi.yaml)** - Complete OpenAPI 3.0 spec
- **[SDKs](./sdks/)** - Client SDK documentation and examples
- **[Testing](./testing.md)** - API testing guidelines and tools
