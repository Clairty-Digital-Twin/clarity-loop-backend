# Rate Limiting Configuration

## Overview

CLARITY implements application-level rate limiting using the `slowapi` middleware to protect against abuse, ensure fair resource usage, and prevent DoS attacks. Rate limiting is applied per-user for authenticated requests and per-IP for anonymous requests.

## Architecture

### Middleware Stack Order
1. CORS Middleware
2. Security Headers
3. Authentication Middleware
4. **Rate Limiting Middleware** ← Processes after authentication
5. Request Logging (dev only)

### Key Components

- **slowapi**: FastAPI-compatible rate limiting library
- **Redis Support**: Optional distributed rate limiting across instances
- **Custom Key Functions**: User-based and IP-based limiting
- **Rate Limit Headers**: X-RateLimit-* headers on all responses

## Configuration

### Default Rate Limits

| Endpoint Type | Rate Limit | Key Function | Purpose |
|--------------|------------|--------------|----------|
| Global | 1000/hour | User/IP | Default for all endpoints |
| Authentication | 20/hour | IP only | Prevent brute force |
| Login | 10/minute | IP only | Additional login protection |
| Registration | 5/hour | IP only | Prevent spam accounts |
| Health Data | 100/hour | User/IP | Normal usage patterns |
| AI/ML Analysis | 20/hour | User/IP | Resource-intensive operations |
| Read Operations | 200/minute | User/IP | General data retrieval |
| Write Operations | 60/minute | User/IP | Data modifications |

### Environment Variables

```bash
# Redis URL for distributed rate limiting (optional)
REDIS_URL=redis://localhost:6379/0

# Custom rate limits (optional)
RATE_LIMIT_GLOBAL=1000/hour
RATE_LIMIT_AUTH=20/hour
RATE_LIMIT_AI=50/hour
```

## Implementation Details

### Key Extraction Functions

1. **get_user_id_or_ip(request)**
   - Extracts user ID from authenticated requests
   - Falls back to IP address for anonymous requests
   - Used for most endpoints

2. **get_ip_only(request)**
   - Always uses IP address
   - Used for authentication endpoints
   - Prevents account enumeration attacks

### Rate Limiting Strategy

- **Algorithm**: Fixed window
- **Storage**: In-memory (default) or Redis (distributed)
- **Headers**: Automatic X-RateLimit-* headers
- **Key Format**: `endpoint:key` (e.g., `/api/v1/login:ip:192.168.1.1`)

## Applied Rate Limits by Endpoint

### Authentication Endpoints
```python
@router.post("/login")
@auth_limiter.limit("10/minute")  # 10 login attempts per minute per IP

@router.post("/register")  
@auth_limiter.limit("5/hour")     # 5 registrations per hour per IP
```

### Health Data Endpoints
```python
@router.post("/health-data")
@health_limiter.limit("100/hour") # 100 uploads per hour per user
```

### AI/ML Endpoints
```python
@router.post("/pat/step-analysis")
@ai_limiter.limit("20/hour")      # 20 AI analyses per hour per user
```

## Response Headers

All rate-limited endpoints include these headers:

| Header | Description | Example |
|--------|-------------|---------|
| X-RateLimit-Limit | Request limit per window | `100` |
| X-RateLimit-Remaining | Requests remaining | `95` |
| X-RateLimit-Reset | Window reset time (Unix timestamp) | `1642089600` |

## Error Responses

When rate limit is exceeded:

```json
{
  "type": "rate_limit_exceeded",
  "title": "Too Many Requests",
  "detail": "Rate limit exceeded. Please retry after some time.",
  "status": 429,
  "instance": "https://api.clarity.health/requests/12345"
}
```

## Testing Rate Limits

### Manual Testing
```bash
# Test login rate limit
for i in {1..12}; do
  curl -X POST http://localhost:8000/api/v1/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"test@example.com","password":"test"}' \
    -w "\nStatus: %{http_code}\n"
  sleep 0.5
done
```

### Automated Testing
```bash
# Run rate limiting test script
python scripts/test_rate_limiting.py
```

## Monitoring

### CloudWatch Metrics
- `RateLimitExceeded` - Count of rate limit violations
- `RateLimitByEndpoint` - Violations per endpoint
- `RateLimitByUser` - Top users hitting limits

### Logs
```python
# Rate limit exceeded
WARNING: Rate limit exceeded for /api/v1/auth/login - Key: ip:192.168.1.1, Limit: 10/minute

# Debug logging
DEBUG: Rate limiting by user ID: user123
DEBUG: Rate limiting by IP address: 192.168.1.1
```

## Best Practices

1. **Differentiated Limits**: Resource-intensive endpoints have stricter limits
2. **User vs IP**: Authenticated endpoints use per-user limits for fairness
3. **Auth Protection**: Authentication endpoints always use IP-based limits
4. **Graceful Degradation**: Clear error messages with retry information
5. **Monitoring**: Track rate limit violations to adjust limits

## Troubleshooting

### Common Issues

1. **"Rate limit exceeded" for legitimate users**
   - Check if limits are too restrictive
   - Consider user tier-based limits
   - Monitor usage patterns

2. **Redis connection failures**
   - Falls back to in-memory storage
   - Check REDIS_URL configuration
   - Verify Redis is running

3. **Missing rate limit headers**
   - Ensure endpoint has @limiter.limit decorator
   - Check if Request parameter is included
   - Verify middleware order

### Debug Commands

```python
# Check current rate limit status
from clarity.middleware.rate_limiting import limiter
status = await limiter.get_window_stats(request, "100/hour")
```

## Security Considerations

1. **IP Spoofing**: Use proper proxy headers (X-Forwarded-For)
2. **Distributed Attacks**: Consider implementing CAPTCHA
3. **Account Lockout**: Combine with account lockout service
4. **Cost Protection**: Prevents abuse of expensive operations

## Future Enhancements

1. **Dynamic Limits**: Adjust based on system load
2. **User Tiers**: Different limits for free/premium users
3. **Geographical Limits**: Region-based rate limiting
4. **API Keys**: Support for API key-based limits
5. **Sliding Window**: More accurate rate limiting algorithm