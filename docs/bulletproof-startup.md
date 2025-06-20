# CLARITY Bulletproof Startup System

The CLARITY platform features a bulletproof startup system that provides **zero-crash guarantee** during container initialization with crystal-clear error feedback and automatic remediation.

## 🛡️ Core Features

### Zero-Crash Guarantee

- **Pre-flight validation** - All configuration validated before any service initialization
- **Circuit breakers** - Automatic failure detection and recovery for external services
- **Graceful degradation** - Falls back to mock services when externals are unavailable
- **Comprehensive error handling** - Every possible failure scenario is caught and handled

### Crystal-Clear Feedback

- **Real-time progress reporting** - Colored terminal output with detailed status
- **Actionable error messages** - Every error includes specific solutions
- **Comprehensive error catalog** - 50+ documented error scenarios with remediation steps
- **Startup performance metrics** - Track initialization time and bottlenecks

### Service Health Matrix

- **Independent health checks** - Cognito, DynamoDB, S3 validated separately
- **Circuit breaker protection** - Automatic failure detection prevents cascading issues
- **Response time monitoring** - Track service performance during startup
- **Smart degradation** - Continue with reduced functionality when services are down

## 🚀 Quick Start

### Basic Usage

```bash
# Start with bulletproof system (unified main.py)
python -m clarity.main --bulletproof

# Start in standard mode (default)
python -m clarity.main

# Validate configuration without starting
python -m clarity.main --validate

# Use environment variable for startup mode
STARTUP_MODE=bulletproof python -m clarity.main
```

### Environment Variables

```bash
# Core settings
ENVIRONMENT=production                    # development, testing, production
AWS_REGION=us-east-1                     # AWS region for all services
DEBUG=false                              # Enable debug mode

# Authentication
ENABLE_AUTH=true                         # Enable Cognito authentication
COGNITO_USER_POOL_ID=us-east-1_example   # Cognito User Pool ID
COGNITO_CLIENT_ID=example123             # Cognito App Client ID

# External services
SKIP_EXTERNAL_SERVICES=false             # Skip AWS services (use mocks)
SKIP_AWS_INIT=false                      # Skip AWS initialization

# Security
SECRET_KEY=your-production-secret-key    # Application secret (32+ chars)
CORS_ALLOWED_ORIGINS=https://app.com     # Comma-separated CORS origins

# Timeouts
STARTUP_TIMEOUT=30                       # Maximum startup time (seconds)
HEALTH_CHECK_TIMEOUT=5                   # Per-service health check timeout
```

## 📋 Configuration Schema

The startup system uses comprehensive Pydantic validation with 100+ validation rules:

### AWS Configuration

```python
aws:
  region: str = "us-east-1"              # AWS region (validated format)
  access_key_id: str = ""                # Access key (optional with IAM)
  secret_access_key: str = ""            # Secret key (hidden in logs)
  session_token: str = ""                # Session token for temp creds
```

### Service Configuration

```python
cognito:
  user_pool_id: str = ""                 # Format: region_poolId
  client_id: str = ""                    # Min 20 characters
  region: str = ""                       # Defaults to AWS region

dynamodb:
  table_name: str = "clarity-health-data" # Alphanumeric + hyphens/underscores
  endpoint_url: str = ""                 # For local testing

s3:
  bucket_name: str = "clarity-health-uploads"    # Valid S3 bucket name
  ml_models_bucket: str = "clarity-ml-models"    # ML models bucket
  endpoint_url: str = ""                         # For local testing
```

### Security Configuration

```python
security:
  secret_key: str = "dev-secret-key"     # Min 8 chars, production requires custom
  cors_origins: list[str]                # No wildcards in production
  max_request_size: int = 10MB           # DoS protection
  rate_limit_requests: int = 100         # Requests per minute
```

## 🔍 Validation Modes

### 1. Configuration-Only Validation

Validates environment variables and configuration schema without checking external services.

```bash
python scripts/startup_validator.py --config-only
```

**Use cases:**

- CI/CD pipeline validation
- Local development setup verification
- Configuration file testing

### 2. Dry-Run Validation

Validates configuration AND tests external service connectivity without starting the application.

```bash
python scripts/startup_validator.py --dry-run
```

**Checks performed:**

- ✅ Configuration schema validation
- ✅ AWS service connectivity (Cognito, DynamoDB, S3)
- ✅ Service health and response times
- ✅ Circuit breaker functionality
- ✅ Permission validation

### 3. Full Startup

Complete application startup with bulletproof orchestration.

```bash
# Using command line flag
python -m clarity.main --bulletproof

# Using environment variable
STARTUP_MODE=bulletproof python -m clarity.main

# Automatic in production
ENVIRONMENT=production python -m clarity.main
```

**Process:**

1. **Pre-flight validation** - Configuration and environment checks
2. **Service health checks** - External service connectivity
3. **Service initialization** - Dependency injection and container setup
4. **Application startup** - FastAPI app with middleware configuration

## 🛠️ Deployment Integration

### Bulletproof Deployment Script

```bash
# Full deployment with validation
./ops/deploy_bulletproof.sh

# Validation only
./ops/deploy_bulletproof.sh --validate

# Build and push only
./ops/deploy_bulletproof.sh --build
```

**Deployment phases:**

1. **Pre-deployment validation** - Configuration and AWS resource checks
2. **Container build** - Docker image with platform validation
3. **Service deployment** - ECS task definition and service update
4. **Post-deployment validation** - Health checks and smoke tests

### Docker Integration

The system integrates with the existing Docker entrypoint for seamless container startup:

```dockerfile
# Dockerfile already includes bulletproof startup
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Environment variables control behavior
ENV STARTUP_DRY_RUN=false
ENV STARTUP_TIMEOUT=30
ENV HEALTH_CHECK_TIMEOUT=5
```

## 🎯 Error Handling

### Error Catalog

The system includes a comprehensive error catalog with solutions for 50+ scenarios:

| Error Code | Category | Description |
|------------|----------|-------------|
| CONFIG_001 | Configuration | Missing required environment variable |
| CONFIG_002 | Configuration | Invalid configuration value |
| CONFIG_003 | Configuration | Production security requirements |
| CRED_001 | Credentials | AWS credentials not found |
| CRED_002 | Permissions | Insufficient AWS permissions |
| NET_001 | Networking | Service timeout |
| RES_001 | Resources | AWS resource not found |
| RES_002 | Resources | DynamoDB table not ready |
| DEP_001 | Dependencies | Required dependency unavailable |
| ENV_001 | Environment | Development vs production mismatch |

### Error Resolution

Each error includes:

- **Clear description** - What went wrong
- **Common causes** - Why this typically happens  
- **Step-by-step solutions** - How to fix it
- **Documentation links** - Additional resources
- **Related errors** - Connected issues to check

### Example Error Output

```
🚨 AWS Credentials Not Found (CRED_001)
============================================================

📝 Description: AWS credentials are not available or invalid.
📊 Severity: HIGH
🏷️  Category: Credentials

🔍 Common Causes:
  • AWS credentials not configured
  • IAM role not attached to ECS task
  • Invalid AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY
  • Expired temporary credentials

💡 Solutions:

  1. Configure AWS credentials
     • For ECS: Attach IAM role to task definition
     • For local: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
     • For local: Run 'aws configure' to set up credentials
     • Verify credentials with 'aws sts get-caller-identity'

  2. Use development mode with mock services
     • Set SKIP_EXTERNAL_SERVICES=true
     • Set ENVIRONMENT=development
     • This will use mock services instead of AWS
```

## 🔧 Circuit Breakers

The system includes circuit breakers for all external services to prevent cascading failures:

### Circuit Breaker States

- **Closed** - Normal operation, requests pass through
- **Open** - Too many failures, requests blocked for recovery period
- **Half-Open** - Testing if service has recovered

### Configuration

```python
circuit_breaker:
  failure_threshold: int = 3           # Failures before opening
  recovery_timeout: float = 60.0       # Seconds before retry
  success_threshold: int = 1           # Successes to close circuit
```

### Monitored Services

- **Cognito** - Authentication service
- **DynamoDB** - Health data storage
- **S3** - File storage and ML models
- **Gemini** - AI service (optional)

## 📊 Performance Metrics

The startup system tracks comprehensive performance metrics:

### Startup Metrics

- **Total startup time** - End-to-end initialization duration
- **Phase timings** - Time per startup phase
- **Service response times** - Health check latencies
- **Error rates** - Failure percentages by service

### Health Check Metrics

```python
service_health_check_duration_seconds:
  - service: cognito
    status: success
    duration: 0.245

service_health_check_total:
  - service: dynamodb
    status: success
    count: 1
```

### Circuit Breaker Metrics

```python
circuit_breaker_state:
  - service: s3
    state: closed
    failure_count: 0

circuit_breaker_trips_total:
  - service: cognito
    count: 0
```

## 🧪 Testing

### Running Tests

```bash
# All startup system tests
pytest tests/startup/ -v

# Specific test files
pytest tests/startup/test_config_schema.py -v
pytest tests/startup/test_health_checks.py -v
pytest tests/startup/test_orchestrator.py -v

# Integration tests
pytest tests/startup/test_integration.py -v
```

### Test Coverage

- **Configuration validation** - 15 test cases
- **Health checks** - 20 test cases with mocked AWS services
- **Orchestrator** - 15 test cases covering all scenarios
- **Integration** - 10 end-to-end scenarios
- **Error handling** - 25 failure scenarios

### Mock Services

The system includes comprehensive mocks for development and testing:

```python
# Automatic mock detection
if config.should_use_mock_services():
    auth_provider = MockAuthProvider()
    repository = MockHealthDataRepository()
    # External services bypassed
```

## 🎨 Progress Reporting

The startup system provides real-time progress feedback:

### Console Output

```
🚀 Starting CLARITY Digital Twin
============================================================

⚙️ Validating Configuration
  ✅ Loading configuration schema (15ms)
  ✅ Validating environment requirements (8ms)
  ✅ Configuration summary (2ms)

🔍 Checking Services
  ✅ Health check: cognito (245ms)
  ✅ Health check: dynamodb (156ms)
  ✅ Health check: s3_uploads (98ms)
  ✅ Health check: s3_models (102ms)

🔧 Starting Services  
  ✅ Initializing dependency container (89ms)

✅ Startup Complete (515ms)
============================================================
```

### Colored Output

- 🟢 **Green** - Successful operations
- 🟡 **Yellow** - Warnings or degraded services
- 🔴 **Red** - Errors or failures
- 🔵 **Blue** - Information and progress
- 🟣 **Purple** - Debug information

## 🔐 Production Considerations

### Security Validations

- Custom SECRET_KEY required (no default in production)
- Explicit CORS origins (no wildcards)
- Strong authentication settings
- Secure communication settings

### Performance Requirements

- Startup time < 5 seconds (target)
- Health check timeout < 5 seconds per service
- Circuit breaker protection for all external calls
- Graceful degradation when services are slow

### Monitoring Integration

- Prometheus metrics export
- Structured logging with correlation IDs
- Health check endpoints for load balancers
- Circuit breaker state monitoring

## 🆘 Troubleshooting

### Common Issues

**1. Configuration Validation Fails**

```bash
# Check specific validation errors
python scripts/startup_validator.py --config-only

# Common fixes:
export SECRET_KEY="your-production-secret-key"
export COGNITO_USER_POOL_ID="us-east-1_yourpool"
export CORS_ALLOWED_ORIGINS="https://yourapp.com"
```

**2. Service Health Checks Fail**

```bash
# Test with mock services
export SKIP_EXTERNAL_SERVICES=true
python scripts/startup_validator.py --dry-run

# Check AWS credentials
aws sts get-caller-identity
```

**3. Startup Timeout**

```bash
# Increase timeout
export STARTUP_TIMEOUT=60

# Check individual service health
python scripts/startup_validator.py --dry-run --skip-services cognito dynamodb
```

**4. Circuit Breakers Tripping**

```bash
# Check service status
aws ecs describe-services --cluster clarity-backend-cluster --services clarity-backend-service

# Reset by restarting
export SKIP_EXTERNAL_SERVICES=true  # Temporary bypass
```

### Debug Mode

Enable detailed debug logging:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
python -m clarity.main_bulletproof
```

This provides:

- Detailed configuration dumps
- Service call tracing
- Circuit breaker state changes
- Performance timing details

## 📚 Additional Resources

- [AWS ECS Deployment Guide](../ops/README.md)
- [Configuration Reference](./02-api-reference.md)
- [Security Implementation](./operations/security.md)
- [Monitoring Setup](./operations/monitoring.md)

---

**The CLARITY bulletproof startup system ensures your application never crashes during initialization while providing crystal-clear feedback for any configuration issues. Deploy with confidence! 🚀**
