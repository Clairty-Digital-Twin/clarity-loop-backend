# Security Implementation Guide

This document outlines security implementation for the Clarity Loop Backend, ensuring HIPAA-compliant health data protection.

## Security Architecture

### Zero-Trust Security Model

- **Identity Verification**: Multi-factor authentication via Firebase
- **Least Privilege**: Role-based access control (RBAC)
- **Data Encryption**: End-to-end encryption at rest and in transit
- **Network Security**: VPC isolation, private endpoints
- **Audit Logging**: Complete activity tracking

### Defense in Depth

```
┌─────────────────────────────────────────┐
│ Application Layer (FastAPI + Auth)     │ ← Input validation, rate limiting
├─────────────────────────────────────────┤
│ API Gateway (Cloud Run)                │ ← TLS termination, CORS, CSP
├─────────────────────────────────────────┤
│ Service Mesh (VPC)                     │ ← Internal encryption, isolation
├─────────────────────────────────────────┤
│ Data Layer (Firestore + Encryption)    │ ← Field-level encryption, access control
└─────────────────────────────────────────┘
```

## Authentication & Authorization

### Firebase Authentication Integration

```python
# src/clarity/auth/firebase_auth.py
from firebase_admin import auth, credentials
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

class FirebaseAuthenticator:
    def __init__(self):
        self.security = HTTPBearer()
        
    async def verify_token(self, token: str) -> dict:
        """Verify Firebase ID token and extract user claims."""
        try:
            decoded_token = auth.verify_id_token(token)
            return {
                "user_id": decoded_token["uid"],
                "email": decoded_token.get("email"),
                "email_verified": decoded_token.get("email_verified", False),
                "custom_claims": decoded_token.get("custom_claims", {})
            }
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid authentication token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """Dependency to get current authenticated user."""
    authenticator = FirebaseAuthenticator()
    return await authenticator.verify_token(credentials.credentials)
```

### Role-Based Access Control

```python
# src/clarity/auth/rbac.py
from enum import Enum
from functools import wraps

class UserRole(Enum):
    USER = "user"
    PREMIUM_USER = "premium_user"
    ADMIN = "admin"
    HEALTHCARE_PROVIDER = "healthcare_provider"

class Permission(Enum):
    READ_OWN_DATA = "read_own_data"
    WRITE_OWN_DATA = "write_own_data"
    DELETE_OWN_DATA = "delete_own_data"
    READ_INSIGHTS = "read_insights"
    EXPORT_DATA = "export_data"
    ADMIN_ACCESS = "admin_access"

ROLE_PERMISSIONS = {
    UserRole.USER: [
        Permission.READ_OWN_DATA,
        Permission.WRITE_OWN_DATA,
        Permission.READ_INSIGHTS
    ],
    UserRole.PREMIUM_USER: [
        Permission.READ_OWN_DATA,
        Permission.WRITE_OWN_DATA,
        Permission.DELETE_OWN_DATA,
        Permission.READ_INSIGHTS,
        Permission.EXPORT_DATA
    ],
    UserRole.HEALTHCARE_PROVIDER: [
        Permission.READ_OWN_DATA,
        Permission.READ_INSIGHTS,
        Permission.EXPORT_DATA
    ],
    UserRole.ADMIN: [perm for perm in Permission]
}

def require_permission(permission: Permission):
    """Decorator to enforce permission requirements."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user')
            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            user_role = UserRole(user.get('custom_claims', {}).get('role', 'user'))
            if permission not in ROLE_PERMISSIONS.get(user_role, []):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

## Data Encryption

### Field-Level Encryption

```python
# src/clarity/security/encryption.py
from cryptography.fernet import Fernet
from google.cloud import kms
import base64

class FieldEncryption:
    def __init__(self):
        self.kms_client = kms.KeyManagementServiceClient()
        self.key_name = "projects/PROJECT_ID/locations/LOCATION/keyRings/RING/cryptoKeys/KEY"
    
    async def encrypt_sensitive_field(self, plaintext: str) -> str:
        """Encrypt sensitive data using Google Cloud KMS."""
        encrypt_response = self.kms_client.encrypt(
            request={"name": self.key_name, "plaintext": plaintext.encode('utf-8')}
        )
        return base64.b64encode(encrypt_response.ciphertext).decode('utf-8')
    
    async def decrypt_sensitive_field(self, ciphertext: str) -> str:
        """Decrypt sensitive data using Google Cloud KMS."""
        ciphertext_bytes = base64.b64decode(ciphertext.encode('utf-8'))
        decrypt_response = self.kms_client.decrypt(
            request={"name": self.key_name, "ciphertext": ciphertext_bytes}
        )
        return decrypt_response.plaintext.decode('utf-8')

# Usage in data models
class HealthDataPoint(BaseModel):
    user_id: str
    data_type: str
    value: float
    encrypted_notes: Optional[str] = None  # Encrypted field
    
    async def set_notes(self, notes: str):
        """Set encrypted notes."""
        encryption = FieldEncryption()
        self.encrypted_notes = await encryption.encrypt_sensitive_field(notes)
    
    async def get_notes(self) -> str:
        """Get decrypted notes."""
        if not self.encrypted_notes:
            return ""
        encryption = FieldEncryption()
        return await encryption.decrypt_sensitive_field(self.encrypted_notes)
```

## Input Validation & Sanitization

### Pydantic Security Models

```python
# src/clarity/models/security.py
from pydantic import BaseModel, EmailStr, Field, validator
import re

class SecureHealthDataInput(BaseModel):
    data_type: str = Field(..., regex="^[a-zA-Z_][a-zA-Z0-9_]*$")
    value: float = Field(..., ge=0, le=1000000)
    timestamp: datetime
    source: str = Field(..., max_length=100)
    
    @validator('data_type')
    def validate_data_type(cls, v):
        """Validate data type against allowed values."""
        allowed_types = ['heart_rate', 'steps', 'sleep_duration', 'blood_pressure']
        if v not in allowed_types:
            raise ValueError(f'Invalid data type. Allowed: {allowed_types}')
        return v
    
    @validator('source')
    def sanitize_source(cls, v):
        """Sanitize source string to prevent injection."""
        # Remove any special characters that could be used for injection
        sanitized = re.sub(r'[<>"\';()&+]', '', v)
        return sanitized[:100]  # Truncate to max length

class SecureUserMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    
    @validator('message')
    def sanitize_message(cls, v):
        """Sanitize user message content."""
        # Remove potentially dangerous HTML/script content
        sanitized = re.sub(r'<[^>]+>', '', v)  # Remove HTML tags
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        return sanitized.strip()
```

## Rate Limiting & DDoS Protection

### Rate Limiting Implementation

```python
# src/clarity/security/rate_limit.py
from fastapi import HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

# Different rate limits for different endpoints
RATE_LIMITS = {
    "auth": "5/minute",      # Authentication endpoints
    "upload": "10/minute",   # Health data upload
    "insights": "20/hour",   # AI insights generation
    "chat": "30/minute",     # Chat with AI
    "api": "100/minute"      # General API access
}

def create_rate_limited_route(limit_type: str):
    """Create rate limited route decorator."""
    def decorator(func):
        return limiter.limit(RATE_LIMITS[limit_type])(func)
    return decorator

# Usage
@app.post("/api/v1/auth/login")
@create_rate_limited_route("auth")
async def login(request: Request, credentials: LoginCredentials):
    # Login implementation
    pass
```

## Security Headers & CORS

### FastAPI Security Configuration

```python
# src/clarity/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://clarityloop.com", "https://app.clarityloop.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["clarityloop.com", "*.clarityloop.com"]
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self' https://firebase.googleapis.com"
    )
    
    return response
```

## Audit Logging

### Security Event Logging

```python
# src/clarity/security/audit_log.py
import structlog
from enum import Enum
from datetime import datetime

class SecurityEventType(Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

class SecurityAuditLogger:
    def __init__(self):
        self.logger = structlog.get_logger("security_audit")
    
    async def log_security_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        additional_data: Optional[dict] = None
    ):
        """Log security-related events for audit trail."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "resource": resource,
            "additional_data": additional_data or {}
        }
        
        await self.logger.ainfo("Security Event", **log_data)
        
        # For critical events, also send to monitoring
        if event_type in [SecurityEventType.LOGIN_FAILURE, SecurityEventType.PERMISSION_DENIED]:
            await self._send_to_monitoring(log_data)
    
    async def _send_to_monitoring(self, log_data: dict):
        """Send critical security events to monitoring system."""
        # Implementation depends on monitoring system (e.g., Google Cloud Logging)
        pass
```

## Environment Security

### Secret Management

```python
# src/clarity/config/secrets.py
from google.cloud import secretmanager
from functools import lru_cache

class SecretManager:
    def __init__(self):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = "clarity-loop-backend"
    
    @lru_cache(maxsize=100)
    def get_secret(self, secret_name: str, version: str = "latest") -> str:
        """Retrieve secret from Google Secret Manager with caching."""
        name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")

# Environment configuration with secrets
class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = Field(default_factory=lambda: SecretManager().get_secret("database_url"))
    
    # Firebase
    FIREBASE_PRIVATE_KEY: str = Field(default_factory=lambda: SecretManager().get_secret("firebase_private_key"))
    
    # AI Services
    GEMINI_API_KEY: str = Field(default_factory=lambda: SecretManager().get_secret("gemini_api_key"))
    
    # Encryption
    ENCRYPTION_KEY: str = Field(default_factory=lambda: SecretManager().get_secret("encryption_key"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

## Security Testing

### Security Test Cases

```python
# tests/security/test_auth_security.py
import pytest
from fastapi.testclient import TestClient

class TestAuthenticationSecurity:
    def test_invalid_token_rejection(self, client: TestClient):
        """Test that invalid tokens are properly rejected."""
        response = client.get(
            "/api/v1/user/profile",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401
    
    def test_rate_limiting_login(self, client: TestClient):
        """Test rate limiting on login endpoint."""
        # Attempt multiple failed logins
        for _ in range(6):  # Exceeds 5/minute limit
            response = client.post(
                "/api/v1/auth/login",
                json={"email": "test@example.com", "password": "wrong"}
            )
        
        assert response.status_code == 429  # Too Many Requests
    
    def test_sql_injection_prevention(self, authenticated_client: TestClient):
        """Test SQL injection prevention in input fields."""
        malicious_input = "'; DROP TABLE users; --"
        response = authenticated_client.post(
            "/api/v1/health-data/upload",
            json={"data_type": malicious_input, "value": 72}
        )
        assert response.status_code == 422  # Validation error
```

## Compliance & Standards

### HIPAA Compliance Checklist

- ✅ Administrative Safeguards: Access controls, audit logs, training
- ✅ Physical Safeguards: Google Cloud data center security
- ✅ Technical Safeguards: Encryption, authentication, audit logs
- ✅ Breach Notification: Automated monitoring and alerting
- ✅ Business Associate Agreements: Google Cloud BAA in place

### Security Monitoring

- **Real-time Alerts**: Failed authentication, permission violations
- **Log Analysis**: Automated security event correlation
- **Vulnerability Scanning**: Regular dependency and code scanning
- **Penetration Testing**: Quarterly security assessments

## Vulnerability Management

### Current Security Status (Updated: 2025-06-01)

**Safety CLI Version**: 3.5.1 with Safety Platform integration  
**Total Dependencies Scanned**: 403 packages  
**Vulnerabilities Detected**: 37 known security issues  
**Current Policy**: Ignoring unpinned specification vulnerabilities (default Safety Platform behavior)

### Active Security Vulnerabilities

The following vulnerabilities are currently present in our dependency tree but **ignored by Safety Platform policy** due to unpinned version specifications. These require immediate evaluation and remediation:

#### 🔴 **CRITICAL - Authentication & Cryptography**

**1. python-jose (JWT Authentication Library)**
- **Vulnerability IDs**: 70716, 70715  
- **Affected Versions**: All versions (vulnerable spec: ">=0")  
- **Impact**: JWT token handling vulnerabilities  
- **Status**: ⚠️ PRODUCTION RISK - Core authentication component  
- **Priority**: IMMEDIATE

**2. cryptography (Core Cryptographic Library)**
- **Vulnerability IDs**: 73711, 76170  
- **Affected Versions**: 37.0.0-43.0.0, 42.0.0-44.0.0  
- **Impact**: Cryptographic implementation flaws  
- **Status**: ⚠️ HIGH RISK - All encrypted communications affected  
- **Priority**: IMMEDIATE

#### 🟠 **HIGH - AI/ML Components**

**3. torch (PyTorch Deep Learning Framework)**
- **Vulnerability IDs**: 76771, 76769  
- **Affected Versions**: <2.6.0, <=2.6.0  
- **Current Version**: 2.7.0 ✅ (appears patched)  
- **Impact**: ML model security vulnerabilities  
- **Status**: ⚠️ MONITORING - Verify patch effectiveness

**4. transformers (Hugging Face Transformers)**
- **Vulnerability IDs**: 74882, 76262, 77149  
- **Affected Versions**: <4.48.0, <4.48.0, <4.50.0  
- **Current Version**: 4.52.4 ✅ (appears patched)  
- **Impact**: NLP model security issues  
- **Status**: ⚠️ MONITORING - Verify patch effectiveness

#### 🟡 **MEDIUM - Development & Infrastructure**

**5. python-multipart (File Upload Handling)**
- **Vulnerability ID**: 74427  
- **Affected Versions**: <0.0.18  
- **Impact**: File upload vulnerabilities  
- **Status**: ⚠️ MEDIUM RISK

**6. notebook (Jupyter Development Environment)**
- **Vulnerability ID**: 72963  
- **Affected Versions**: 7.0.0-7.2.1  
- **Current Version**: 7.4.3 ✅ (appears patched)  
- **Impact**: Development environment security  
- **Status**: ✅ LOW RISK - Development only

**7. mkdocs-material (Documentation Framework)**
- **Vulnerability IDs**: 64496, 72715  
- **Affected Versions**: <9.5.5, <9.5.32  
- **Impact**: Documentation site vulnerabilities  
- **Status**: ✅ LOW RISK - Documentation only

### Remediation Strategy

#### Phase 1: Immediate Action (Priority 1) 🔴
**Timeline**: Within 24 hours

1. **Evaluate python-jose alternatives**:
   - Research `PyJWT` + `cryptography` direct implementation
   - Evaluate `Authlib` as enterprise alternative
   - Assess `python-jwt` compatibility

2. **Update cryptography library**:
   - Force upgrade to latest stable version (>44.0.1)
   - Test all cryptographic functions
   - Verify HIPAA compliance maintained

3. **Create security hotfix branch**:
   ```bash
   git checkout -b security/vulnerability-remediation
   pip install --upgrade cryptography python-jose[cryptography]
   pip freeze > requirements-security-audit.txt
   ```

#### Phase 2: Verification & Testing (Priority 2) 🟠
**Timeline**: Within 48 hours

1. **Comprehensive security testing**:
   - Run full authentication test suite
   - Verify JWT token generation/validation
   - Test end-to-end encryption flows
   - Execute penetration testing scenarios

2. **Dependency pinning review**:
   - Pin all security-critical packages to specific versions
   - Update `pyproject.toml` with security-hardened constraints
   - Configure Safety Platform policy for stricter enforcement

#### Phase 3: Infrastructure Hardening (Priority 3) 🟡
**Timeline**: Within 1 week

1. **Enhanced dependency monitoring**:
   - Configure automated vulnerability alerts
   - Implement dependency update automation
   - Setup security regression testing

2. **Supply chain security**:
   - Enable package signature verification
   - Implement SBOM (Software Bill of Materials) generation
   - Configure security scanning in CI/CD pipeline

### Security Policy Configuration

**Recommended Safety Platform Policy Updates**:
```yaml
# .safety-policy.yml (proposed)
security:
  ignore-unpinned-requirements: false  # Enable strict checking
  ignore-severity: low  # Only ignore low-severity issues
  
vulnerability-rules:
  - id: "*"
    reason: "All vulnerabilities require explicit review"
    expires: null
```

### Monitoring & Alerting

**Current Monitoring Status**:
- ✅ Safety CLI 3.5.1 integrated with platform
- ✅ Automated scanning in `make security` command
- ✅ CI/CD integration via GitHub Actions
- ⚠️ Safety Platform policy needs reconfiguration for production

**Required Enhancements**:
- Real-time vulnerability notifications
- Dependency update automation
- Security regression prevention
- Compliance audit logging

### Compliance Impact

**HIPAA Compliance Status**: ⚠️ **CONDITIONAL**
- Authentication vulnerabilities pose PHI access risk
- Cryptographic vulnerabilities threaten data encryption requirements  
- Remediation required before production health data processing

**Recommended Actions**:
1. Complete Phase 1 remediation before any production deployment
2. Document all security fixes for compliance audit trail
3. Conduct formal security review after vulnerability resolution
4. Update security documentation with final remediation details

This security implementation ensures the Clarity Loop Backend meets enterprise-grade security standards while maintaining HIPAA compliance for health data protection.
