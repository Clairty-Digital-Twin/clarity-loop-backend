# Security Architecture

This document outlines the comprehensive security architecture for Clarity Loop Backend, designed with HIPAA-inspired compliance and zero-trust principles.

## Security Design Principles

### 1. Zero-Trust Architecture
- **Never Trust, Always Verify**: No implicit trust between any components
- **Least Privilege Access**: Minimal necessary permissions for all entities
- **Continuous Verification**: Ongoing authentication and authorization
- **Assume Breach**: Design for compromise detection and containment

### 2. Defense in Depth
- **Multiple Security Layers**: Redundant security controls at every level
- **Network Segmentation**: Isolated security zones and micro-perimeters
- **Application Security**: Secure coding practices and runtime protection
- **Data Protection**: Encryption, masking, and access controls

### 3. Privacy by Design
- **Data Minimization**: Collect only necessary health data
- **Purpose Limitation**: Use data only for stated purposes
- **Consent Management**: Granular user consent and control
- **Data Retention**: Automated deletion and retention policies

## Authentication Architecture

### Firebase Identity Platform Integration

#### User Authentication Flow
```
iOS/watchOS App → Firebase Auth → Custom Claims → API Gateway
                       ↓
              Identity Verification → JWT Token → Service Access
                       ↓
              MFA (Optional) → Biometric → Session Management
```

#### Supported Authentication Methods
- **Sign in with Apple**: Primary authentication for iOS users
- **Google Authentication**: Secondary option with enhanced security
- **Email/Password**: Backup authentication with strong password policies
- **Multi-Factor Authentication**: TOTP, SMS, and biometric options

#### Security Features
- **Account Protection**: Suspicious activity detection and account lockout
- **Session Management**: Secure token lifecycle and refresh policies
- **Device Binding**: Optional device registration and verification
- **Audit Logging**: Comprehensive authentication event logging

### Service-to-Service Authentication

#### Workload Identity Federation
```
Cloud Run Service → Service Account → Workload Identity → GCP APIs
                         ↓
                  IAM Policy Check → Resource Access → Audit Log
```

#### Service Account Management
- **Principle of Least Privilege**: Minimal necessary permissions
- **Key Rotation**: Automated service account key rotation
- **Impersonation**: Short-lived token impersonation
- **Monitoring**: Service account usage tracking and alerting

## Authorization Framework

### Role-Based Access Control (RBAC)

#### User Roles
```json
{
  "patient": {
    "permissions": ["read:own_health_data", "write:own_health_data"],
    "resources": ["user/{userId}/*"]
  },
  "clinician": {
    "permissions": ["read:patient_data", "write:clinical_notes"],
    "resources": ["user/{patientId}/clinical/*"]
  },
  "researcher": {
    "permissions": ["read:anonymized_data"],
    "resources": ["research/datasets/*"]
  },
  "admin": {
    "permissions": ["system:admin"],
    "resources": ["system/*"]
  }
}
```

#### Permission Model
- **Resource-Based**: Permissions tied to specific data resources
- **Contextual**: Time, location, and device-based access controls
- **Hierarchical**: Inherited permissions from parent resources
- **Auditable**: Complete permission usage audit trail

### API Authorization

#### JWT Token Validation
```python
# Pseudo-code for token validation flow
async def validate_token(token: str) -> TokenClaims:
    # 1. Verify JWT signature
    payload = jwt.decode(token, public_key, algorithms=["RS256"])
    
    # 2. Validate token claims
    validate_expiration(payload.exp)
    validate_issuer(payload.iss)
    validate_audience(payload.aud)
    
    # 3. Check custom claims
    user_roles = payload.get("custom_claims", {}).get("roles", [])
    
    # 4. Verify user status
    user_status = await check_user_status(payload.sub)
    
    return TokenClaims(payload, user_roles, user_status)
```

#### Resource-Level Authorization
- **Path-Based**: URL path pattern matching for resource access
- **Method-Based**: HTTP method restrictions per resource
- **Data-Level**: Row and column-level data access controls
- **Time-Based**: Temporal access restrictions and windows

## Data Protection

### Encryption Strategy

#### Encryption at Rest
- **Database Encryption**: AES-256 encryption for all Firestore data
- **Storage Encryption**: Customer-managed encryption keys (CMEK) for Cloud Storage
- **Key Management**: Cloud KMS for encryption key lifecycle
- **Backup Encryption**: Encrypted backups with separate key management

#### Encryption in Transit
- **TLS 1.3**: All network communication encrypted with latest TLS
- **Certificate Management**: Automated SSL certificate lifecycle
- **Perfect Forward Secrecy**: Ephemeral key exchange for session security
- **HSTS**: HTTP Strict Transport Security enforcement

#### Application-Level Encryption
```python
# Client-side encryption for sensitive health data
from cryptography.fernet import Fernet

class HealthDataEncryption:
    def __init__(self, user_key: bytes):
        self.cipher = Fernet(user_key)
    
    def encrypt_health_record(self, data: dict) -> str:
        """Encrypt health data before storage"""
        json_data = json.dumps(data)
        encrypted_data = self.cipher.encrypt(json_data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_health_record(self, encrypted_data: str) -> dict:
        """Decrypt health data after retrieval"""
        encrypted_bytes = base64.b64decode(encrypted_data)
        decrypted_data = self.cipher.decrypt(encrypted_bytes)
        return json.loads(decrypted_data.decode())
```

### Data Classification and Handling

#### Health Data Classification
- **PHI (Protected Health Information)**: HIPAA-defined identifiable health data
- **De-identified Data**: Anonymized data for research and analytics
- **Aggregate Data**: Statistical data without individual identification
- **System Data**: Non-health operational and performance data

#### Data Handling Policies
```yaml
data_policies:
  phi_data:
    storage: encrypted_at_rest
    transmission: tls_1_3_required
    access_logging: mandatory
    retention: user_controlled
    deletion: secure_multi_pass
  
  deidentified_data:
    storage: standard_encryption
    transmission: tls_required
    access_logging: standard
    retention: 7_years
    deletion: standard_deletion
  
  system_data:
    storage: standard_encryption
    transmission: tls_required
    access_logging: basic
    retention: 1_year
    deletion: standard_deletion
```

## Network Security

### VPC Architecture

#### Network Segmentation
```
Internet → Cloud Load Balancer → WAF → Cloud Armor
    ↓
Public Subnet (DMZ) → API Gateway (Cloud Run)
    ↓
Private Subnet → Internal Services (Cloud Run)
    ↓
Database Subnet → Firestore/Cloud SQL (Private)
```

#### Firewall Rules
- **Ingress Controls**: Strict inbound traffic filtering
- **Egress Controls**: Outbound traffic monitoring and restriction
- **Service-to-Service**: Internal communication controls
- **Geographic Restrictions**: Location-based access controls

### Web Application Firewall (WAF)

#### Cloud Armor Configuration
- **OWASP Top 10 Protection**: Automated protection against common vulnerabilities
- **Rate Limiting**: Request rate limits per IP and user
- **Geographic Filtering**: Country and region-based access controls
- **Custom Rules**: Application-specific attack pattern detection

#### DDoS Protection
- **Infrastructure Protection**: Google's global DDoS protection
- **Application Layer**: Layer 7 DDoS mitigation
- **Adaptive Protection**: Machine learning-based attack detection
- **Incident Response**: Automated scaling and traffic shaping

## Application Security

### Secure Development Practices

#### Code Security
- **Static Analysis**: Automated code security scanning (Snyk, SonarQube)
- **Dependency Scanning**: Third-party library vulnerability detection
- **Secret Scanning**: Prevention of hardcoded secrets in code
- **Security Linting**: Real-time security issue detection

#### Input Validation
```python
from pydantic import BaseModel, validator
from typing import Optional
import re

class HealthDataInput(BaseModel):
    heart_rate: Optional[int]
    steps: Optional[int]
    sleep_duration: Optional[float]
    
    @validator('heart_rate')
    def validate_heart_rate(cls, v):
        if v is not None and (v < 30 or v > 220):
            raise ValueError('Heart rate must be between 30-220 bpm')
        return v
    
    @validator('steps')
    def validate_steps(cls, v):
        if v is not None and (v < 0 or v > 100000):
            raise ValueError('Steps must be between 0-100000')
        return v
    
    class Config:
        # Prevent additional fields
        extra = 'forbid'
        # Validate on assignment
        validate_assignment = True
```

#### API Security
- **Rate Limiting**: Per-endpoint and user-based rate limits
- **Request Size Limits**: Maximum payload size enforcement
- **Timeout Controls**: Request timeout and connection limits
- **CORS Configuration**: Strict cross-origin resource sharing policies

### Runtime Security

#### Container Security
- **Minimal Base Images**: Distroless container images
- **Vulnerability Scanning**: Continuous container image scanning
- **Runtime Monitoring**: Container behavior analysis
- **Immutable Infrastructure**: Read-only container filesystems

#### Secret Management
- **Cloud Secret Manager**: Centralized secret storage and rotation
- **Environment Variables**: Secure injection at runtime
- **Key Rotation**: Automated secret rotation policies
- **Access Auditing**: Secret access logging and monitoring

## Compliance and Auditing

### HIPAA-Inspired Controls

#### Administrative Safeguards
- **Security Officer**: Designated security responsibility
- **Workforce Training**: Regular security awareness training
- **Access Management**: Formal access request and approval process
- **Incident Response**: Documented incident response procedures

#### Physical Safeguards
- **Google Cloud Security**: Inherited physical security controls
- **Workstation Security**: Secure development environment requirements
- **Media Controls**: Secure data storage and disposal
- **Access Controls**: Physical access logging and monitoring

#### Technical Safeguards
- **Access Control**: Unique user identification and authentication
- **Audit Controls**: Comprehensive audit trail generation
- **Integrity**: Data integrity verification and protection
- **Transmission Security**: Secure data transmission protocols

### Audit Trail Management

#### Log Categories
```yaml
audit_logs:
  authentication:
    events: [login, logout, failed_login, mfa_challenge]
    retention: 7_years
    integrity: cryptographic_hash
  
  data_access:
    events: [read, write, delete, export]
    retention: 7_years
    integrity: cryptographic_hash
  
  administrative:
    events: [config_change, user_creation, permission_change]
    retention: 10_years
    integrity: cryptographic_hash
  
  system:
    events: [service_start, service_stop, error, alert]
    retention: 3_years
    integrity: standard_hash
```

#### Audit Trail Protection
- **Immutable Logs**: Write-once audit log storage
- **Cryptographic Integrity**: Hash chains for log integrity verification
- **Separate Storage**: Audit logs stored separately from application data
- **Access Controls**: Restricted audit log access with approval workflow

## Incident Response

### Security Incident Classification
- **P0 - Critical**: Data breach or system compromise
- **P1 - High**: Security vulnerability exploitation
- **P2 - Medium**: Policy violation or suspicious activity
- **P3 - Low**: Security configuration issue

### Response Procedures
1. **Detection**: Automated monitoring and alerting
2. **Assessment**: Rapid impact and scope assessment
3. **Containment**: Immediate threat containment measures
4. **Investigation**: Forensic analysis and root cause identification
5. **Recovery**: System restoration and security hardening
6. **Lessons Learned**: Post-incident review and improvement

### Breach Notification
- **User Notification**: Affected user notification within 24 hours
- **Regulatory Reporting**: Compliance with applicable regulations
- **Documentation**: Complete incident documentation and timeline
- **Remediation**: Security improvements based on incident analysis

## Security Monitoring

### Continuous Monitoring
- **Security Information and Event Management (SIEM)**: Cloud Security Command Center
- **Behavioral Analytics**: User and entity behavior analysis
- **Threat Intelligence**: Integration with threat intelligence feeds
- **Vulnerability Management**: Continuous vulnerability assessment

### Key Security Metrics
- **Authentication Success Rate**: Failed login attempt monitoring
- **API Abuse Detection**: Unusual API usage pattern detection
- **Data Access Patterns**: Abnormal data access behavior
- **System Performance**: Security control performance impact

### Alerting and Response
- **Real-time Alerts**: Critical security event notifications
- **Escalation Procedures**: Tiered response escalation
- **Automated Response**: Immediate threat mitigation actions
- **Communication**: Stakeholder notification and updates
