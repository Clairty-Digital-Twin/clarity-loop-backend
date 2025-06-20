# Security Procedures & Implementation Documentation

## Overview

This document outlines the security implementations completed for the Clarity Loop Backend and provides procedures for administrators managing the system.

## Completed Security Implementations

### 1. AWS Cognito Self-Signup Disabled ✅ COMPLETED

#### Configuration Changes Made

- **Application Level Protection**: `ENABLE_SELF_SIGNUP=false` in environment configuration
- **AWS Cognito Level Protection**: Self-signup disabled directly in AWS Cognito console

#### Security Benefits

- **Dual-layer protection**: Both application and AWS service level blocks
- **403 Forbidden**: Application-level protection returns proper error codes
- **500 Server Error**: AWS Cognito-level protection prevents service-level registration
- **Complete lockdown**: Registration endpoint secured at multiple levels

#### Verification Status

✅ App-level protection tested and confirmed  
✅ AWS Cognito-level protection tested and confirmed  
✅ Both security layers active and working perfectly

---

### 2. Security Headers Middleware ✅ COMPLETED

#### Implementation

- **Location**: `src/clarity/middleware/security_headers.py`
- **Registration**: Integrated in main application middleware stack

#### Headers Configured

```python
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Content-Security-Policy: default-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
```

#### Security Benefits

- **XSS Protection**: Prevents cross-site scripting attacks
- **Clickjacking Protection**: X-Frame-Options prevents embedding
- **HTTPS Enforcement**: HSTS header enforces secure connections
- **Content Type Security**: Prevents MIME type confusion attacks

---

### 3. AWS WAF Rate Limiting ✅ COMPLETED

#### Implementation Components

**1. Centralized Configuration (`ops/env.sh`)**:

```bash
export REGION=us-east-1
export ALB_NAME="clarity-alb"
export ALB_ARN="$(aws elbv2 describe-load-balancers --names "$ALB_NAME" --query 'LoadBalancers[0].LoadBalancerArn' --output text --region "$REGION")"
export ALB_DNS="$(aws elbv2 describe-load-balancers --names "$ALB_NAME" --query 'LoadBalancers[0].DNSName' --output text --region "$REGION")"
```

**2. Idempotent Deployment Script (`ops/deploy-waf-rate-limiting.sh`)**:

- Safe re-runs without conflicts
- Automatic WebACL creation and ALB association
- Handles existing configurations gracefully

**3. Rate Limiting Configuration**:

- **Rate Limit**: 100 requests per 5-minute window
- **Action**: Block requests exceeding limit
- **Scope**: Regional (ALB protection)
- **CloudWatch Logging**: Enabled for monitoring

**4. Automated Testing**:

- **HTTPS Testing**: Validates WAF only evaluates post-TLS termination
- **Rate Limit Validation**: Confirms 100 req/5min enforcement
- **CI Integration**: `github/workflows/security.yml` smoke tests

#### Security Benefits

- **DDoS Protection**: Prevents request flooding attacks
- **Automated Response**: No manual intervention required
- **CloudWatch Integration**: Real-time monitoring and alerting
- **Load Balancer Protection**: Shields backend services

---

## Administrator Procedures

### User Account Management

#### Creating New User Accounts (Manual Process)

Since self-signup is disabled, new users must be created manually by administrators:

**AWS Cognito Console Method**:

1. Log into AWS Console
2. Navigate to Cognito User Pools
3. Select the Clarity backend user pool
4. Click "Create user"
5. Provide required user details
6. Set temporary password (user will be required to change on first login)

**AWS CLI Method**:

```bash
aws cognito-idp admin-create-user \
  --user-pool-id <USER_POOL_ID> \
  --username <USERNAME> \
  --user-attributes Name=email,Value=<EMAIL> \
  --temporary-password <TEMP_PASSWORD> \
  --message-action SUPPRESS
```

#### User Account Lifecycle

1. **Creation**: Admin creates account with temporary password
2. **First Login**: User must change password
3. **Verification**: Email verification may be required
4. **Access**: User can access application after setup

### Security Monitoring

#### WAF Monitoring

- **CloudWatch Logs**: Monitor `/aws/wafv2/clarity-backend-rate-limiting`
- **Rate Limit Events**: Watch for blocked requests in logs
- **ALB Health**: Monitor load balancer status

#### Security Headers Verification

```bash
curl -I https://<ALB_DNS>/health
# Verify all security headers are present
```

#### Key Metrics to Monitor

- **403/429 Responses**: Rate limiting activation
- **Failed Login Attempts**: Authentication monitoring  
- **Unusual Traffic Patterns**: Potential attack indicators

### Emergency Procedures

#### High Rate Limit Events

1. **Immediate**: Review CloudWatch logs for attack patterns
2. **Assess**: Determine if legitimate traffic or attack
3. **Adjust**: Temporarily lower rate limits if under attack
4. **Investigate**: Check source IPs and request patterns
5. **Report**: Document incident and response

#### Security Header Issues

1. **Verify**: Check if headers are being set correctly
2. **Test**: Use browser dev tools or curl to inspect
3. **Restart**: Restart application if middleware issues
4. **Escalate**: Contact development team if persistent

### Support Team Guidelines

#### Common User Issues

**"I can't create an account"**:

- **Response**: "Account creation is now handled by our administrators for security. Please contact your organization's admin or our support team."
- **Action**: Direct to account request process

**"I'm getting blocked/rate limited"**:

- **Response**: "We have security measures to protect our service. Please wait a few minutes and try again."
- **Action**: Check if legitimate user or potential attack

**"The site seems slow"**:

- **Check**: Review WAF logs for blocks
- **Verify**: Confirm ALB health status
- **Escalate**: Contact infrastructure team if needed

---

## Security Testing Procedures

### Pre-Deployment Checklist

- [ ] Security headers middleware active
- [ ] WAF rules properly configured  
- [ ] Rate limits set appropriately
- [ ] Self-signup disabled at both levels
- [ ] Monitoring and alerting configured

### Regular Security Audits

- **Weekly**: Review CloudWatch logs for anomalies
- **Monthly**: Test rate limiting functionality
- **Quarterly**: Comprehensive security header audit
- **Annually**: Full penetration testing

---

## Technical Implementation Details

### Code Locations

- **Security Headers**: `src/clarity/middleware/security_headers.py`
- **Auth Configuration**: Environment variables and AWS Cognito
- **WAF Configuration**: `ops/aws-waf-rate-limiting.json`
- **Deployment Scripts**: `ops/deploy-waf-rate-limiting.sh`

### Environment Variables

```bash
ENABLE_SELF_SIGNUP=false  # Application-level protection
AWS_REGION=us-east-1
ALB_NAME=clarity-alb
```

### Dependencies

- **AWS CLI**: Required for WAF deployment
- **Proper IAM Permissions**: For WAF and Cognito management
- **CloudWatch Access**: For monitoring and logging

---

## Contact Information

**For Security Issues**:

- Development Team: [Contact Info]
- Infrastructure Team: [Contact Info]
- Security Team: [Contact Info]

**For Account Management**:

- Admin Team: [Contact Info]
- Support Team: [Contact Info]

---

*Last Updated: December 2024*  
*Version: 1.0*  
*Status: Production Ready* ✅
