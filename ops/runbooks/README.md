# Operations Runbooks & SLOs

Comprehensive incident response procedures and service level objectives for the Clarity Loop Backend.

## Service Level Objectives (SLOs)

### API Gateway SLOs

| Metric | SLO | Measurement Window | Error Budget |
|--------|-----|-------------------|---------------|
| **Availability** | 99.9% | 30 days | 43 minutes/month |
| **Response Time** | 95th percentile < 200ms | 5 minutes | 5% above threshold |
| **Error Rate** | < 0.1% | 5 minutes | 0.1% of requests |
| **Throughput** | > 1000 requests/minute | 1 minute | <1000 req/min for >1 min |

### ML Service SLOs

| Metric | SLO | Measurement Window | Error Budget |
|--------|-----|-------------------|---------------|
| **Processing Time** | 95th percentile < 500ms | 5 minutes | 5% above threshold |
| **Model Accuracy** | > 85% sleep detection | 24 hours | <85% for >1 hour |
| **Cold Start** | < 2 seconds | Per request | >2s for >5% requests |
| **Queue Depth** | < 100 pending jobs | 1 minute | >100 for >5 minutes |

### Data Pipeline SLOs

| Metric | SLO | Measurement Window | Error Budget |
|--------|-----|-------------------|---------------|
| **End-to-End Latency** | 95th percentile < 30s | 5 minutes | 5% above threshold |
| **Data Loss** | 0% | 24 hours | Zero tolerance |
| **Pub/Sub Delivery** | 99.9% | 1 hour | 0.1% message loss |
| **Insight Generation** | 99% success rate | 1 hour | 1% failure rate |

### Infrastructure SLOs

| Metric | SLO | Measurement Window | Error Budget |
|--------|-----|-------------------|---------------|
| **Firestore Availability** | 99.95% | 30 days | 21 minutes/month |
| **Storage Durability** | 99.999999999% | Annual | 11 9's standard |
| **Network Latency** | < 10ms within region | 5 minutes | Regional baseline |

## Alert Thresholds

### Critical Alerts (Immediate Response)

#### Service Availability
```yaml
# API Gateway Down
alert: api_gateway_down
condition: up{job="api-gateway"} == 0
for: 30s
severity: critical
description: "API Gateway is completely unavailable"
```

#### Data Loss Risk
```yaml
# Pub/Sub Message Backlog
alert: pubsub_message_backlog_critical
condition: pubsub_subscription_num_undelivered_messages > 1000
for: 5m
severity: critical
description: "Pub/Sub backlog exceeding critical threshold - data loss risk"
```

#### Security Breach
```yaml
# Authentication Failures
alert: auth_failure_spike
condition: rate(auth_failures_total[5m]) > 10
for: 2m
severity: critical
description: "Potential security breach - authentication failure spike"
```

### Warning Alerts (Response within 1 hour)

#### Performance Degradation
```yaml
# API Response Time
alert: api_response_time_high
condition: histogram_quantile(0.95, api_request_duration_seconds) > 0.2
for: 5m
severity: warning
description: "API 95th percentile response time exceeding SLO"
```

#### Resource Utilization
```yaml
# CPU Utilization
alert: high_cpu_utilization
condition: rate(cpu_usage_seconds_total[5m]) > 0.8
for: 10m
severity: warning
description: "High CPU utilization detected"
```

#### ML Model Performance
```yaml
# Model Accuracy Drop
alert: model_accuracy_degradation
condition: ml_model_accuracy < 0.85
for: 1h
severity: warning
description: "ML model accuracy below SLO threshold"
```

## Incident Response Procedures

### Severity Levels

**P0 - Critical**: Service completely unavailable or data loss
- **Response Time**: 5 minutes
- **Escalation**: Immediate on-call notification
- **Communication**: Status page update within 10 minutes

**P1 - High**: Significant performance degradation
- **Response Time**: 30 minutes
- **Escalation**: Primary on-call engineer
- **Communication**: Internal stakeholders notified

**P2 - Medium**: Minor issues affecting subset of users
- **Response Time**: 2 hours during business hours
- **Escalation**: Assigned to team queue
- **Communication**: Engineering team notification

### On-Call Rotation

```yaml
# PagerDuty Configuration
primary_oncall:
  - rotation: weekly
  - escalation_delay: 5_minutes
  
secondary_oncall:
  - rotation: weekly
  - escalation_delay: 15_minutes
  
manager_escalation:
  - escalation_delay: 30_minutes
```

## Common Failure Modes & Responses

### 1. Pub/Sub Message Backlog

**Symptoms**:
- Messages accumulating in subscription
- End-to-end latency increasing
- Users not receiving insights

**Diagnosis**:
```bash
# Check subscription backlog
gcloud pubsub subscriptions describe healthkit-processing-subscription

# Check consumer lag
gcloud monitoring metrics list --filter="resource.type=pubsub_subscription"
```

**Immediate Actions**:
1. Scale up ML service instances
2. Check for processing errors in logs
3. Verify Vertex AI quotas not exceeded
4. Monitor message processing rate

**Resolution**:
```bash
# Scale Cloud Run service
gcloud run services update clarity-ml-service \
  --min-instances=5 \
  --max-instances=50 \
  --region=us-central1

# Check error rates
gcloud logging read "resource.type=cloud_run_revision AND \
  severity>=ERROR" --limit=50
```

### 2. PAT Model Cold Start Issues

**Symptoms**:
- First request to ML service taking >2 seconds
- Intermittent timeouts
- Users experiencing delays

**Diagnosis**:
```bash
# Check Cloud Run cold start metrics
gcloud monitoring metrics list --filter="resource.type=cloud_run_revision AND \
  metric.type=run.googleapis.com/container/startup_latency"

# Review instance allocation
gcloud run services describe clarity-ml-service --region=us-central1
```

**Immediate Actions**:
1. Increase minimum instances to prevent cold starts
2. Implement keep-alive requests
3. Check Vertex AI model loading time

**Resolution**:
```bash
# Set minimum instances
gcloud run services update clarity-ml-service \
  --min-instances=2 \
  --region=us-central1

# Verify model caching
curl http://ml-service/health/model-status
```

### 3. Vertex AI Quota Exceeded

**Symptoms**:
- ML requests failing with quota errors
- 429 rate limit responses
- Backup processing queue growing

**Diagnosis**:
```bash
# Check current quota usage
gcloud compute project-info describe --format="value(quotas[].usage,quotas[].limit)"

# Review Vertex AI usage
gcloud ai quotas list --region=us-central1
```

**Immediate Actions**:
1. Implement request queuing with exponential backoff
2. Request quota increase from Google Cloud
3. Activate secondary region if configured

**Resolution**:
```bash
# Submit quota increase request
gcloud support cases create \
  --display-name="Vertex AI Quota Increase" \
  --description="Need increased quota for production workload"

# Monitor quota utilization
gcloud monitoring dashboards list --filter="vertex_ai_quota"
```

### 4. Firestore Write Conflicts

**Symptoms**:
- Transaction failures
- Data inconsistency reports
- Increased error rates

**Diagnosis**:
```bash
# Check Firestore metrics
gcloud firestore operations list

# Review transaction conflict rates
gcloud logging read "resource.type=firestore_database AND \
  jsonPayload.error_code=ABORTED"
```

**Immediate Actions**:
1. Implement exponential backoff for retries
2. Review transaction structure for optimization
3. Check for concurrent write patterns

**Resolution**:
```bash
# Optimize document structure
# Review batch write operations
# Implement proper transaction retry logic
```

### 5. Authentication Service Outage

**Symptoms**:
- Users cannot authenticate
- 401/403 errors across all endpoints
- Firebase Auth console showing issues

**Diagnosis**:
```bash
# Check Firebase Auth status
curl https://status.firebase.google.com/

# Review authentication error logs
gcloud logging read "resource.type=cloud_run_revision AND \
  jsonPayload.auth_error=true"
```

**Immediate Actions**:
1. Check Firebase project configuration
2. Verify service account credentials
3. Implement graceful degradation if possible

**Resolution**:
```bash
# Verify Firebase project settings
firebase projects:list

# Check service account permissions
gcloud iam service-accounts get-iam-policy SERVICE_ACCOUNT_EMAIL
```

## Manual Failover Procedures

### 1. Region Failover

**When to Use**: Primary region (us-central1) completely unavailable

**Steps**:
```bash
# 1. Update DNS to point to secondary region
gcloud dns record-sets transaction start --zone=clarity-zone

gcloud dns record-sets transaction add \
  --name=api.clarity-loop.com \
  --ttl=300 \
  --type=A \
  --zone=clarity-zone \
  NEW_REGION_IP

gcloud dns record-sets transaction execute --zone=clarity-zone

# 2. Scale up services in secondary region
gcloud run services update clarity-api-gateway \
  --region=us-east1 \
  --min-instances=3

# 3. Update Firestore region preference
# (if multi-region setup available)

# 4. Verify health checks pass
curl https://api-us-east1.clarity-loop.com/health
```

### 2. Database Failover

**When to Use**: Primary Firestore instance unavailable

**Steps**:
```bash
# 1. Switch to backup database (if configured)
export FIRESTORE_DATABASE_ID="clarity-backup"

# 2. Update application configuration
kubectl set env deployment/api-gateway \
  FIRESTORE_DATABASE_ID="clarity-backup"

# 3. Verify data consistency
./scripts/verify-data-consistency.sh

# 4. Monitor application behavior
./scripts/health-check-extended.sh
```

### 3. ML Service Failover

**When to Use**: Vertex AI unavailable or PAT model failing

**Steps**:
```bash
# 1. Switch to fallback model (if available)
export ML_MODEL_FALLBACK=true

# 2. Use cached predictions for known patterns
export USE_PREDICTION_CACHE=true

# 3. Queue requests for later processing
export QUEUE_ML_REQUESTS=true

# 4. Notify users of degraded functionality
./scripts/update-status-page.sh "ML processing temporarily degraded"
```

## Rollback Procedures

### Application Rollback

```bash
# 1. Identify last known good version
gcloud run revisions list --service=clarity-api-gateway --region=us-central1

# 2. Route traffic to previous revision
gcloud run services update-traffic clarity-api-gateway \
  --to-revisions=REVISION_NAME=100 \
  --region=us-central1

# 3. Verify rollback success
curl https://api.clarity-loop.com/health

# 4. Monitor for issues
./scripts/post-rollback-monitoring.sh
```

### Database Schema Rollback

```bash
# 1. Stop all write operations
./scripts/maintenance-mode-on.sh

# 2. Run rollback migration
alembic downgrade -1

# 3. Verify data integrity
./scripts/verify-data-integrity.sh

# 4. Resume operations
./scripts/maintenance-mode-off.sh
```

### Infrastructure Rollback

```bash
# 1. Revert Terraform changes
cd terraform/environments/production
git checkout HEAD~1
terraform plan
terraform apply

# 2. Verify infrastructure state
terraform validate
terraform refresh

# 3. Run health checks
./scripts/infrastructure-health-check.sh
```

## Monitoring Dashboards

### Key Performance Indicators (KPIs)

**URL**: `https://console.cloud.google.com/monitoring/dashboards/custom/clarity-kpi`

**Panels**:
- API Gateway request rate and latency
- ML Service processing time and accuracy
- Pub/Sub message throughput and lag
- Firestore read/write operations
- Error rate across all services
- User-facing SLO burn rate

### Error Budget Dashboard

**URL**: `https://console.cloud.google.com/monitoring/dashboards/custom/clarity-error-budget`

**Panels**:
- Current error budget consumption
- Burn rate alerts
- SLO compliance trends
- Incident impact on error budget
- Projected budget exhaustion

### Infrastructure Health

**URL**: `https://console.cloud.google.com/monitoring/dashboards/custom/clarity-infrastructure`

**Panels**:
- Cloud Run instance utilization
- Vertex AI quota usage
- Firestore performance metrics
- Network latency and throughput
- Storage utilization and costs

## Communication Templates

### Incident Status Update

```
🚨 INCIDENT UPDATE - [INCIDENT_ID]

Status: [INVESTIGATING/IDENTIFIED/MONITORING/RESOLVED]
Severity: [P0/P1/P2]
Started: [TIMESTAMP]
Duration: [DURATION]

Impact: [BRIEF_DESCRIPTION]
Affected Services: [LIST_SERVICES]
Users Affected: [PERCENTAGE/NUMBER]

Current Actions:
- [ACTION_1]
- [ACTION_2]

Next Update: [TIMESTAMP]
Incident Commander: [NAME]
```

### Post-Incident Report Template

```
# Post-Incident Report - [INCIDENT_ID]

## Summary
- **Date**: [DATE]
- **Duration**: [TOTAL_DURATION]  
- **Severity**: [P0/P1/P2]
- **Services Affected**: [LIST]
- **Users Impacted**: [PERCENTAGE]

## Timeline
- [TIMESTAMP]: Incident detected
- [TIMESTAMP]: Investigation started
- [TIMESTAMP]: Root cause identified
- [TIMESTAMP]: Fix implemented
- [TIMESTAMP]: Services restored

## Root Cause
[DETAILED_EXPLANATION]

## Impact Assessment
- **User Impact**: [DESCRIPTION]
- **Revenue Impact**: [IF_APPLICABLE]
- **SLO Impact**: [ERROR_BUDGET_CONSUMED]

## Action Items
- [ ] [ACTION_1] - Owner: [NAME] - Due: [DATE]
- [ ] [ACTION_2] - Owner: [NAME] - Due: [DATE]

## Lessons Learned
[KEY_TAKEAWAYS]
```

## Emergency Contacts

### On-Call Escalation

1. **Primary On-Call**: [PHONE/SLACK]
2. **Secondary On-Call**: [PHONE/SLACK]  
3. **Engineering Manager**: [PHONE/SLACK]
4. **VP Engineering**: [PHONE/SLACK]

### External Dependencies

1. **Google Cloud Support**: [SUPPORT_CASE_SYSTEM]
2. **Firebase Support**: [SUPPORT_CHANNEL]
3. **Anthropic Support**: [SUPPORT_EMAIL]
4. **Third-party Monitoring**: [VENDOR_CONTACT]

---

**Goal**: Sub-5-minute incident detection and response  
**Standard**: 99.9% uptime with graceful degradation  
**Recovery**: Automated rollback within 2 minutes for critical issues
