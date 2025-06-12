# CLARITY-AI Monitoring & Observability

**Enterprise-grade monitoring for health data platform**

## 📊 **Monitoring Stack Overview**

### **Core Components**
- **AWS CloudWatch** - Logs, metrics, and alarms
- **Prometheus** - Custom metrics collection  
- **Grafana** - Visualization dashboards
- **ECS Service Discovery** - Dynamic target discovery
- **Custom Health Endpoints** - Application-specific monitoring

### **Architecture**

```mermaid
flowchart TD
    subgraph Applications ["🚀 APPLICATIONS"]
        A[CLARITY-AI API<br/>FastAPI Application<br/>Port 8000]
        B[PAT ML SERVICE<br/>Transformer Models<br/>Health Analytics]
        C[GEMINI INTEGRATION<br/>Natural Language<br/>Insight Generation]
    end
    
    subgraph Monitoring ["📊 MONITORING LAYER"]
        D[PROMETHEUS<br/>Metrics Collection<br/>Custom dashboards]
        E[CLOUDWATCH<br/>AWS Native Logs<br/>Infrastructure metrics]
        F[GRAFANA<br/>Visualization<br/>Alert management]
    end
    
    subgraph Alerting ["🚨 ALERTING"]
        G[SLACK NOTIFICATIONS<br/>Critical alerts<br/>Team collaboration]
        H[EMAIL ALERTS<br/>Stakeholder updates<br/>SLA breaches]
        I[PAGERDUTY<br/>On-call rotation<br/>Incident management]
    end
    
    A --> D
    B --> D
    C --> D
    A --> E
    B --> E
    C --> E
    D --> F
    E --> F
    F --> G
    F --> H
    F --> I
    
    classDef apps fill:#ff9900,stroke:#e65100,stroke-width:3px,color:white
    classDef monitoring fill:#4caf50,stroke:#1b5e20,stroke-width:3px,color:white
    classDef alerting fill:#2196f3,stroke:#0d47a1,stroke-width:3px,color:white
    
    class A,B,C apps
    class D,E,F monitoring
    class G,H,I alerting
```

## 🏥 **Health Checks & Service Monitoring**

### **Application Health Endpoints**

**Primary Health Check:**
```bash
# Basic health status
curl http://localhost:8000/health
```

**Response Format:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-16T10:30:00Z",
  "version": "1.2.0",
  "environment": "production",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "pat_model": "healthy",
    "gemini_api": "healthy"
  },
  "metrics": {
    "response_time_ms": 45,
    "memory_usage_mb": 1024,
    "cpu_usage_percent": 25.5
  }
}
```

**Detailed Health Check:**
```bash
# Comprehensive service status
curl http://localhost:8000/health/detailed
```

**Database Health:**
```bash
# DynamoDB connectivity test
curl http://localhost:8000/health/database
```

**ML Model Health:**
```bash
# PAT model inference test
curl http://localhost:8000/health/models
```

### **ECS Health Monitoring**

**Task Health Configuration:**
```json
{
  "healthCheck": {
    "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
    "interval": 30,
    "timeout": 10,
    "retries": 3,
    "startPeriod": 60
  }
}
```

**Service-Level Monitoring:**
```bash
# Monitor ECS service health
aws ecs describe-services \
  --cluster clarity-backend-cluster \
  --services clarity-backend-service \
  --query "services[0].deployments[0].runningCount"
```

## 📈 **Metrics Collection**

### **Application Metrics**

**FastAPI Built-in Metrics:**
```python
# Custom metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return {
        "requests_total": metrics.request_count,
        "requests_duration_seconds": metrics.request_duration,
        "active_users": metrics.active_users,
        "ml_inference_duration": metrics.ml_processing_time,
        "database_connections": metrics.db_pool_size
    }
```

**Key Performance Indicators:**
- **Response Time**: P50, P95, P99 latency
- **Throughput**: Requests per second
- **Error Rate**: 4xx/5xx error percentage
- **ML Processing Time**: PAT inference duration
- **Database Performance**: Query execution time

### **Infrastructure Metrics**

**ECS Container Metrics:**
- CPU Utilization (target: <70%)
- Memory Utilization (target: <80%)
- Network I/O
- Task count and health

**AWS CloudWatch Metrics:**
```bash
# Get CPU utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=clarity-backend-service \
  --start-time 2025-01-01T00:00:00Z \
  --end-time 2025-01-01T23:59:59Z \
  --period 300 \
  --statistics Average,Maximum
```

### **Business Metrics**

**Health Data Processing:**
- Users analyzed per day
- PAT model inference success rate
- Gemini API response quality
- Data upload success rate

**System Performance:**
- Authentication success rate
- Health insight generation time
- API endpoint availability
- Data pipeline throughput

## 🔔 **Alerting Configuration**

### **Critical Alerts (Immediate Response)**

**Service Down:**
```yaml
# CloudWatch Alarm
AlertName: "CLARITY-AI Service Down"
Threshold: "< 1 healthy task"
Period: "60 seconds"
Evaluation: "2 consecutive periods"
Action: "SNS → Slack + PagerDuty"
```

**High Error Rate:**
```yaml
AlertName: "High Error Rate"
Threshold: "> 5% error rate"
Period: "300 seconds"
Evaluation: "2 consecutive periods"
Action: "SNS → Slack notification"
```

**Database Connection Issues:**
```yaml
AlertName: "Database Connection Failure"
Threshold: "Database health check fails"
Period: "60 seconds"
Evaluation: "1 consecutive period"
Action: "SNS → Immediate PagerDuty"
```

### **Warning Alerts (Next Business Day)**

**High CPU Usage:**
```yaml
AlertName: "High CPU Utilization"
Threshold: "> 80% CPU for 5 minutes"
Period: "300 seconds"
Evaluation: "1 consecutive period"
Action: "SNS → Slack notification"
```

**Memory Pressure:**
```yaml
AlertName: "High Memory Usage"
Threshold: "> 90% memory for 5 minutes"
Period: "300 seconds"
Evaluation: "1 consecutive period"
Action: "SNS → Email notification"
```

## 📋 **Logging Strategy**

### **Structured Logging**

**Log Format:**
```json
{
  "timestamp": "2025-01-16T10:30:00Z",
  "level": "INFO",
  "service": "clarity-api",
  "trace_id": "abc123",
  "user_id": "user_456",
  "endpoint": "/api/v1/analyze",
  "duration_ms": 1250,
  "status_code": 200,
  "message": "Health analysis completed successfully"
}
```

**Log Categories:**
- **API Requests** - All HTTP requests/responses
- **ML Processing** - PAT model inference events
- **Authentication** - Login/logout/token events
- **Database** - Query performance and errors
- **External APIs** - Gemini API calls and responses

### **Log Aggregation**

**CloudWatch Log Groups:**
```bash
# Application logs
/ecs/clarity-backend

# Error logs (filtered)
/ecs/clarity-backend/errors

# Audit logs (security events)
/ecs/clarity-backend/audit
```

**Log Retention:**
- **Application Logs**: 30 days
- **Error Logs**: 90 days  
- **Audit Logs**: 1 year (compliance requirement)

### **Log Querying**

**Common Log Queries:**
```bash
# Find all errors in the last hour
aws logs filter-log-events \
  --log-group-name /ecs/clarity-backend \
  --start-time $(date -d "1 hour ago" +%s)000 \
  --filter-pattern "ERROR"

# Track specific user activity
aws logs filter-log-events \
  --log-group-name /ecs/clarity-backend \
  --filter-pattern "{ $.user_id = \"user_123\" }"

# Monitor ML processing performance
aws logs filter-log-events \
  --log-group-name /ecs/clarity-backend \
  --filter-pattern "{ $.duration_ms > 5000 }"
```

## 📊 **Grafana Dashboards**

### **Executive Dashboard**

**Key Metrics:**
- Daily active users
- Health analyses completed
- System uptime percentage
- Revenue impact metrics

### **Operations Dashboard**

**Infrastructure Metrics:**
- CPU/Memory utilization
- Request latency percentiles
- Error rate trends
- Container health status

### **ML Performance Dashboard**

**Model Metrics:**
- PAT inference time
- Gemini API response time
- Model accuracy metrics
- Data processing throughput

## 🚨 **Incident Response**

### **Severity Levels**

**SEV-1 (Critical):**
- Complete service outage
- Data loss or corruption
- Security breach
- **Response Time**: 15 minutes

**SEV-2 (High):**
- Partial service degradation
- High error rates (>10%)
- Authentication issues
- **Response Time**: 1 hour

**SEV-3 (Medium):**
- Performance degradation
- Non-critical feature failure
- **Response Time**: 4 hours

**SEV-4 (Low):**
- Documentation issues
- Minor UI problems
- **Response Time**: Next business day

### **On-Call Rotation**

**Primary On-Call:**
- Platform Engineering Team
- 24/7 coverage
- PagerDuty escalation

**Secondary On-Call:**
- ML Engineering Team
- Business hours coverage
- Model-specific issues

## 🔧 **Monitoring Setup**

### **Prometheus Configuration**

**Prometheus Config (`ops/prometheus.yml`):**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'clarity-backend'
    ec2_sd_configs:
      - region: us-east-1
        port: 8000
    relabel_configs:
      - source_labels: [__meta_ec2_tag_Service]
        target_label: service
```

### **Grafana Setup**

**Data Sources:**
- Prometheus (application metrics)
- CloudWatch (infrastructure metrics)
- CloudWatch Logs (log analysis)

**Dashboard Templates:**
```bash
# Import pre-built dashboards
curl -H "Content-Type: application/json" \
  -d @ops/grafana/dashboards/clarity-overview.json \
  http://admin:admin@grafana:3000/api/dashboards/db
```

### **Alert Manager Configuration**

**Slack Integration:**
```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'slack-critical'

receivers:
- name: 'slack-critical'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#clarity-alerts'
    title: 'CLARITY-AI Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## 📝 **Runbooks**

### **Common Operations**

**Service Restart:**
```bash
# Force new deployment (restart all tasks)
aws ecs update-service \
  --cluster clarity-backend-cluster \
  --service clarity-backend-service \
  --force-new-deployment
```

**Scale Service:**
```bash
# Scale to 4 instances
aws ecs update-service \
  --cluster clarity-backend-cluster \
  --service clarity-backend-service \
  --desired-count 4
```

**Check Service Health:**
```bash
# Run comprehensive health check
./ops/monitor-deployment.sh
```

---

**Monitoring Status**: ✅ Production Ready  
**Last Updated**: January 2025  
**SLA Target**: 99.9% uptime 