# PAT Model Observability

This module provides comprehensive observability for PAT (Proxy Actigraphy Transformer) model operations, including metrics collection, health monitoring, and alerting.

## Overview

The observability system tracks:
- Model loading operations (attempts, failures, duration)
- Checksum verification for security
- Cache performance metrics
- S3 download operations
- Model validation results
- Fallback operations
- Security violations

## Metrics

### Model Loading Metrics

- `clarity_pat_model_load_attempts_total` - Total load attempts (labels: model_size, version, source)
- `clarity_pat_model_load_failures_total` - Failed load attempts (labels: model_size, version, error_type)
- `clarity_pat_model_load_duration_seconds` - Load duration histogram

### Security Metrics

- `clarity_pat_checksum_verification_attempts_total` - Checksum verification attempts
- `clarity_pat_checksum_verification_failures_total` - Failed verifications (security violations)
- `clarity_pat_security_violations_total` - All security violations (labels: violation_type, severity)

### Cache Metrics

- `clarity_pat_model_cache_hits_total` - Cache hits
- `clarity_pat_model_cache_misses_total` - Cache misses
- `clarity_pat_model_cache_size_count` - Current number of cached models
- `clarity_pat_model_cache_memory_bytes` - Estimated cache memory usage

### Progress Tracking

- `clarity_pat_model_loading_progress_ratio` - Real-time loading progress (0.0-1.0)
  - Stages: download, checksum, load, validate

## API Endpoints

### Health Check Endpoints

#### Basic Health Check
```
GET /api/v1/health
```
Simple health check for load balancers.

#### Readiness Check
```
GET /api/v1/health/ready
```
Comprehensive readiness check including database, storage, and model availability.

#### PAT Model Health
```
GET /api/v1/health/pat
Authorization: Bearer <token>
```
Detailed PAT model health status including:
- Overall health score (0-100)
- Loaded models information
- Cache status
- Integrity verification results
- Recent failures
- Active alerts

#### PAT Metrics Summary
```
GET /api/v1/health/metrics/pat
Authorization: Bearer <token>
```
Summary of key PAT model metrics.

## Integration with Model Loader

The metrics are integrated into the `PATModelLoader` class:

```python
from clarity.ml.pat_model_loader import PATModelLoader

# Metrics are automatically collected during operations
loader = PATModelLoader(model_dir, s3_service)
model = await loader.load_model(ModelSize.SMALL)
```

## Alerts

Critical alerts include:
- **Checksum verification failures** - Potential security breach
- **High failure rate** - Model loading failures > 10%
- **Low health score** - Overall system health < 70
- **Security violations** - Any critical security events

See `alerts.yaml` for the complete Prometheus alert configuration.

## Dashboard Integration

The metrics are designed to work with Grafana dashboards. Key panels should include:

1. **Model Load Performance**
   - Load success rate
   - Average load time by model size
   - Cache hit rate

2. **Security Overview**
   - Checksum verification status
   - Security violations over time
   - Failed verification attempts

3. **System Health**
   - Overall health score gauge
   - Active alerts
   - Model version tracking

4. **Resource Usage**
   - Cache memory usage
   - S3 download bandwidth
   - Number of cached models

## Security Considerations

1. **Checksum Verification**: All models must pass checksum verification before use
2. **Alert on Failures**: Security violations trigger immediate alerts
3. **Audit Trail**: All model operations are logged with metrics
4. **Version Tracking**: Current model versions are tracked for compliance

## Example Usage

### Monitoring Model Load Operations

```python
async with track_model_load("small", "v1.0", "s3") as ctx:
    # Load model
    model = await load_model_from_s3()
    ctx["success"] = True
```

### Recording Security Violations

```python
if checksum_mismatch:
    record_security_violation("checksum_mismatch", "small", "critical")
```

### Updating Loading Progress

```python
update_loading_progress("small", "v1.0", "download", 0.5)  # 50% complete
```

## Production Deployment

1. **Prometheus Configuration**
   - Ensure Prometheus scrapes `/metrics` endpoint
   - Deploy alert rules from `alerts.yaml`

2. **AlertManager Setup**
   - Configure notification channels (PagerDuty, Slack, etc.)
   - Set up escalation policies for critical alerts

3. **Grafana Dashboards**
   - Import provided dashboard JSON
   - Customize thresholds based on your SLOs

4. **Health Check Integration**
   - Configure load balancer health checks to use `/api/v1/health`
   - Set up Kubernetes readiness probes with `/api/v1/health/ready`

## Troubleshooting

### High Failure Rate
1. Check model file integrity
2. Verify S3 permissions and connectivity
3. Review error logs for specific failure types

### Low Cache Hit Rate
1. Increase cache TTL if appropriate
2. Review model usage patterns
3. Consider pre-loading frequently used models

### Security Violations
1. Immediately investigate checksum mismatches
2. Verify model source integrity
3. Check for unauthorized model modifications
4. Review access logs