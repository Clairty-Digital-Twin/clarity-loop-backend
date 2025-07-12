# Enhanced Feature Flag System

## Overview

The CLARITY Digital Twin Platform includes an enhanced feature flag system with auto-refresh capabilities, providing:

- **Automatic Configuration Refresh**: Periodic and event-driven updates
- **Pub/Sub Integration**: Real-time configuration changes via Redis
- **Circuit Breaker Pattern**: Graceful handling of config store failures
- **Comprehensive Metrics**: Prometheus metrics for monitoring
- **Thread-Safe Operations**: Safe concurrent access in multi-threaded environments

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Feature Flag    │  │ Health Check │  │   Metrics    │  │
│  │   Decorator     │  │   Endpoint   │  │  Dashboard   │  │
│  └────────┬────────┘  └──────┬───────┘  └──────┬───────┘  │
│           │                   │                  │           │
│  ┌────────▼───────────────────▼──────────────────▼───────┐  │
│  │         Enhanced Feature Flag Manager                  │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
│  │  │  Cache   │  │  Refresh │  │ Circuit  │           │  │
│  │  │  Layer   │  │  Engine  │  │ Breaker  │           │  │
│  │  └──────────┘  └─────┬────┘  └──────────┘           │  │
│  └───────────────────────┼───────────────────────────────┘  │
└──────────────────────────┼───────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌──────────┐    ┌──────────────┐   ┌─────────────┐
   │  Redis   │    │ Config Store │   │ Prometheus  │
   │ Pub/Sub  │    │   (AWS SSM)  │   │   Metrics   │
   └──────────┘    └──────────────┘   └─────────────┘
```

## Features

### 1. Auto-Refresh Capabilities

The enhanced system supports multiple refresh modes:

```python
from clarity.core.feature_flags_enhanced import RefreshMode

# Refresh modes:
# - PERIODIC: Regular interval-based refresh
# - PUBSUB: Event-driven refresh via Redis pub/sub
# - BOTH: Combined periodic and event-driven
# - NONE: No automatic refresh (manual only)
```

### 2. Circuit Breaker Protection

Protects against cascading failures when the config store is unavailable:

```python
# Circuit breaker states:
# - CLOSED: Normal operation, requests pass through
# - OPEN: Failures exceeded threshold, requests fail fast
# - HALF-OPEN: Testing if service recovered

# Configuration:
circuit_breaker_failure_threshold=3  # Opens after 3 failures
circuit_breaker_recovery_timeout=30  # Tries recovery after 30s
```

### 3. Staleness Detection

Monitors configuration age and alerts when stale:

```python
# Staleness thresholds by environment:
# - Production: 5 minutes
# - Staging: 10 minutes
# - Development: 1 hour

if manager.is_config_stale():
    logger.warning("Using stale feature flag configuration")
```

### 4. Thread-Safe Operations

All operations are protected with thread locks:

```python
# Safe for concurrent access
with manager._lock:
    manager.config = new_config
    manager.clear_cache()
```

## Usage

### Basic Setup

```python
from clarity.core.config_aws import Settings
from clarity.core.feature_flags_integration import (
    create_enhanced_feature_flag_manager,
    setup_feature_flags_for_app,
)

# For FastAPI applications
app = FastAPI()
settings = Settings()
manager = setup_feature_flags_for_app(app, settings)

# For standalone usage
manager = create_enhanced_feature_flag_manager(settings)
```

### Checking Feature Flags

```python
from clarity.core.feature_flags_integration import (
    is_mania_risk_enabled,
    is_pat_model_v2_enabled,
    is_enhanced_security_enabled,
)

# Simple checks with auto-refresh
if is_mania_risk_enabled(user_id="123"):
    # Run mania risk analysis
    pass

# Direct manager usage
manager = get_enhanced_feature_flag_manager()
if manager.is_enabled("new_feature", user_id="123"):
    # Use new feature
    pass
```

### Manual Refresh

```python
# Synchronous refresh
success = manager.refresh()

# Asynchronous refresh
success = await manager.refresh_async()

# Check refresh status
health = get_feature_flag_health()
print(f"Config age: {health['config_age_seconds']}s")
print(f"Circuit breaker: {health['circuit_breaker_state']}")
```

## Configuration

### Environment Variables

```bash
# Refresh mode (periodic, pubsub, both, none)
FEATURE_FLAG_REFRESH_MODE=both

# Refresh interval in seconds
FEATURE_FLAG_REFRESH_INTERVAL=60

# Redis URL for pub/sub
REDIS_URL=redis://localhost:6379

# Environment (affects default behavior)
ENVIRONMENT=production
```

### Enhanced Configuration

```python
from clarity.core.feature_flags_enhanced import EnhancedFeatureFlagConfig

config = EnhancedFeatureFlagConfig(
    # Refresh settings
    refresh_interval_seconds=60,
    refresh_mode=RefreshMode.BOTH,
    
    # Redis pub/sub
    redis_url="redis://localhost:6379",
    pubsub_channel="feature_flags:config-changed",
    
    # Circuit breaker
    circuit_breaker_failure_threshold=3,
    circuit_breaker_recovery_timeout=30,
    
    # Staleness detection
    stale_config_threshold_seconds=300,
    
    # Metrics
    enable_metrics=True,
)
```

## Monitoring

### Prometheus Metrics

The system exports the following metrics:

```prometheus
# Refresh operations
feature_flag_refresh_success_total
feature_flag_refresh_failure_total{error_type="..."}
feature_flag_refresh_duration_seconds

# Configuration state
feature_flag_stale_config (0=fresh, 1=stale)
feature_flag_last_refresh_timestamp

# Circuit breaker
feature_flag_circuit_breaker_state (0=closed, 1=open, 2=half-open)
```

### Health Endpoint

```python
@app.get("/feature-flags/health")
async def feature_flag_health():
    return get_feature_flag_health()

# Response:
{
    "healthy": true,
    "config_age_seconds": 45.2,
    "config_stale": false,
    "circuit_breaker_state": "closed",
    "last_refresh": "2024-01-20T10:30:00Z",
    "refresh_failures": 0,
    "refresh_mode": "both",
    "cache_size": 12
}
```

### Grafana Dashboard

Import the dashboard configuration:

```python
from clarity.monitoring.feature_flag_metrics import router

app.include_router(router)

# Access dashboard config at:
# GET /metrics/feature-flags/grafana
```

## Best Practices

### 1. Environment-Specific Configuration

```python
# Production: Aggressive refresh, both periodic and pub/sub
if environment == "production":
    refresh_mode = RefreshMode.BOTH
    refresh_interval = 60  # 1 minute

# Staging: Periodic only, longer interval
elif environment == "staging":
    refresh_mode = RefreshMode.PERIODIC
    refresh_interval = 120  # 2 minutes

# Development: No auto-refresh
else:
    refresh_mode = RefreshMode.NONE
```

### 2. Graceful Degradation

```python
# Always provide sensible defaults
is_enabled = manager.is_enabled(
    "experimental_feature",
    user_id=user_id,
    default=False  # Safe default
)

# Handle stale configuration
if manager.is_config_stale():
    # Log warning but continue with cached values
    logger.warning("Using cached feature flags")
```

### 3. Circuit Breaker Tuning

```python
# Adjust thresholds based on config store reliability
config = EnhancedFeatureFlagConfig(
    # More tolerant for flaky networks
    circuit_breaker_failure_threshold=5,
    circuit_breaker_recovery_timeout=60,
)
```

### 4. Monitoring and Alerting

```yaml
# Prometheus alert rules
groups:
  - name: feature_flags
    rules:
      - alert: FeatureFlagConfigStale
        expr: feature_flag_stale_config == 1
        for: 5m
        annotations:
          summary: "Feature flag configuration is stale"
          
      - alert: FeatureFlagCircuitBreakerOpen
        expr: feature_flag_circuit_breaker_state == 1
        for: 2m
        annotations:
          summary: "Feature flag circuit breaker is open"
```

## Testing

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, patch

def test_refresh_with_circuit_breaker():
    """Test circuit breaker behavior."""
    manager = EnhancedFeatureFlagManager()
    
    # Simulate failures
    with patch.object(manager, "_fetch_config_from_store") as mock:
        mock.side_effect = Exception("Network error")
        
        # First two failures - circuit stays closed
        assert not manager.refresh()
        assert manager.get_circuit_breaker_state() == "closed"
        
        assert not manager.refresh()
        assert manager.get_circuit_breaker_state() == "closed"
        
        # Third failure - circuit opens
        assert not manager.refresh()
        assert manager.get_circuit_breaker_state() == "open"
```

### Integration Tests

```python
@pytest.mark.integration
async def test_pubsub_refresh():
    """Test pub/sub triggered refresh."""
    manager = EnhancedFeatureFlagManager(
        enhanced_config=EnhancedFeatureFlagConfig(
            refresh_mode=RefreshMode.PUBSUB
        )
    )
    
    # Publish config change event
    redis = await aioredis.create_redis_pool("redis://localhost:6379")
    await redis.publish("feature_flags:config-changed", "refresh")
    
    # Wait for refresh
    await asyncio.sleep(1)
    
    # Verify refresh occurred
    assert manager._last_refresh_time is not None
```

## Troubleshooting

### Common Issues

1. **Config not refreshing**
   - Check refresh mode configuration
   - Verify Redis connectivity for pub/sub
   - Check circuit breaker state
   - Review logs for refresh failures

2. **High memory usage**
   - Clear cache periodically: `manager.clear_cache()`
   - Reduce cache TTL in configuration
   - Monitor cache size via metrics

3. **Circuit breaker stuck open**
   - Check config store connectivity
   - Review failure threshold settings
   - Manually reset if needed (restart service)

4. **Stale configuration warnings**
   - Verify refresh interval is appropriate
   - Check for network issues
   - Ensure config store permissions

### Debug Logging

Enable debug logging for detailed information:

```python
import logging

logging.getLogger("clarity.core.feature_flags_enhanced").setLevel(logging.DEBUG)
```

## Migration Guide

### From Basic to Enhanced

```python
# Before (basic feature flags)
from clarity.core.feature_flags import get_feature_flag_manager
manager = get_feature_flag_manager()

# After (enhanced with auto-refresh)
from clarity.core.feature_flags_integration import (
    get_enhanced_feature_flag_manager
)
manager = get_enhanced_feature_flag_manager()

# The API remains the same
is_enabled = manager.is_enabled("feature_name")
```

### Adding to Existing Application

```python
# In your FastAPI startup
from clarity.core.feature_flags_integration import setup_feature_flags_for_app

@app.on_event("startup")
async def startup():
    # Set up enhanced feature flags
    setup_feature_flags_for_app(app, settings)
    
    # Your other startup tasks...
```

## Future Enhancements

1. **A/B Testing Integration**: Support for experiment allocation
2. **Feature Flag UI**: Web interface for flag management
3. **Audit Logging**: Track all flag changes and access
4. **Multi-Region Support**: Geo-distributed flag synchronization
5. **Webhook Notifications**: Alert on flag changes
6. **Flag Dependencies**: Support for dependent flags
7. **Gradual Rollout**: Time-based percentage increases