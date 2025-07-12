# Mania Risk Module - Production Logging & Observability

## Overview

The mania risk module now includes production-ready logging and observability features to ensure reliable monitoring and debugging in production environments.

## Key Features Implemented

### 1. Structured Logging

The `ManiaRiskAnalyzer` now includes comprehensive structured logging:

- **Analysis Start**: Logs available data sources and user context
- **Component Analysis**: Each analysis component (sleep, circadian, activity, physiology) logs its results
- **Final Results**: Complete analysis results with scores, factors, and recommendations
- **Performance Metrics**: Analysis duration tracking

Example log output:
```json
{
  "timestamp": "2025-07-12T01:30:00Z",
  "level": "INFO",
  "logger": "clarity.ml.mania_risk_analyzer",
  "message": "Mania risk analysis completed",
  "user_id": "user...1234",
  "risk_score": 0.75,
  "alert_level": "high",
  "confidence": 0.85,
  "component_scores": {
    "sleep": 0.45,
    "circadian": 0.25,
    "activity": 0.05,
    "physiology": 0.0
  },
  "num_factors": 3,
  "top_factors": [
    "Critically low sleep: 3.0h (HealthKit)",
    "Disrupted circadian rhythm (score: 0.30)",
    "High activity fragmentation: 0.90"
  ],
  "analysis_duration_seconds": 0.023
}
```

### 2. Rate Limiting for High Alerts

To prevent alert fatigue, the module implements rate limiting for high-severity alerts:

- **24-hour window**: Only one high alert per user per 24 hours
- **Automatic downgrade**: Subsequent high-risk detections are downgraded to "moderate"
- **In-memory cache**: Simple implementation suitable for single-instance deployments
- **Production ready**: Can be extended to use Redis or DynamoDB for multi-instance deployments

### 3. PHI Privacy Protection

All logging follows HIPAA-compliant practices:

- **User ID sanitization**: User IDs are truncated (first 4 + last 4 characters)
- **No health values**: Specific health measurements are not logged
- **Aggregate scores only**: Only risk scores and categories are logged
- **Configurable redaction**: Easy to extend for additional privacy requirements

### 4. Edge Case Testing

Comprehensive boundary value tests ensure reliability:

```python
@pytest.mark.parametrize("test_case,expected", [
    # Sleep duration boundary tests
    ({"sleep_hours": 5.0}, {"has_factor": False}),  # Exactly at threshold
    ({"sleep_hours": 4.9}, {"has_factor": True}),   # Just below threshold
    # Circadian rhythm boundary tests
    ({"circadian": 0.5}, {"has_circadian_factor": False}),  # At threshold
    ({"circadian": 0.49}, {"has_circadian_factor": True}),  # Below threshold
    # Activity ratio boundary tests
    ({"activity_ratio": 1.5}, {"has_activity_factor": True}),   # At threshold
    ({"activity_ratio": 1.49}, {"has_activity_factor": False}), # Below threshold
])
```

### 5. Schema Regression Testing

API contract stability is ensured through schema tests:

```python
def test_actigraphy_analysis_schema_includes_mania_fields():
    """Ensure ActigraphyAnalysis schema includes required mania risk fields."""
    assert "mania_risk_score" in ActigraphyAnalysis.model_fields
    assert "mania_alert_level" in ActigraphyAnalysis.model_fields
```

## Configuration

### Logging Configuration

Configure logging using the provided YAML configuration:

```yaml
loggers:
  clarity.ml.mania_risk_analyzer:
    level: INFO
    handlers: [console, mania_risk]
    propagate: false
```

### Environment Variables

- `MANIA_RISK_LOG_LEVEL`: Set logging level (default: INFO)
- `MANIA_ALERT_RATE_LIMIT_HOURS`: Hours between high alerts (default: 24)
- `MANIA_LOG_PHI`: Enable/disable PHI logging (default: false)

## Monitoring & Alerting

### Key Metrics to Monitor

1. **High Alert Rate**: Track frequency of high-risk detections
2. **Analysis Duration**: Monitor performance degradation
3. **Data Coverage**: Track analyses with insufficient data
4. **Error Rate**: Monitor analysis failures

### Recommended Alerts

1. **High Alert Surge**: > 5 high alerts/hour across all users
2. **Performance Degradation**: Analysis duration > 1 second
3. **Data Quality**: > 20% analyses with low confidence
4. **Error Rate**: > 1% analysis failures

## Production Deployment Checklist

- [ ] Configure centralized logging (CloudWatch, Datadog, etc.)
- [ ] Set up alert rate limiting with persistent storage (Redis/DynamoDB)
- [ ] Configure monitoring dashboards
- [ ] Set up alerting rules
- [ ] Test log rotation and retention policies
- [ ] Verify PHI compliance in all environments
- [ ] Load test with expected production volume

## Future Enhancements

1. **Distributed Rate Limiting**: Use Redis for multi-instance deployments
2. **Advanced Analytics**: Track risk score trends over time
3. **ML Model Monitoring**: Track model drift and performance
4. **A/B Testing**: Support for testing threshold adjustments
5. **Audit Trail**: Comprehensive audit logging for compliance