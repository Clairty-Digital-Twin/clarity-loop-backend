# End-to-End Test Specification

Complete system validation ensuring the entire Clarity Loop Backend pipeline works seamlessly from HealthKit data ingestion to AI-powered insights.

## Test Overview

**Objective**: Validate the complete user journey with deterministic, reproducible results

**Flow**: `HealthKit Data → API Gateway → Pub/Sub → PAT ML Service → Gemini Insights → Firestore → User Notification`

**Success Criteria**: All tests pass with <5% variance in expected results

## Test Environment Setup

### Prerequisites

```bash
# Start full test environment
make test-env-up

# Verify all services are healthy
make health-check-all

# Load test fixtures
make load-test-fixtures
```

### Required Services

- **API Gateway**: `localhost:8000`
- **ML Service**: `localhost:8001`
- **Firestore Emulator**: `localhost:8080`
- **Pub/Sub Emulator**: `localhost:8085`
- **Redis Cache**: `localhost:6379`

## Test Fixtures

### 1. HealthKit Sample Data (24-hour cycle)

**File**: `tests/fixtures/healthkit_24h_sample.json`

```json
{
  "user_id": "test-user-001",
  "device_info": {
    "model": "Apple Watch Series 9",
    "os_version": "watchOS 10.1",
    "app_version": "1.0.0"
  },
  "collection_period": {
    "start": "2024-01-15T00:00:00Z",
    "end": "2024-01-15T23:59:59Z",
    "timezone": "America/New_York"
  },
  "actigraphy_data": {
    "sampling_rate": "1_minute",
    "total_samples": 1440,
    "data": [
      {"timestamp": "2024-01-15T00:00:00Z", "activity_count": 0, "heart_rate": 58},
      {"timestamp": "2024-01-15T00:01:00Z", "activity_count": 0, "heart_rate": 57},
      // ... 1438 more samples (sleep period: low activity)
      {"timestamp": "2024-01-15T07:30:00Z", "activity_count": 45, "heart_rate": 72},
      {"timestamp": "2024-01-15T07:31:00Z", "activity_count": 52, "heart_rate": 78},
      // ... wake-up and morning activity
      {"timestamp": "2024-01-15T12:00:00Z", "activity_count": 120, "heart_rate": 95},
      // ... active daytime period
      {"timestamp": "2024-01-15T22:30:00Z", "activity_count": 15, "heart_rate": 65},
      {"timestamp": "2024-01-15T23:59:00Z", "activity_count": 2, "heart_rate": 60}
    ]
  },
  "sleep_data": {
    "sleep_periods": [
      {
        "start": "2024-01-15T23:15:00Z",
        "end": "2024-01-16T07:30:00Z",
        "sleep_stage_transitions": [
          {"timestamp": "2024-01-15T23:15:00Z", "stage": "awake"},
          {"timestamp": "2024-01-15T23:35:00Z", "stage": "light"},
          {"timestamp": "2024-01-16T00:15:00Z", "stage": "deep"},
          {"timestamp": "2024-01-16T02:45:00Z", "stage": "rem"},
          {"timestamp": "2024-01-16T07:15:00Z", "stage": "light"},
          {"timestamp": "2024-01-16T07:30:00Z", "stage": "awake"}
        ]
      }
    ]
  },
  "heart_rate_variability": {
    "rmssd_values": [35.2, 38.1, 42.3, 39.7, 33.8],
    "recording_times": ["07:30", "12:00", "15:30", "19:00", "22:30"]
  }
}
```

### 2. Expected PAT Feature Vector

**File**: `tests/fixtures/expected_pat_features.json`

```json
{
  "user_id": "test-user-001",
  "analysis_timestamp": "2024-01-16T08:00:00Z",
  "model_version": "PAT-small-v1.0",
  "processing_time_ms": 450,
  "features": {
    "sleep_metrics": {
      "sleep_efficiency": 0.847,
      "sleep_onset_latency_minutes": 20,
      "wake_after_sleep_onset_minutes": 45,
      "total_sleep_time_hours": 8.25,
      "sleep_fragmentation_index": 0.23
    },
    "circadian_metrics": {
      "rhythm_strength": 0.72,
      "acrophase_hour": 14.5,
      "amplitude": 0.68,
      "interdaily_stability": 0.65,
      "intradaily_variability": 0.34
    },
    "activity_metrics": {
      "daily_activity_count": 145680,
      "peak_activity_time": "14:30",
      "activity_fragmentation": 0.28,
      "sedentary_percentage": 0.65,
      "active_percentage": 0.35
    },
    "heart_rate_metrics": {
      "resting_hr": 58,
      "max_hr": 145,
      "hr_variability_rmssd": 37.8,
      "recovery_hr_slope": -0.42
    },
    "wellness_indicators": {
      "sleep_quality_score": 0.78,
      "recovery_score": 0.72,
      "stress_indicator": 0.31,
      "energy_level_prediction": 0.75
    }
  },
  "confidence_scores": {
    "sleep_detection": 0.94,
    "circadian_phase": 0.88,
    "activity_classification": 0.91,
    "overall_analysis": 0.91
  }
}
```

### 3. Gemini Prompt Template

**File**: `tests/fixtures/gemini_prompt_template.txt`

```
You are a health insights AI assistant analyzing actigraphy data for personalized wellness recommendations.

USER CONTEXT:
- User ID: {{user_id}}
- Analysis Date: {{analysis_date}}
- Previous insights available: {{has_history}}

ACTIGRAPHY ANALYSIS:
Sleep Quality: {{sleep_efficiency}} ({{sleep_quality_interpretation}})
Circadian Rhythm: {{rhythm_strength}} ({{rhythm_interpretation}})
Activity Pattern: {{activity_summary}}
Recovery Score: {{recovery_score}}

GENERATE INSIGHTS:
1. Sleep Quality Assessment (2-3 sentences)
2. Activity Recommendations (2-3 actionable items)
3. Circadian Health Tips (1-2 specific suggestions)
4. Tomorrow's Focus (1 priority area)

TONE: Supportive, scientific, actionable
LENGTH: 150-200 words total
FORMAT: Natural language, avoid medical diagnosis
```

### 4. Expected Gemini Response

**File**: `tests/fixtures/expected_gemini_response.json`

```json
{
  "user_id": "test-user-001",
  "insight_id": "insight-20240116-080000",
  "generated_at": "2024-01-16T08:00:15Z",
  "model_used": "gemini-2.0-flash-exp",
  "prompt_version": "v1.2",
  "insights": {
    "sleep_assessment": "Your sleep efficiency of 84.7% shows good overall sleep quality, though the 20-minute sleep onset suggests some difficulty falling asleep. Your deep sleep phases were well-distributed throughout the night, supporting physical recovery.",
    "activity_recommendations": [
      "Consider a 10-minute wind-down routine starting at 11 PM to improve sleep onset time",
      "Your peak activity around 2:30 PM aligns well with your natural energy - plan important tasks during this window",
      "Try incorporating 2-3 short movement breaks during your sedentary periods to boost circulation"
    ],
    "circadian_tips": [
      "Your circadian rhythm strength of 72% is solid - maintain consistent sleep/wake times to strengthen it further",
      "Consider 15 minutes of morning sunlight exposure to reinforce your natural rhythm"
    ],
    "tomorrow_focus": "Focus on establishing a consistent evening routine to reduce sleep onset time and improve overall sleep efficiency."
  },
  "wellness_scores": {
    "sleep_quality": 7.8,
    "recovery_readiness": 7.2,
    "energy_prediction": 7.5,
    "stress_level": 3.1
  },
  "confidence_score": 0.89,
  "token_usage": {
    "prompt_tokens": 445,
    "completion_tokens": 189,
    "total_tokens": 634
  }
}
```

## E2E Test Scenarios

### Scenario 1: Complete Happy Path

**Test ID**: `E2E-001`
**Description**: Full pipeline from HealthKit upload to insight delivery

#### Test Steps

```bash
# 1. Upload HealthKit data
curl -X POST http://localhost:8000/api/v1/healthkit \
  -H "Authorization: Bearer ${TEST_TOKEN}" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/healthkit_24h_sample.json

# Expected Response: 201 Created
{
  "upload_id": "upload-12345",
  "status": "processing",
  "estimated_completion": "2024-01-16T08:02:00Z"
}
```

```bash
# 2. Verify Pub/Sub message published
curl http://localhost:8085/v1/projects/test-project/topics/healthkit-ingestion:publish

# Expected: Message found in topic
```

```bash
# 3. Check ML processing status
curl http://localhost:8001/api/v1/processing/status/upload-12345

# Expected Response: 200 OK
{
  "upload_id": "upload-12345",
  "status": "completed",
  "processing_time_ms": 450,
  "confidence_score": 0.91
}
```

```bash
# 4. Verify PAT features generated
curl http://localhost:8000/api/v1/users/test-user-001/features/latest

# Expected: Features match tests/fixtures/expected_pat_features.json
# Tolerance: ±5% for numerical values
```

```bash
# 5. Check Gemini insight generation
curl http://localhost:8000/api/v1/users/test-user-001/insights/latest

# Expected: Response structure matches expected_gemini_response.json
# Content validation: Key phrases present, appropriate tone
```

```bash
# 6. Verify Firestore document created
curl http://localhost:8080/v1/projects/test-project/databases/(default)/documents/users/test-user-001

# Expected: Complete user document with latest analysis
```

#### Success Criteria

- ✅ **Upload Success**: 201 status, valid upload_id returned
- ✅ **Processing Speed**: <500ms for PAT analysis
- ✅ **Feature Accuracy**: ±5% variance from expected values
- ✅ **Insight Quality**: Contains required sections, appropriate length
- ✅ **Data Persistence**: All data correctly stored in Firestore
- ✅ **End-to-End Time**: <30 seconds total pipeline completion

### Scenario 2: Error Handling & Recovery

**Test ID**: `E2E-002`
**Description**: Validate graceful error handling and system recovery

#### Test Cases

```bash
# 2a. Invalid HealthKit data
curl -X POST http://localhost:8000/api/v1/healthkit \
  -H "Authorization: Bearer ${TEST_TOKEN}" \
  -d '{"invalid": "data"}'

# Expected: 400 Bad Request with detailed validation errors
```

```bash
# 2b. ML service temporarily unavailable
# Stop ML service: docker-compose stop ml-service
curl -X POST http://localhost:8000/api/v1/healthkit \
  -d @tests/fixtures/healthkit_24h_sample.json

# Expected: 202 Accepted, queued for retry
# Restart ML service, verify eventual processing
```

```bash
# 2c. Gemini API rate limit
# Simulate rate limiting in test environment
# Expected: Exponential backoff, eventual success
```

### Scenario 3: Performance & Scale

**Test ID**: `E2E-003`
**Description**: Validate system performance under load

```bash
# 3a. Concurrent uploads
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/v1/healthkit \
    -H "Authorization: Bearer test-user-${i}" \
    -d @tests/fixtures/healthkit_24h_sample.json &
done
wait

# Expected: All uploads process successfully within 60 seconds
```

```bash
# 3b. Large data upload (7 days of data)
curl -X POST http://localhost:8000/api/v1/healthkit \
  -d @tests/fixtures/healthkit_7d_sample.json

# Expected: Successful processing within 2 minutes
```

### Scenario 4: Security & Authentication

**Test ID**: `E2E-004`
**Description**: Validate security controls and authentication

```bash
# 4a. No authentication token
curl -X POST http://localhost:8000/api/v1/healthkit \
  -d @tests/fixtures/healthkit_24h_sample.json

# Expected: 401 Unauthorized
```

```bash
# 4b. Invalid token
curl -X POST http://localhost:8000/api/v1/healthkit \
  -H "Authorization: Bearer invalid-token" \
  -d @tests/fixtures/healthkit_24h_sample.json

# Expected: 401 Unauthorized
```

```bash
# 4c. Cross-user data access attempt
curl http://localhost:8000/api/v1/users/different-user/insights \
  -H "Authorization: Bearer ${TEST_TOKEN}"

# Expected: 403 Forbidden
```

## Automated Test Execution

### Test Runner Script

**File**: `tests/run_e2e_tests.sh`

```bash
#!/bin/bash
set -euo pipefail

echo "🧪 Starting E2E Test Suite"

# Setup test environment
echo "📋 Setting up test environment..."
make test-env-up
make load-test-fixtures

# Wait for services to be ready
echo "⏳ Waiting for services..."
./scripts/wait-for-services.sh

# Run test scenarios
echo "🎯 Running E2E scenarios..."

declare -A test_results

# Scenario 1: Happy Path
echo "Running E2E-001: Happy Path"
if ./tests/scenarios/happy_path.sh; then
  test_results["E2E-001"]="PASS"
else
  test_results["E2E-001"]="FAIL"
fi

# Scenario 2: Error Handling
echo "Running E2E-002: Error Handling"
if ./tests/scenarios/error_handling.sh; then
  test_results["E2E-002"]="PASS"
else
  test_results["E2E-002"]="FAIL"
fi

# Scenario 3: Performance
echo "Running E2E-003: Performance"
if ./tests/scenarios/performance.sh; then
  test_results["E2E-003"]="PASS"
else
  test_results["E2E-003"]="FAIL"
fi

# Scenario 4: Security
echo "Running E2E-004: Security"
if ./tests/scenarios/security.sh; then
  test_results["E2E-004"]="PASS"
else
  test_results["E2E-004"]="FAIL"
fi

# Generate test report
echo "📊 Generating test report..."
./tests/generate_report.sh "${test_results[@]}"

# Cleanup
echo "🧹 Cleaning up test environment..."
make test-env-down

echo "✅ E2E Test Suite Complete"
```

### Continuous Validation

```yaml
# .github/workflows/e2e-tests.yml
name: E2E Tests
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run E2E Test Suite
        run: |
          make test-env-setup
          ./tests/run_e2e_tests.sh
      - name: Upload test results
        uses: actions/upload-artifact@v4
        with:
          name: e2e-test-results
          path: tests/results/
```

## Test Data Management

### Fixture Generation

```python
# tests/fixtures/generate_fixtures.py
import json
from datetime import datetime, timedelta
import numpy as np

def generate_realistic_actigraphy(duration_hours=24):
    """Generate realistic actigraphy data with circadian patterns"""
    samples = []
    base_time = datetime(2024, 1, 15, 0, 0, 0)

    for minute in range(duration_hours * 60):
        timestamp = base_time + timedelta(minutes=minute)
        hour = timestamp.hour + timestamp.minute / 60

        # Circadian activity pattern
        if 0 <= hour < 7:  # Sleep period
            activity = np.random.poisson(2)
            heart_rate = 55 + np.random.normal(0, 3)
        elif 7 <= hour < 9:  # Wake up
            activity = np.random.poisson(40)
            heart_rate = 70 + np.random.normal(0, 5)
        elif 9 <= hour < 12:  # Morning active
            activity = np.random.poisson(80)
            heart_rate = 85 + np.random.normal(0, 8)
        elif 12 <= hour < 14:  # Peak activity
            activity = np.random.poisson(120)
            heart_rate = 95 + np.random.normal(0, 10)
        elif 14 <= hour < 18:  # Afternoon
            activity = np.random.poisson(70)
            heart_rate = 80 + np.random.normal(0, 7)
        elif 18 <= hour < 22:  # Evening
            activity = np.random.poisson(50)
            heart_rate = 75 + np.random.normal(0, 6)
        else:  # Pre-sleep
            activity = np.random.poisson(20)
            heart_rate = 65 + np.random.normal(0, 4)

        samples.append({
            "timestamp": timestamp.isoformat() + "Z",
            "activity_count": max(0, int(activity)),
            "heart_rate": max(45, min(180, int(heart_rate)))
        })

    return samples
```

### Validation Helpers

```python
# tests/validators/response_validator.py
def validate_pat_features(actual, expected, tolerance=0.05):
    """Validate PAT feature extraction results"""
    errors = []

    for category, metrics in expected["features"].items():
        if category not in actual["features"]:
            errors.append(f"Missing feature category: {category}")
            continue

        for metric, expected_value in metrics.items():
            actual_value = actual["features"][category].get(metric)

            if actual_value is None:
                errors.append(f"Missing metric: {category}.{metric}")
                continue

            if isinstance(expected_value, (int, float)):
                diff = abs(actual_value - expected_value) / expected_value
                if diff > tolerance:
                    errors.append(
                        f"Metric {category}.{metric}: "
                        f"expected {expected_value}, "
                        f"got {actual_value} "
                        f"(diff: {diff:.2%})"
                    )

    return errors

def validate_gemini_insights(response, expected_structure):
    """Validate Gemini-generated insights"""
    errors = []

    required_sections = [
        "sleep_assessment",
        "activity_recommendations",
        "circadian_tips",
        "tomorrow_focus"
    ]

    for section in required_sections:
        if section not in response.get("insights", {}):
            errors.append(f"Missing insight section: {section}")

    # Validate content quality
    insights = response.get("insights", {})
    total_length = sum(len(str(v)) for v in insights.values())

    if total_length < 100:
        errors.append("Insights too short")
    elif total_length > 300:
        errors.append("Insights too long")

    return errors
```

## Performance Benchmarks

### Expected Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| HealthKit Upload Response | <200ms | API Gateway response time |
| PAT Feature Extraction | <500ms | ML Service processing time |
| Gemini Insight Generation | <2s | LLM API response time |
| End-to-End Pipeline | <30s | Upload to Firestore completion |
| Concurrent Users | 100+ | Without degradation |
| Data Throughput | 1MB/req | HealthKit JSON payload |

### Load Testing

```bash
# Load test with Apache Bench
ab -n 100 -c 10 -H "Authorization: Bearer ${TEST_TOKEN}" \
   -p tests/fixtures/healthkit_24h_sample.json \
   -T application/json \
   http://localhost:8000/api/v1/healthkit

# Expected: 95th percentile < 500ms
```

---

**Goal**: Deterministic, automated validation of complete system functionality
**Coverage**: Happy path, error cases, performance, security
**Automation**: CI/CD integrated, continuous monitoring
**Reliability**: <1% false positive rate, reproducible results
