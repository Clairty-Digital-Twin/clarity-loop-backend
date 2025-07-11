"""Unit tests for health_data.py model validators.

These tests focus on the pure validation logic in the models,
providing quick coverage gains without requiring API infrastructure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import ValidationError
import pytest

from clarity.core.config import get_settings
from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthDataUpload,
    HealthMetric,
    HealthMetricType,
    MentalHealthIndicator,
    SleepData,
)


class TestSleepDataValidator:
    """Test SleepData.validate_sleep_times validator."""

    def test_valid_sleep_times(self):
        """Test valid sleep start and end times."""
        start = datetime.now(UTC)
        end = start + timedelta(hours=8)

        sleep = SleepData(
            total_sleep_minutes=480,  # 8 hours
            sleep_efficiency=0.85,
            awakenings=2,
            sleep_start=start,
            sleep_end=end,
        )

        assert sleep.sleep_start == start
        assert sleep.sleep_end == end
        assert sleep.total_sleep_minutes == 480

    def test_sleep_end_before_start(self):
        """Test validation fails when sleep end is before start."""
        start = datetime.now(UTC)
        end = start - timedelta(hours=1)  # End before start

        with pytest.raises(
            ValidationError, match="Sleep end time must be after start time"
        ):
            SleepData(
                total_sleep_minutes=60,
                sleep_efficiency=0.85,
                awakenings=2,
                sleep_start=start,
                sleep_end=end,
            )

    def test_sleep_end_equals_start(self):
        """Test validation fails when sleep end equals start."""
        time = datetime.now(UTC)

        with pytest.raises(
            ValidationError, match="Sleep end time must be after start time"
        ):
            SleepData(
                total_sleep_minutes=0,
                sleep_efficiency=0.85,
                awakenings=2,
                sleep_start=time,
                sleep_end=time,
            )

    def test_inconsistent_total_minutes(self):
        """Test validation fails when total_sleep_minutes doesn't match duration."""
        start = datetime.now(UTC)
        end = start + timedelta(hours=8)  # 480 minutes

        with pytest.raises(
            ValidationError,
            match="Total sleep minutes inconsistent with start/end times",
        ):
            SleepData(
                total_sleep_minutes=120,  # Only 2 hours, but duration is 8 hours
                sleep_efficiency=0.85,
                awakenings=2,
                sleep_start=start,
                sleep_end=end,
            )

    def test_sleep_times_required(self):
        """Test that sleep_start and sleep_end are required fields."""
        # Both fields are required
        with pytest.raises(ValidationError) as exc_info:
            SleepData(
                total_sleep_minutes=480,
                sleep_efficiency=0.85,
                awakenings=2,
            )

        errors = exc_info.value.errors()
        assert len(errors) == 2
        assert any(e["loc"] == ("sleep_start",) for e in errors)
        assert any(e["loc"] == ("sleep_end",) for e in errors)

    @pytest.mark.parametrize(
        "hours,minutes",
        [
            (7, 420),  # 7 hours
            (8, 480),  # 8 hours
            (9, 540),  # 9 hours
            (10, 600),  # 10 hours
        ],
    )
    def test_various_sleep_durations(self, hours: int, minutes: int):
        """Test various valid sleep durations."""
        start = datetime.now(UTC)
        end = start + timedelta(hours=hours)

        sleep = SleepData(
            total_sleep_minutes=minutes,
            sleep_efficiency=0.85,
            awakenings=2,
            sleep_start=start,
            sleep_end=end,
        )

        assert sleep.total_sleep_minutes == minutes


class TestHealthMetricValidator:
    """Test HealthMetric.validate_metric_data validator."""

    def test_heart_rate_requires_biometric_data(self):
        """Test heart rate metric requires biometric data."""
        # Valid case
        metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=72.0),
        )
        assert metric.biometric_data.heart_rate == 72.0

        # Invalid case - missing biometric data
        with pytest.raises(ValidationError, match="requires biometric_data"):
            HealthMetric(metric_type=HealthMetricType.HEART_RATE)

    def test_sleep_analysis_requires_sleep_data(self):
        """Test sleep analysis metric requires sleep data."""
        # Valid case
        sleep_start = datetime.now(UTC)
        sleep_data = SleepData(
            total_sleep_minutes=480,
            sleep_efficiency=0.85,
            awakenings=2,
            sleep_start=sleep_start,
            sleep_end=sleep_start + timedelta(hours=8),
        )
        metric = HealthMetric(
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            sleep_data=sleep_data,
        )
        assert metric.sleep_data == sleep_data

        # Invalid case - missing sleep data
        with pytest.raises(ValidationError, match="requires sleep_data"):
            HealthMetric(metric_type=HealthMetricType.SLEEP_ANALYSIS)

    def test_activity_level_requires_activity_data(self):
        """Test activity level metric requires activity data."""
        # Valid case
        activity = ActivityData(steps=10000, distance=8.5)
        metric = HealthMetric(
            metric_type=HealthMetricType.ACTIVITY_LEVEL,
            activity_data=activity,
        )
        assert metric.activity_data.steps == 10000

        # Invalid case - missing activity data
        with pytest.raises(ValidationError, match="requires activity_data"):
            HealthMetric(metric_type=HealthMetricType.ACTIVITY_LEVEL)

    def test_mood_assessment_requires_mental_health_data(self):
        """Test mood assessment metric requires mental health data."""
        # Valid case
        mental_health = MentalHealthIndicator(
            stress_level=3.5,
            energy_level=7.0,
        )
        metric = HealthMetric(
            metric_type=HealthMetricType.MOOD_ASSESSMENT,
            mental_health_data=mental_health,
        )
        assert metric.mental_health_data.stress_level == 3.5

        # Invalid case - missing mental health data
        with pytest.raises(ValidationError, match="requires mental_health_data"):
            HealthMetric(metric_type=HealthMetricType.MOOD_ASSESSMENT)

    @pytest.mark.parametrize(
        "metric_type,required_field",
        [
            (HealthMetricType.HEART_RATE_VARIABILITY, "biometric_data"),
            (HealthMetricType.BLOOD_PRESSURE, "biometric_data"),
            (HealthMetricType.STRESS_INDICATORS, "mental_health_data"),
            (HealthMetricType.COGNITIVE_METRICS, "mental_health_data"),
        ],
    )
    def test_all_metric_types_validation(
        self, metric_type: HealthMetricType, required_field: str
    ):
        """Test all metric types require appropriate data fields."""
        with pytest.raises(ValidationError, match=f"requires {required_field}"):
            HealthMetric(metric_type=metric_type)

    def test_metric_type_required(self):
        """Test that metric_type is a required field."""
        # metric_type is required and must be a valid enum value
        with pytest.raises(ValidationError) as exc_info:
            HealthMetric.model_validate({"metric_type": None})

        errors = exc_info.value.errors()
        assert any("metric_type" in str(e) for e in errors)

        # Also test with missing metric_type
        with pytest.raises(ValidationError) as exc_info:
            HealthMetric.model_validate({})

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("metric_type",) for e in errors)

    def test_metric_with_raw_data_only(self):
        """Test metric can have raw_data without specific data type."""
        # This tests metrics that might not have a specific data requirement
        raw_data = {"custom_field": "value", "reading": 123.45}
        metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=72.0),
            raw_data=raw_data,
        )
        assert metric.raw_data == raw_data


class TestHealthDataUploadValidator:
    """Test HealthDataUpload.validate_metrics_consistency validator."""

    def test_valid_metrics_upload(self):
        """Test valid health data upload with multiple metrics."""
        metrics = [
            HealthMetric(
                metric_type=HealthMetricType.HEART_RATE,
                biometric_data=BiometricData(heart_rate=72.0),
            ),
            HealthMetric(
                metric_type=HealthMetricType.ACTIVITY_LEVEL,
                activity_data=ActivityData(steps=5000),
            ),
        ]

        upload = HealthDataUpload(
            user_id=uuid4(),
            metrics=metrics,
            upload_source="test_suite",
            client_timestamp=datetime.now(UTC),
        )

        assert len(upload.metrics) == 2
        assert upload.upload_source == "test_suite"

    def test_exceeds_max_metrics_limit(self):
        """Test validation fails when exceeding MAX_METRICS_PER_UPLOAD."""
        # Create more metrics than allowed
        settings = get_settings()
        max_metrics = settings.max_metrics_per_upload
        metrics = []
        for i in range(max_metrics + 1):
            metrics.append(
                HealthMetric(
                    metric_type=HealthMetricType.HEART_RATE,
                    biometric_data=BiometricData(heart_rate=70.0 + i),
                )
            )

        with pytest.raises(ValidationError) as exc_info:
            HealthDataUpload(
                user_id=uuid4(),
                metrics=metrics,
                upload_source="test_suite",
                client_timestamp=datetime.now(UTC),
            )

    def test_duplicate_metric_ids(self):
        """Test validation fails with duplicate metric IDs."""
        # Create metrics with duplicate IDs
        duplicate_id = uuid4()
        metrics = [
            HealthMetric(
                metric_id=duplicate_id,
                metric_type=HealthMetricType.HEART_RATE,
                biometric_data=BiometricData(heart_rate=72.0),
            ),
            HealthMetric(
                metric_id=duplicate_id,  # Same ID
                metric_type=HealthMetricType.ACTIVITY_LEVEL,
                activity_data=ActivityData(steps=5000),
            ),
        ]

        with pytest.raises(ValidationError, match="Duplicate metric IDs not allowed"):
            HealthDataUpload(
                user_id=uuid4(),
                metrics=metrics,
                upload_source="test_suite",
                client_timestamp=datetime.now(UTC),
            )

    def test_empty_metrics_list(self):
        """Test validation fails with empty metrics list."""
        with pytest.raises(ValidationError, match="at least 1 item"):
            HealthDataUpload(
                user_id=uuid4(),
                metrics=[],  # Empty list
                upload_source="test_suite",
                client_timestamp=datetime.now(UTC),
            )

    def test_exactly_max_metrics(self):
        """Test validation passes with exactly MAX_METRICS_PER_UPLOAD metrics."""
        settings = get_settings()
        max_metrics = settings.max_metrics_per_upload
        metrics = []
        for i in range(max_metrics):
            metrics.append(
                HealthMetric(
                    metric_type=HealthMetricType.HEART_RATE,
                    biometric_data=BiometricData(heart_rate=70.0 + i),
                )
            )

        # Should not raise any errors
        upload = HealthDataUpload(
            user_id=uuid4(),
            metrics=metrics,
            upload_source="test_suite",
            client_timestamp=datetime.now(UTC),
        )

        assert len(upload.metrics) == max_metrics

    def test_unique_metric_ids_generated(self):
        """Test that unique metric IDs are generated by default."""
        metrics = [
            HealthMetric(
                metric_type=HealthMetricType.HEART_RATE,
                biometric_data=BiometricData(heart_rate=72.0),
            ),
            HealthMetric(
                metric_type=HealthMetricType.HEART_RATE,
                biometric_data=BiometricData(heart_rate=73.0),
            ),
            HealthMetric(
                metric_type=HealthMetricType.HEART_RATE,
                biometric_data=BiometricData(heart_rate=74.0),
            ),
        ]

        upload = HealthDataUpload(
            user_id=uuid4(),
            metrics=metrics,
            upload_source="test_suite",
            client_timestamp=datetime.now(UTC),
        )

        # Check all metric IDs are unique
        metric_ids = [m.metric_id for m in upload.metrics]
        assert len(metric_ids) == len(set(metric_ids))

    @pytest.mark.parametrize(
        "source",
        [
            "apple_health",
            "fitbit",
            "garmin",
            "samsung_health",
            "manual_entry",
            "test_suite",
        ],
    )
    def test_various_upload_sources(self, source: str):
        """Test various valid upload sources."""
        metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=72.0),
        )

        upload = HealthDataUpload(
            user_id=uuid4(),
            metrics=[metric],
            upload_source=source,
            client_timestamp=datetime.now(UTC),
        )

        assert upload.upload_source == source


class TestFieldValidationBoundaries:
    """Test field-level validation boundaries."""

    def test_sleep_efficiency_bounds(self):
        """Test sleep efficiency validation boundaries."""
        # Valid range: 0.0 to 1.0
        sleep_start = datetime.now(UTC)
        valid = SleepData(
            total_sleep_minutes=480,
            sleep_efficiency=0.85,
            awakenings=2,
            sleep_start=sleep_start,
            sleep_end=sleep_start + timedelta(hours=8),
        )
        assert valid.sleep_efficiency == 0.85

        # Test boundaries
        start = datetime.now(UTC)
        SleepData(
            total_sleep_minutes=480,
            sleep_efficiency=0.0,
            awakenings=0,
            sleep_start=start,
            sleep_end=start + timedelta(hours=8),
        )
        SleepData(
            total_sleep_minutes=480,
            sleep_efficiency=1.0,
            awakenings=0,
            sleep_start=start,
            sleep_end=start + timedelta(hours=8),
        )

        # Test out of bounds
        with pytest.raises(ValidationError):
            SleepData(
                total_sleep_minutes=480,
                sleep_efficiency=-0.1,
                awakenings=0,
                sleep_start=start,
                sleep_end=start + timedelta(hours=8),
            )

        with pytest.raises(ValidationError):
            SleepData(
                total_sleep_minutes=480,
                sleep_efficiency=1.1,
                awakenings=0,
                sleep_start=start,
                sleep_end=start + timedelta(hours=8),
            )

    def test_heart_rate_bounds(self):
        """Test heart rate validation boundaries in the model."""
        # Valid range: 30 to 300 BPM
        BiometricData(heart_rate=30.0)  # Minimum valid
        BiometricData(heart_rate=300.0)  # Maximum valid
        BiometricData(heart_rate=72.0)  # Normal
        BiometricData(heart_rate=180.0)  # Exercise

        # Test out of bounds - too low
        with pytest.raises(ValidationError) as exc_info:
            BiometricData(heart_rate=29.9)
        assert "greater than or equal to 30" in str(exc_info.value)

        # Test out of bounds - too high
        with pytest.raises(ValidationError) as exc_info:
            BiometricData(heart_rate=300.1)
        assert "less than or equal to 300" in str(exc_info.value)

        # Test type validation
        with pytest.raises(ValidationError):
            BiometricData(heart_rate="not a number")

    def test_activity_steps_bounds(self):
        """Test activity steps validation boundaries."""
        # Valid range: 0 to 100,000
        ActivityData(steps=0)  # Minimum
        ActivityData(steps=100000)  # Maximum

        # Test out of bounds
        with pytest.raises(ValidationError):
            ActivityData(steps=-1)  # Negative

        with pytest.raises(ValidationError):
            ActivityData(steps=100001)  # Too high

    def test_stress_level_bounds(self):
        """Test stress level validation boundaries."""
        # Valid range: 1.0 to 10.0
        MentalHealthIndicator(stress_level=1.0)  # Minimum
        MentalHealthIndicator(stress_level=10.0)  # Maximum

        # Test out of bounds
        with pytest.raises(ValidationError):
            MentalHealthIndicator(stress_level=0.9)  # Too low

        with pytest.raises(ValidationError):
            MentalHealthIndicator(stress_level=10.1)  # Too high
