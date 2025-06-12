"""Comprehensive tests for SleepProcessor.

This test suite follows TDD principles and mirrors the patterns from existing
processor tests (cardio, activity, respiration) while achieving high coverage.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from clarity.ml.processors.sleep_processor import SleepProcessor
from clarity.models.health_data import (
    BiometricData,
    HealthMetric,
    HealthMetricType,
    SleepData,
    SleepStage,
)


class TestSleepProcessor:
    """Test suite for SleepProcessor."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.processor = SleepProcessor()

    @pytest.fixture
    @staticmethod
    def sample_sleep_data() -> SleepData:
        """Create sample sleep data for testing."""
        return SleepData(
            total_sleep_minutes=465,  # 7h 45min to match 8-hour window minus awake
            sleep_efficiency=0.97,  # 465/480
            time_to_sleep_minutes=15,
            wake_count=2,
            sleep_stages={
                SleepStage.AWAKE: 15,
                SleepStage.REM: 90,
                SleepStage.LIGHT: 285,
                SleepStage.DEEP: 90,
            },
            sleep_start=datetime(2024, 6, 1, 23, 0, tzinfo=UTC),
            sleep_end=datetime(2024, 6, 2, 7, 0, tzinfo=UTC),  # 8 hours total
        )

    @pytest.fixture
    @staticmethod
    def sample_sleep_metric(sample_sleep_data: SleepData) -> HealthMetric:
        """Create a sample sleep metric."""
        return HealthMetric(
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            sleep_data=sample_sleep_data,
            device_id="test_device",
            raw_data={"test": "data"},
            metadata={"test": "metadata"},
        )

    def test_processor_initialization(self) -> None:
        """Test processor initializes correctly."""
        assert isinstance(self.processor, SleepProcessor)
        # Processor doesn't have processor_name attribute, just check it exists
        assert hasattr(self.processor, "process")

    def test_process_single_night_complete_data(
        self, sample_sleep_metric: HealthMetric
    ) -> None:
        """Test processing single night with complete sleep data."""
        metrics = [sample_sleep_metric]

        result = self.processor.process(metrics)

        # Verify basic sleep metrics
        assert result.total_sleep_minutes == 465
        assert result.sleep_efficiency == 0.97
        assert result.sleep_latency == 15
        assert result.awakenings_count == 2

        # Verify sleep stage percentages
        assert result.rem_percentage == pytest.approx(90 / 465, abs=0.01)  # 90 min REM
        assert result.deep_percentage == pytest.approx(
            90 / 465, abs=0.01
        )  # 90 min Deep

        # Verify WASO calculation
        assert result.waso_minutes == 15  # From sleep stages

        # Verify consistency (single night should be 0)
        assert result.consistency_score == 0.0  # Single night has no consistency

        # Verify overall quality score is reasonable (max 6.0)
        assert 0.0 <= result.overall_quality_score <= 6.0

    def test_process_multiple_nights_consistency(self) -> None:
        """Test consistency calculation with multiple nights."""
        base_time = datetime(2024, 6, 1, 23, 0, tzinfo=UTC)

        metrics = []
        for _i, offset_hours in enumerate([0, 0.5, 1.0]):  # 23:00, 23:30, 00:00
            sleep_start = base_time + timedelta(hours=offset_hours)
            sleep_end = sleep_start + timedelta(hours=7.5)  # Consistent 7.5 hour window

            sleep_data = SleepData(
                total_sleep_minutes=420,  # 7 hours
                sleep_efficiency=0.88,
                time_to_sleep_minutes=10,
                wake_count=1,
                sleep_stages={
                    SleepStage.AWAKE: 20,
                    SleepStage.REM: 84,
                    SleepStage.LIGHT: 252,
                    SleepStage.DEEP: 84,
                },
                sleep_start=sleep_start,
                sleep_end=sleep_end,
            )

            metric = HealthMetric(
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                sleep_data=sleep_data,
                device_id="test_device",
                raw_data={"test": "data"},
                metadata={},
            )

            metrics.append(metric)

        result = self.processor.process(metrics)

        # Consistency should be very low due to 1-hour variation
        # Standard deviation of 30 minutes gives low score
        assert result.consistency_score < 0.5

    def test_process_no_sleep_stages(self) -> None:
        """Test processing when sleep stages are not available."""
        sleep_data = SleepData(
            total_sleep_minutes=465,  # Match 8 hour window - 15 min latency
            sleep_efficiency=0.97,  # 465/480
            time_to_sleep_minutes=12,
            wake_count=1,
            sleep_stages=None,  # No stage breakdown
            sleep_start=datetime(2024, 6, 1, 23, 30, tzinfo=UTC),
            sleep_end=datetime(2024, 6, 2, 7, 30, tzinfo=UTC),  # 8 hours total
        )

        metric = HealthMetric(
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            sleep_data=sleep_data,
            device_id="test_device",
            raw_data={"test": "data"},
            metadata={},
        )

        result = self.processor.process([metric])

        assert result.total_sleep_minutes == 465
        assert result.sleep_efficiency == 0.97
        assert result.rem_percentage == 0.0  # No stage data
        assert result.deep_percentage == 0.0  # No stage data
        # WASO calculated from time in bed calculation
        assert result.waso_minutes == pytest.approx(3.0, abs=1.0)  # 480 - 465 - 12

    def test_process_empty_metrics(self) -> None:
        """Test processing with no sleep metrics."""
        result = self.processor.process([])

        # Should return empty features
        assert result.total_sleep_minutes == 0.0
        assert result.sleep_efficiency == 0.0
        assert result.consistency_score == 0.0

    def test_process_invalid_metrics(self) -> None:
        """Test processing with metrics that have no sleep data."""
        metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,  # Not sleep data
            biometric_data=BiometricData(
                heart_rate=72.0,
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                oxygen_saturation=98.0,
                heart_rate_variability=45.0,
                respiratory_rate=16.0,
                body_temperature=37.0,  # 37°C instead of 98.6°F
                blood_glucose=90.0,
            ),
            sleep_data=None,
            device_id="test_device",
            raw_data={"test": "data"},
            metadata={},
        )

        result = self.processor.process([metric])

        # Should return empty features
        assert result.total_sleep_minutes == 0.0
        assert result.overall_quality_score == 0.0

    def test_get_summary_stats(self, sample_sleep_metric: HealthMetric) -> None:
        """Test summary statistics generation."""
        summary = self.processor.get_summary_stats([sample_sleep_metric])

        # Verify summary structure
        assert "overall_quality_rating" in summary
        assert "sleep_duration_hours" in summary
        assert "sleep_efficiency_rating" in summary
        assert "sleep_latency_rating" in summary
        assert "waso_rating" in summary
        assert "rem_sleep_rating" in summary
        assert "deep_sleep_rating" in summary
        assert "consistency_rating" in summary

        # Verify reasonable values
        assert summary["sleep_duration_hours"] == pytest.approx(7.75, abs=0.1)  # 465/60
        assert summary["sleep_efficiency_rating"] in {
            "excellent",
            "good",
            "fair",
            "poor",
        }
        assert summary["overall_quality_rating"] in {
            "excellent",
            "good",
            "fair",
            "poor",
        }
