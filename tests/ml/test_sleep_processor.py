"""Comprehensive tests for SleepProcessor.

This test suite follows TDD principles and mirrors the patterns from existing
processor tests (cardio, activity, respiration) while achieving high coverage.
"""

from datetime import UTC, datetime, timedelta

import pytest

from clarity.ml.processors.sleep_processor import SleepProcessor
from clarity.models.health_data import (
    HealthMetric,
    HealthMetricType,
    SleepData,
    SleepStage,
)


class TestSleepProcessor:
    """Test suite for SleepProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = SleepProcessor()

    @pytest.fixture
    def sample_sleep_data(self):
        """Create sample sleep data for testing."""
        return SleepData(
            total_sleep_minutes=465,  # 7h 45min to match 8-hour window minus awake time
            sleep_efficiency=0.875,   # 87.5%
            time_to_sleep_minutes=15,
            wake_count=2,
            sleep_stages={
                SleepStage.AWAKE: 15,   # Reduced to match total
                SleepStage.REM: 90,
                SleepStage.LIGHT: 240,
                SleepStage.DEEP: 135    # Adjusted to sum correctly
            },
            sleep_start=datetime(2024, 6, 1, 23, 0, tzinfo=UTC),
            sleep_end=datetime(2024, 6, 2, 7, 0, tzinfo=UTC)  # 8 hours total
        )

    @pytest.fixture
    def sample_sleep_metric(self, sample_sleep_data):
        """Create a sample sleep metric."""
        return HealthMetric(
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            sleep_data=sample_sleep_data,
            device_id="test_device",
            raw_data={"test": "data"},
            metadata={"test": "metadata"}
        )

    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        assert isinstance(self.processor, SleepProcessor)
        # Processor doesn't have processor_name attribute, just check it exists
        assert hasattr(self.processor, 'process')

    def test_process_single_night_complete_data(self, sample_sleep_metric):
        """Test processing single night with complete sleep data."""
        metrics = [sample_sleep_metric]

        result = self.processor.process(metrics)

        assert result.total_sleep_minutes == 465
        assert result.sleep_efficiency == 0.875
        assert result.sleep_latency == 15.0
        assert result.waso_minutes == 15.0  # Awake minutes after sleep onset
        assert result.awakenings_count == 2
        assert result.rem_percentage == pytest.approx(0.194, abs=0.001)  # 90/465
        assert result.deep_percentage == pytest.approx(0.290, abs=0.001)  # 135/465
        assert result.consistency_score == 0.0  # Single night has no consistency

    def test_process_multiple_nights_consistency_calculation(self):
        """Test consistency score calculation with multiple nights."""
        # Create 3 nights with varying sleep start times
        metrics = []
        base_time = datetime(2024, 6, 1, 23, 0, tzinfo=UTC)

        for i, offset_hours in enumerate([0, 0.5, 1.0]):  # 23:00, 23:30, 00:00
            sleep_start = base_time + timedelta(hours=offset_hours)
            sleep_end = sleep_start + timedelta(hours=7.5)  # Consistent 7.5 hour window

            sleep_data = SleepData(
                total_sleep_minutes=420,  # 7 hours actual sleep
                sleep_efficiency=0.93,    # 420/450 = 0.93
                time_to_sleep_minutes=10,
                wake_count=1,
                sleep_stages={
                    SleepStage.AWAKE: 20,   # 450 - 420 - 10 = 20
                    SleepStage.REM: 90,
                    SleepStage.LIGHT: 240,
                    SleepStage.DEEP: 90
                },
                sleep_start=sleep_start,
                sleep_end=sleep_end
            )

            metric = HealthMetric(
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                sleep_data=sleep_data,
                device_id="test_device",
                raw_data={"test": "data"},
                metadata={"test": "metadata"}
            )
            metrics.append(metric)

        result = self.processor.process(metrics)

        # Consistency should be very low due to 1-hour variation
        # Standard deviation of 30 minutes gives low score
        assert result.consistency_score < 0.5

    def test_process_excellent_consistency(self):
        """Test excellent consistency score with regular sleep times."""
        metrics = []
        base_time = datetime(2024, 6, 1, 23, 0, tzinfo=UTC)

        # Create 5 nights with very consistent sleep times (within 10 minutes)
        for i in range(5):
            sleep_start = base_time + timedelta(days=i, minutes=i * 5)  # 0-20 min variation
            sleep_end = sleep_start + timedelta(hours=7.5)

            sleep_data = SleepData(
                total_sleep_minutes=420,
                sleep_efficiency=0.93,
                time_to_sleep_minutes=10,
                wake_count=1,
                sleep_stages={
                    SleepStage.AWAKE: 20,
                    SleepStage.REM: 100,
                    SleepStage.LIGHT: 200,
                    SleepStage.DEEP: 100
                },
                sleep_start=sleep_start,
                sleep_end=sleep_end
            )

            metric = HealthMetric(
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                sleep_data=sleep_data,
                device_id="test_device",
                raw_data={"test": "data"},
                metadata={"test": "metadata"}
            )
            metrics.append(metric)

        result = self.processor.process(metrics)

        # Should have excellent consistency (>0.9)
        assert result.consistency_score > 0.9

    def test_process_no_sleep_stages(self):
        """Test processing when sleep stages are not available."""
        sleep_data = SleepData(
            total_sleep_minutes=465,  # Match 8 hour window - 15 min latency
            sleep_efficiency=0.97,    # 465/480
            time_to_sleep_minutes=12,
            wake_count=1,
            sleep_stages=None,  # No stage breakdown
            sleep_start=datetime(2024, 6, 1, 23, 30, tzinfo=UTC),
            sleep_end=datetime(2024, 6, 2, 7, 30, tzinfo=UTC)  # 8 hours total
        )

        metric = HealthMetric(
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            sleep_data=sleep_data,
            device_id="test_device",
            raw_data={"test": "data"},
            metadata={"test": "metadata"}
        )

        result = self.processor.process([metric])

        assert result.total_sleep_minutes == 465
        assert result.sleep_efficiency == 0.97
        assert result.rem_percentage == 0.0  # No stage data
        assert result.deep_percentage == 0.0  # No stage data
        # WASO calculated from time in bed calculation
        assert result.waso_minutes == pytest.approx(3.0, abs=1.0)  # 480 - 465 - 12

    def test_process_empty_metrics(self):
        """Test processing with empty metrics list."""
        result = self.processor.process([])

        assert result.total_sleep_minutes == 0
        assert result.sleep_efficiency == 0.0
        assert result.sleep_latency == 0.0
        assert result.waso_minutes == 0.0
        assert result.awakenings_count == 0
        assert result.rem_percentage == 0.0
        assert result.deep_percentage == 0.0
        assert result.consistency_score == 0.0

    def test_get_summary_stats(self, sample_sleep_metric):
        """Test summary statistics generation."""
        result = self.processor.process([sample_sleep_metric])

        summary = self.processor.get_summary_stats(result)

        assert "sleep_quality_score" in summary
        assert "total_nights_analyzed" in summary
        assert "avg_sleep_duration_hours" in summary
        assert "sleep_efficiency_rating" in summary

        # Verify reasonable values
        assert 0 <= summary["sleep_quality_score"] <= 1
        assert summary["total_nights_analyzed"] == 1
        assert summary["avg_sleep_duration_hours"] == pytest.approx(7.75, abs=0.1)  # 465/60
        assert summary["sleep_efficiency_rating"] in ["excellent", "good", "fair", "poor"]
