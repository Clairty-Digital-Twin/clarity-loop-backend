"""Unit tests for PersonalBaselineTracker module."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from clarity.ml.personal_baseline_tracker import PersonalBaselineTracker
from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthMetric,
    HealthMetricType,
    SleepData,
)


class TestPersonalBaselineTracker:
    """Test suite for personal baseline tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = PersonalBaselineTracker(min_days_for_baseline=7)
        self.user_id = "test_user_123"

    def create_health_metrics(self, days=28, base_date=None):
        """Create a set of health metrics for testing."""
        if base_date is None:
            base_date = datetime.now(UTC)

        metrics = []
        for i in range(days):
            date = base_date - timedelta(days=i)

            # Create sleep data with some variation
            sleep_hours = 7.5 + np.sin(i * 0.3) * 0.5
            sleep_start = date.replace(hour=23, minute=0) - timedelta(
                hours=np.sin(i * 0.2)
            )
            sleep_end = sleep_start + timedelta(hours=sleep_hours)

            sleep_data = SleepData(
                sleep_start=sleep_start,
                sleep_end=sleep_end,
                total_sleep_minutes=int(sleep_hours * 60),
                sleep_efficiency=0.85 + np.sin(i * 0.4) * 0.05,
                time_to_sleep_minutes=15 + i % 10,
            )

            # Create activity data
            activity_data = ActivityData(
                steps=10000 + int(np.sin(i * 0.25) * 2000),
                active_energy=300 + int(np.sin(i * 0.35) * 50),
                exercise_minutes=30 + i % 20,
            )

            # Create biometric data
            bio_data = BiometricData(
                heart_rate=60 + int(np.sin(i * 0.15) * 5),
                heart_rate_variability=50 + int(np.sin(i * 0.45) * 10),
            )

            metric = HealthMetric(
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                sleep_data=sleep_data,
                activity_data=activity_data,
                biometric_data=bio_data,
                created_at=date,
            )
            metrics.append(metric)

        return metrics

    def test_update_baseline_first_time(self):
        """Test creating baseline for new user."""
        metrics = self.create_health_metrics(days=14)

        baseline = self.tracker.update_baseline(self.user_id, metrics)

        assert baseline is not None
        assert baseline.user_id == self.user_id
        assert baseline.data_days == 14

        # Check sleep baselines
        assert 7.0 <= baseline.sleep_duration_median <= 8.0
        assert baseline.sleep_duration_p25 < baseline.sleep_duration_median
        assert baseline.sleep_duration_p75 > baseline.sleep_duration_median

        # Check activity baselines
        assert 8000 <= baseline.daily_steps_median <= 12000
        assert baseline.daily_steps_p25 < baseline.daily_steps_median
        assert baseline.daily_steps_p75 > baseline.daily_steps_median

        # Check physiological baselines
        assert 55 <= baseline.resting_hr_median <= 65
        assert 40 <= baseline.hrv_median <= 60

        # Check confidence
        assert baseline.confidence_score > 0.7

    def test_update_baseline_incremental(self):
        """Test updating existing baseline with new data."""
        # Create initial baseline
        initial_metrics = self.create_health_metrics(days=14)
        initial_baseline = self.tracker.update_baseline(self.user_id, initial_metrics)

        # Add new data (with newer timestamps)
        import time

        time.sleep(0.001)  # Ensure timestamp difference
        new_metrics = self.create_health_metrics(days=7, base_date=datetime.now(UTC))
        updated_baseline = self.tracker.update_baseline(self.user_id, new_metrics)

        assert updated_baseline is not None
        assert updated_baseline.last_updated >= initial_baseline.last_updated
        assert updated_baseline.data_days == 7  # Only counts new data

    def test_calculate_deviation_scores(self):
        """Test deviation score calculation from baseline."""
        # Create baseline
        metrics = self.create_health_metrics(days=28)
        baseline = self.tracker.update_baseline(self.user_id, metrics)

        # Test normal values (should have low z-scores)
        normal_metrics = {
            "sleep_hours": baseline.sleep_duration_median,
            "daily_steps": baseline.daily_steps_median,
            "resting_hr": baseline.resting_hr_median,
        }

        deviations = self.tracker.calculate_deviation_scores(
            self.user_id, normal_metrics
        )

        assert abs(deviations["sleep_duration_z"]) < 0.5
        assert abs(deviations["activity_z"]) < 0.5
        assert abs(deviations["hr_z"]) < 0.5

    def test_calculate_deviation_extreme_values(self):
        """Test deviation scores for extreme values."""
        # Create baseline
        metrics = self.create_health_metrics(days=28)
        _ = self.tracker.update_baseline(self.user_id, metrics)

        # Test extreme values
        extreme_metrics = {
            "sleep_hours": 3.0,  # Very low sleep
            "daily_steps": 25000,  # Very high activity
            "resting_hr": 90,  # Elevated HR
        }

        deviations = self.tracker.calculate_deviation_scores(
            self.user_id, extreme_metrics
        )

        assert deviations["sleep_duration_z"] < -2.0  # Significantly below normal
        assert deviations["activity_z"] > 2.0  # Significantly above normal
        assert deviations["hr_z"] > 2.0  # Significantly above normal

    def test_circadian_shift_calculation(self):
        """Test circadian shift detection."""
        # Create baseline
        metrics = self.create_health_metrics(days=28)
        baseline = self.tracker.update_baseline(self.user_id, metrics)

        # Test phase advance (earlier sleep)
        early_metrics = {
            "sleep_midpoint": baseline.sleep_midpoint_median - 2  # 2 hours earlier
        }

        deviations = self.tracker.calculate_deviation_scores(
            self.user_id, early_metrics
        )
        assert deviations["circadian_shift_hours"] == pytest.approx(-2.0, abs=0.1)

        # Test phase delay (later sleep)
        late_metrics = {
            "sleep_midpoint": baseline.sleep_midpoint_median + 3  # 3 hours later
        }

        deviations = self.tracker.calculate_deviation_scores(self.user_id, late_metrics)
        assert deviations["circadian_shift_hours"] == pytest.approx(3.0, abs=0.1)

    def test_wraparound_handling(self):
        """Test handling of circadian wraparound."""
        # Create metrics with sleep around midnight
        metrics = []
        for i in range(14):
            date = datetime.now(UTC) - timedelta(days=i)
            sleep_start = date.replace(hour=23, minute=30)
            sleep_end = (date + timedelta(days=1)).replace(hour=7, minute=30)

            sleep_data = SleepData(
                sleep_start=sleep_start,
                sleep_end=sleep_end,
                total_sleep_minutes=480,
                sleep_efficiency=0.85,
            )

            metric = HealthMetric(
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                sleep_data=sleep_data,
                created_at=date,
            )
            metrics.append(metric)

        baseline = self.tracker.update_baseline(self.user_id, metrics)

        # Midpoint should be around 3:30 AM
        assert 3.0 <= baseline.sleep_midpoint_median <= 4.0

        # Test wraparound in deviation calculation
        late_night_metrics = {"sleep_midpoint": 23.0}  # 11 PM
        deviations = self.tracker.calculate_deviation_scores(
            self.user_id, late_night_metrics
        )

        # Should detect as phase advance, not delay
        assert deviations["circadian_shift_hours"] < -3.0

    def test_variability_baseline_calculation(self):
        """Test variability baseline calculation."""
        # Create highly variable data
        metrics = []
        for i in range(14):
            date = datetime.now(UTC) - timedelta(days=i)

            # Alternating high/low pattern
            if i % 2 == 0:
                steps = 15000
                sleep_hours = 9
            else:
                steps = 5000
                sleep_hours = 5

            activity_data = ActivityData(steps=steps)
            sleep_start = date.replace(hour=23)
            sleep_end = sleep_start + timedelta(hours=sleep_hours)
            sleep_data = SleepData(
                total_sleep_minutes=int(sleep_hours * 60),
                sleep_efficiency=0.85,
                sleep_start=sleep_start,
                sleep_end=sleep_end,
            )

            metric = HealthMetric(
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                activity_data=activity_data,
                sleep_data=sleep_data,
                created_at=date,
            )
            metrics.append(metric)

        baseline = self.tracker.update_baseline(self.user_id, metrics)

        # Should have high variability
        assert baseline.activity_variability_baseline > 0.3
        assert baseline.sleep_variability_baseline > 0.2

    def test_confidence_calculation(self):
        """Test baseline confidence calculation."""
        # Test with minimal data
        minimal_metrics = self.create_health_metrics(days=7)
        minimal_baseline = self.tracker.update_baseline(self.user_id, minimal_metrics)

        # Test with optimal data
        optimal_metrics = self.create_health_metrics(days=28)
        optimal_baseline = self.tracker.update_baseline(self.user_id, optimal_metrics)

        # With min_days=7, both get full data confidence
        # The only difference would be recency, which is negligible for fresh data
        assert minimal_baseline.confidence_score <= optimal_baseline.confidence_score
        assert minimal_baseline.confidence_score >= 0.7  # 7 days = minimum
        assert optimal_baseline.confidence_score >= 0.7  # Both should be high

    def test_missing_data_handling(self):
        """Test handling of missing data types."""
        # Create metrics with only sleep data
        sleep_only_metrics = []
        for i in range(14):
            date = datetime.now(UTC) - timedelta(days=i)
            sleep_data = SleepData(
                total_sleep_minutes=450,
                sleep_start=date.replace(hour=23),
                sleep_end=date.replace(hour=7) + timedelta(days=1),
                sleep_efficiency=0.85,
            )

            metric = HealthMetric(
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                sleep_data=sleep_data,
                created_at=date,
            )
            sleep_only_metrics.append(metric)

        baseline = self.tracker.update_baseline(self.user_id, sleep_only_metrics)

        assert baseline.sleep_duration_median > 0
        assert baseline.daily_steps_median == 0.0  # No activity data
        assert baseline.resting_hr_median == 0.0  # No biometric data

    def test_export_baseline(self):
        """Test baseline export functionality."""
        metrics = self.create_health_metrics(days=14)
        _ = self.tracker.update_baseline(self.user_id, metrics)

        exported = self.tracker.export_baseline(self.user_id)

        assert exported is not None
        assert exported["user_id"] == self.user_id
        assert "sleep" in exported
        assert "activity" in exported
        assert "physiology" in exported
        assert "variability" in exported
        assert "circadian" in exported
        assert "metadata" in exported

        # Check structure
        assert "duration_median_hours" in exported["sleep"]
        assert "daily_steps_median" in exported["activity"]
        assert "resting_hr_median" in exported["physiology"]
        assert "sleep_cv" in exported["variability"]
        assert "typical_bedtime" in exported["circadian"]
        assert "confidence" in exported["metadata"]

    def test_get_baseline_nonexistent_user(self):
        """Test getting baseline for non-existent user."""
        baseline = self.tracker.get_baseline("nonexistent_user")
        assert baseline is None

        deviations = self.tracker.calculate_deviation_scores("nonexistent_user", {})
        assert deviations == {}
