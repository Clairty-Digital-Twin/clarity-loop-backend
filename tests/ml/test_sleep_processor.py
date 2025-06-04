"""Comprehensive tests for SleepProcessor.

This test suite follows TDD principles and mirrors the patterns from existing
processor tests (cardio, activity, respiration) while achieving high coverage.
"""

import pytest
from datetime import datetime, UTC, timedelta
from unittest.mock import Mock, patch
import numpy as np

from clarity.ml.processors.sleep_processor import SleepProcessor
from clarity.models.health_data import (
    HealthMetric,
    HealthMetricType,
    SleepData,
    SleepStage
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
            total_sleep_minutes=420,  # 7 hours
            sleep_efficiency=0.875,   # 87.5%
            time_to_sleep_minutes=15,
            wake_count=2,
            sleep_stages={
                SleepStage.AWAKE: 45,
                SleepStage.REM: 90,
                SleepStage.LIGHT: 240,
                SleepStage.DEEP: 90
            },
            sleep_start=datetime(2024, 6, 1, 23, 0, tzinfo=UTC),
            sleep_end=datetime(2024, 6, 2, 7, 0, tzinfo=UTC)
        )

    @pytest.fixture
    def sample_sleep_metric(self, sample_sleep_data):
        """Create a sample sleep metric."""
        return HealthMetric(
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            sleep_data=sample_sleep_data
        )

    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        assert self.processor.processor_name == "SleepProcessor"
        assert self.processor.version == "1.0.0"

    def test_process_single_night_complete_data(self, sample_sleep_metric):
        """Test processing single night with complete sleep data."""
        metrics = [sample_sleep_metric]
        
        result = self.processor.process(metrics)
        
        assert result.total_sleep_minutes == 420
        assert result.sleep_efficiency == 0.875
        assert result.sleep_latency == 15.0
        assert result.waso_minutes == 45.0  # Awake minutes after sleep onset
        assert result.awakenings_count == 2
        assert result.rem_percentage == pytest.approx(0.214, abs=0.001)  # 90/420
        assert result.deep_percentage == pytest.approx(0.214, abs=0.001)  # 90/420
        assert result.consistency_score == 0.5  # Default for single night

    def test_process_multiple_nights_consistency_calculation(self):
        """Test consistency score calculation with multiple nights."""
        # Create 3 nights with varying sleep start times
        metrics = []
        base_time = datetime(2024, 6, 1, 23, 0, tzinfo=UTC)
        
        for i, offset_hours in enumerate([0, 0.5, 1.0]):  # 23:00, 23:30, 00:00
            sleep_start = base_time + timedelta(hours=offset_hours)
            sleep_end = sleep_start + timedelta(hours=8)
            
            sleep_data = SleepData(
                total_sleep_minutes=420,
                sleep_efficiency=0.85,
                time_to_sleep_minutes=10,
                wake_count=1,
                sleep_stages={
                    SleepStage.AWAKE: 30,
                    SleepStage.REM: 90,
                    SleepStage.LIGHT: 240,
                    SleepStage.DEEP: 90
                },
                sleep_start=sleep_start,
                sleep_end=sleep_end
            )
            
            metric = HealthMetric(
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                sleep_data=sleep_data
            )
            metrics.append(metric)
        
        result = self.processor.process(metrics)
        
        # Consistency should be lower due to 1-hour variation
        assert result.consistency_score < 0.8
        assert result.consistency_score > 0.0

    def test_process_excellent_consistency(self):
        """Test excellent consistency score with regular sleep times."""
        metrics = []
        base_time = datetime(2024, 6, 1, 23, 0, tzinfo=UTC)
        
        # Create 5 nights with very consistent sleep times (within 10 minutes)
        for i in range(5):
            sleep_start = base_time + timedelta(days=i, minutes=i*5)  # 0-20 min variation
            sleep_end = sleep_start + timedelta(hours=8)
            
            sleep_data = SleepData(
                total_sleep_minutes=420,
                sleep_efficiency=0.90,
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
                sleep_data=sleep_data
            )
            metrics.append(metric)
        
        result = self.processor.process(metrics)
        
        # Should have excellent consistency (>0.9)
        assert result.consistency_score > 0.9

    def test_process_no_sleep_stages(self):
        """Test processing when sleep stages are not available."""
        sleep_data = SleepData(
            total_sleep_minutes=400,
            sleep_efficiency=0.83,
            time_to_sleep_minutes=12,
            wake_count=1,
            sleep_stages=None,  # No stage breakdown
            sleep_start=datetime(2024, 6, 1, 23, 30, tzinfo=UTC),
            sleep_end=datetime(2024, 6, 2, 7, 30, tzinfo=UTC)
        )
        
        metric = HealthMetric(
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            sleep_data=sleep_data
        )
        
        result = self.processor.process([metric])
        
        assert result.total_sleep_minutes == 400
        assert result.sleep_efficiency == 0.83
        assert result.rem_percentage == 0.0  # No stage data
        assert result.deep_percentage == 0.0  # No stage data
        # WASO calculated from time in bed calculation
        assert result.waso_minutes == pytest.approx(68.0, abs=1.0)  # 480 - 400 - 12

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
        assert summary["avg_sleep_duration_hours"] == 7.0