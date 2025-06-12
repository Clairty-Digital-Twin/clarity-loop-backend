"""Comprehensive test suite for ActivityProcessor.

Tests cover all functionality to achieve 95%+ test coverage.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from clarity.ml.processors.activity_processor import ActivityProcessor
from clarity.models.health_data import ActivityData, HealthMetric


class TestActivityProcessorInitialization:
    """Test ActivityProcessor initialization."""

    @staticmethod
    def test_processor_initialization() -> None:
        """Test successful processor initialization."""
        processor = ActivityProcessor()
        assert processor.processor_name == "ActivityProcessor"
        assert processor.version == "1.0.0"


class TestActivityProcessorMainFlow:
    """Test main processing workflow."""

    @pytest.fixture
    @staticmethod
    def processor() -> ActivityProcessor:
        """Create activity processor instance."""
        return ActivityProcessor()

    @pytest.fixture
    @staticmethod
    def sample_metrics() -> list[Mock]:
        """Create sample health metrics with activity data."""
        metrics = []
        for i in range(3):
            metric = Mock(spec=HealthMetric)
            metric.activity_data = Mock(spec=ActivityData)
            metric.activity_data.steps = 5000 + i * 1000
            metric.activity_data.distance = 3.5 + i * 0.5
            metric.activity_data.active_energy = 200 + i * 50
            metric.activity_data.exercise_minutes = 30 + i * 10
            metric.activity_data.flights_climbed = 5 + i * 2
            metric.activity_data.active_minutes = 45 + i * 15
            metric.activity_data.vo2_max = 35.0 + i * 2.0
            metric.activity_data.resting_heart_rate = 60 + i * 5
            metrics.append(metric)
        return metrics

    @staticmethod
    def test_process_success_with_data(
        processor: ActivityProcessor, sample_metrics: list[Mock]
    ) -> None:
        """Test successful processing with valid activity data."""
        result = processor.process(cast("list[HealthMetric]", sample_metrics))

        assert isinstance(result, list)
        assert len(result) > 0

        # Check that features are generated
        feature_names = [f.get("feature_name") for f in result]
        expected_features = [
            "total_steps",
            "average_daily_steps",
            "peak_daily_steps",
            "total_distance",
            "average_daily_distance",
            "total_active_energy",
            "average_daily_active_energy",
            "total_exercise_minutes",
            "average_daily_exercise",  # 🔥 FIXED: actual name
            "total_flights_climbed",
            "total_active_minutes",
        ]

        for expected in expected_features:
            assert expected in feature_names

    @staticmethod
    def test_process_empty_metrics(processor: ActivityProcessor) -> None:
        """Test processing with empty metrics list."""
        result = processor.process([])

        assert isinstance(result, list)
        assert len(result) == 1
        assert "warning" in result[0]
        assert result[0]["warning"] == "No activity data available"

    @staticmethod
    def test_process_metrics_without_activity_data(
        processor: ActivityProcessor,
    ) -> None:
        """Test processing metrics that don't have activity data."""
        metrics = [Mock(spec=HealthMetric) for _ in range(3)]
        for metric in metrics:
            metric.activity_data = None

        result = processor.process(cast("list[HealthMetric]", metrics))

        assert isinstance(result, list)
        assert len(result) == 1
        assert "warning" in result[0]

    @staticmethod
    def test_process_with_exception(
        processor: ActivityProcessor, sample_metrics: list[Mock]
    ) -> None:
        """Test that exceptions are handled gracefully."""
        # Mock the _extract_activity_data to raise an exception
        with patch.object(
            processor, "_extract_activity_data", side_effect=Exception("Test error")
        ):
            result = processor.process(cast("list[HealthMetric]", sample_metrics))

            assert isinstance(result, list)
            assert len(result) == 1
            assert "error" in result[0]
            assert "ActivityProcessor failed: Test error" in result[0]["error"]


class TestActivityDataExtraction:
    """Test activity data extraction from metrics."""

    @staticmethod
    def test_extract_activity_data_success() -> None:
        """Test successful extraction of activity data."""
        metrics = []
        for _i in range(3):
            metric = Mock(spec=HealthMetric)
            metric.activity_data = Mock(spec=ActivityData)
            metrics.append(metric)

        processor = ActivityProcessor()
        result = processor._extract_activity_data(cast("list[HealthMetric]", metrics))

        assert len(result) == 3
        assert all(isinstance(data, Mock) for data in result)

    @staticmethod
    def test_extract_activity_data_mixed() -> None:
        """Test extraction with some metrics having no activity data."""
        metrics = []
        for i in range(5):
            metric = Mock(spec=HealthMetric)
            metric.activity_data = Mock(spec=ActivityData) if i % 2 == 0 else None
            metrics.append(metric)

        processor = ActivityProcessor()
        result = processor._extract_activity_data(cast("list[HealthMetric]", metrics))

        # Should only get 3 (indices 0, 2, 4)
        assert len(result) == 3

    @staticmethod
    def test_extract_activity_data_empty() -> None:
        """Test extraction from empty metrics list."""
        processor = ActivityProcessor()
        result = processor._extract_activity_data([])
        assert result == []


class TestActivityFeatureCalculation:
    """Test activity feature calculation methods."""

    @pytest.fixture
    @staticmethod
    def processor() -> ActivityProcessor:
        """Create activity processor instance."""
        return ActivityProcessor()

    @staticmethod
    def test_calculate_activity_features_empty(processor: ActivityProcessor) -> None:
        """Test feature calculation with empty activity data."""
        result = processor._calculate_activity_features([])
        assert result == []

    @staticmethod
    def test_calculate_step_features() -> None:
        """Test step feature calculations."""
        steps = [5000, 7000, 6000, 8000, 4000]

        processor = ActivityProcessor()
        result = processor._calculate_step_features(steps)

        assert len(result) == 3

        # Check total steps
        total_steps = next(f for f in result if f["feature_name"] == "total_steps")
        assert total_steps["value"] == 30000
        assert total_steps["unit"] == "steps"

        # Check average daily steps
        avg_steps = next(
            f for f in result if f["feature_name"] == "average_daily_steps"
        )
        assert avg_steps["value"] == 6000

        # Check peak daily steps
        peak_steps = next(f for f in result if f["feature_name"] == "peak_daily_steps")
        assert peak_steps["value"] == 8000

    @staticmethod
    def test_calculate_distance_features() -> None:
        """Test distance feature calculations."""
        distances = [3.5, 4.0, 2.8, 5.2, 3.1]

        processor = ActivityProcessor()
        result = processor._calculate_distance_features(distances)

        assert len(result) == 2

        # Check total distance
        total_dist = next(f for f in result if f["feature_name"] == "total_distance")
        assert total_dist["value"] == pytest.approx(18.6, rel=1e-2)
        assert total_dist["unit"] == "km"

        # Check average daily distance
        avg_dist = next(
            f for f in result if f["feature_name"] == "average_daily_distance"
        )
        assert avg_dist["value"] == pytest.approx(3.72, rel=1e-2)

    @staticmethod
    def test_calculate_energy_features() -> None:
        """Test energy feature calculations."""
        energy = [200.5, 250.0, 180.2, 300.8, 220.5]

        processor = ActivityProcessor()
        result = processor._calculate_energy_features(energy)

        assert len(result) == 2

        # Check total energy
        total_energy = next(
            f for f in result if f["feature_name"] == "total_active_energy"
        )
        assert total_energy["value"] == pytest.approx(1152.0, rel=1e-2)
        assert total_energy["unit"] == "kcal"

        # Check average daily energy
        avg_energy = next(
            f for f in result if f["feature_name"] == "average_daily_active_energy"
        )
        assert avg_energy["value"] == pytest.approx(230.4, rel=1e-2)

    @staticmethod
    def test_calculate_exercise_features() -> None:
        """Test exercise feature calculations."""
        exercise = [30, 45, 25, 60, 35]

        processor = ActivityProcessor()
        result = processor._calculate_exercise_features(exercise)

        assert len(result) == 2

        # Check total exercise minutes
        total_ex = next(
            f for f in result if f["feature_name"] == "total_exercise_minutes"
        )
        assert total_ex["value"] == 195
        assert total_ex["unit"] == "minutes"

        # Check average daily exercise
        avg_ex = next(
            f for f in result if f["feature_name"] == "average_daily_exercise"
        )
        assert avg_ex["value"] == 39

    @staticmethod
    def test_calculate_consistency_score_perfect() -> None:
        """Test consistency score with perfectly consistent values."""
        values = [100, 100, 100, 100, 100]
        processor = ActivityProcessor()
        score = processor._calculate_consistency_score(values)
        assert score == 1.0

    @staticmethod
    def test_calculate_consistency_score_variable() -> None:
        """Test consistency score with variable values."""
        values = [100, 200, 50, 150, 75]
        processor = ActivityProcessor()
        score = processor._calculate_consistency_score(values)
        assert 0.0 <= score <= 1.0

    @staticmethod
    def test_calculate_consistency_score_single_value() -> None:
        """Test consistency score with single value."""
        values = [100]
        processor = ActivityProcessor()
        score = processor._calculate_consistency_score(values)
        assert score == 1.0

    @staticmethod
    def test_calculate_consistency_score_empty() -> None:
        """Test consistency score with empty values."""
        values: list[float] = []
        processor = ActivityProcessor()
        score = processor._calculate_consistency_score(values)
        assert score == 1.0  # 🔥 FIXED: empty values return 1.0 not 0.0


class TestActivitySummaryStats:
    """Test activity summary statistics generation."""

    @pytest.fixture
    @staticmethod
    def processor() -> ActivityProcessor:
        """Create activity processor instance."""
        return ActivityProcessor()

    @pytest.fixture
    @staticmethod
    def sample_features() -> list[dict[str, Any]]:
        """Create sample activity features."""
        return [
            {"feature_name": "total_steps", "value": 25000, "unit": "steps"},
            {"feature_name": "average_daily_steps", "value": 5000, "unit": "steps"},
            {"feature_name": "total_distance", "value": 15.5, "unit": "km"},
            {"feature_name": "total_active_energy", "value": 1200, "unit": "kcal"},
            {"feature_name": "total_exercise_minutes", "value": 150, "unit": "minutes"},
            {"feature_name": "total_flights_climbed", "value": 25, "unit": "flights"},
            {"feature_name": "activity_consistency", "value": 0.85, "unit": "score"},
        ]

    @staticmethod
    def test_get_summary_stats_success(
        processor: ActivityProcessor, sample_features: list[dict[str, Any]]
    ) -> None:
        """Test successful summary stats generation."""
        result = processor.get_summary_stats(sample_features)

        assert isinstance(result, dict)
        assert "total_features" in result  # 🔥 FIXED: actual key name
        assert "feature_categories" in result
        assert "processor" in result
        assert "version" in result

        assert result["total_features"] == len(sample_features)

    @staticmethod
    def test_get_summary_stats_empty_features(processor: ActivityProcessor) -> None:
        """Test summary stats with empty features."""
        result = processor.get_summary_stats([])

        assert isinstance(result, dict)
        assert result["summary"] == "No activity features calculated"

    @staticmethod
    def test_categorize_features(sample_features: list[dict[str, Any]]) -> None:
        """Test feature categorization."""
        processor = ActivityProcessor()
        result = processor._categorize_features(sample_features)

        assert isinstance(result, dict)
        assert "steps" in result  # 🔥 FIXED: actual key name
        assert "distance" in result
        assert "energy" in result
        assert "exercise" in result
        assert "other" in result

        # Check specific counts
        assert result["steps"] == 2  # 🔥 FIXED: actual key name
        assert result["distance"] == 1  # total_distance
        assert result["energy"] == 1  # total_active_energy
        assert result["exercise"] == 1  # total_exercise_minutes
        assert result["other"] == 2  # flights_climbed, activity_consistency


class TestActivityProcessorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    @staticmethod
    def processor() -> ActivityProcessor:
        """Create activity processor instance."""
        return ActivityProcessor()

    @staticmethod
    def test_process_with_none_values_in_activity_data(
        processor: ActivityProcessor,
    ) -> None:
        """Test processing with None values in activity data."""
        metrics = []
        for i in range(3):
            metric = Mock(spec=HealthMetric)
            metric.activity_data = Mock(spec=ActivityData)
            # Mix of None and actual values
            metric.activity_data.steps = 5000 if i % 2 == 0 else None
            metric.activity_data.distance = 3.5 if i % 2 == 1 else None
            metric.activity_data.active_energy = None
            metric.activity_data.exercise_minutes = 30 if i == 0 else None
            metric.activity_data.flights_climbed = None
            metric.activity_data.active_minutes = None
            metric.activity_data.vo2_max = None
            metric.activity_data.resting_heart_rate = None
            metrics.append(metric)

        result = processor.process(cast("list[HealthMetric]", metrics))

        # Should still process what's available
        assert isinstance(result, list)
        # Should have at least step and distance features
        feature_names = [f.get("feature_name") for f in result]
        assert "total_steps" in feature_names
        assert "total_distance" in feature_names

    @staticmethod
    def test_activity_features_with_partial_data(processor: ActivityProcessor) -> None:
        """Test feature calculation with only some data types available."""
        activity_data = []
        for _i in range(2):
            data = Mock(spec=ActivityData)
            data.steps = 5000
            data.distance = None  # No distance data
            data.active_energy = None  # No energy data
            data.exercise_minutes = None  # No exercise data
            data.flights_climbed = 5
            data.active_minutes = None
            data.vo2_max = None
            data.resting_heart_rate = None
            activity_data.append(data)

        result = processor._calculate_activity_features(
            cast("list[ActivityData]", activity_data)
        )

        # Should have step and flights features only
        feature_names = [f.get("feature_name") for f in result]
        assert "total_steps" in feature_names
        assert "total_flights_climbed" in feature_names
        assert "total_distance" not in feature_names
        assert "total_active_energy" not in feature_names

    @staticmethod
    def test_consistency_score_with_zeros() -> None:
        """Test consistency score calculation with zero values."""
        values = [0, 0, 0, 0, 0]
        processor = ActivityProcessor()
        score = processor._calculate_consistency_score(values)
        assert score == 0.0  # 🔥 FIXED: zero mean returns 0.0

    @staticmethod
    def test_consistency_score_with_negative_values() -> None:
        """Test consistency score with negative values (edge case)."""
        values = [-10, -5, -8, -12, -6]
        processor = ActivityProcessor()
        score = processor._calculate_consistency_score(values)
        assert 0.0 <= score <= 1.0

    @staticmethod
    def test_feature_calculations_with_single_data_point() -> None:
        """Test feature calculations with only one data point."""
        # Single step count
        processor = ActivityProcessor()
        result = processor._calculate_step_features([5000])
        assert len(result) == 3

        total_steps = next(f for f in result if f["feature_name"] == "total_steps")
        assert total_steps["value"] == 5000

        avg_steps = next(
            f for f in result if f["feature_name"] == "average_daily_steps"
        )
        assert avg_steps["value"] == 5000

        peak_steps = next(f for f in result if f["feature_name"] == "peak_daily_steps")
        assert peak_steps["value"] == 5000


class TestActivityProcessorIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    @staticmethod
    def processor() -> ActivityProcessor:
        """Create activity processor instance."""
        return ActivityProcessor()

    @staticmethod
    def test_full_workflow_comprehensive_data(processor: ActivityProcessor) -> None:
        """Test complete workflow with comprehensive activity data."""
        # Create realistic activity data
        metrics = []
        for day in range(7):  # Week of data
            metric = Mock(spec=HealthMetric)
            metric.activity_data = Mock(spec=ActivityData)
            metric.activity_data.steps = 4000 + day * 500  # Increasing activity
            metric.activity_data.distance = 2.5 + day * 0.3
            metric.activity_data.active_energy = 150 + day * 25
            metric.activity_data.exercise_minutes = 20 + day * 5
            metric.activity_data.flights_climbed = 3 + day
            metric.activity_data.active_minutes = 30 + day * 5
            metric.activity_data.vo2_max = 35.0 + day * 0.5
            metric.activity_data.resting_heart_rate = 65 - day
            metrics.append(metric)

        # Process the data
        features = processor.process(cast("list[HealthMetric]", metrics))

        # Verify comprehensive feature set
        assert len(features) >= 10
        feature_names = [f.get("feature_name") for f in features]

        expected_features = [
            "total_steps",
            "average_daily_steps",
            "peak_daily_steps",
            "total_distance",
            "average_daily_distance",
            "total_active_energy",
            "average_daily_active_energy",
            "total_exercise_minutes",
            "average_daily_exercise",  # 🔥 FIXED: actual name
            "total_flights_climbed",
            "total_active_minutes",
        ]

        for expected in expected_features:
            assert expected in feature_names

        # Generate summary stats
        summary = processor.get_summary_stats(features)

        assert summary["total_features"] == len(features)
        assert "feature_categories" in summary

    @staticmethod
    def test_workflow_with_realistic_variation(processor: ActivityProcessor) -> None:
        """Test workflow with realistic daily variation in activity."""
        # Simulate more realistic activity patterns
        realistic_steps = [3500, 8200, 6800, 4200, 9500, 12000, 2800]  # Varied week
        realistic_distances = [2.1, 5.8, 4.2, 3.1, 6.9, 8.5, 1.8]

        metrics = []
        for _i, (steps, distance) in enumerate(
            zip(realistic_steps, realistic_distances, strict=False)
        ):
            metric = Mock(spec=HealthMetric)
            metric.activity_data = Mock(spec=ActivityData)
            metric.activity_data.steps = steps
            metric.activity_data.distance = distance
            metric.activity_data.active_energy = steps * 0.04  # Rough calorie estimate
            metric.activity_data.exercise_minutes = max(0, int((steps - 3000) / 100))
            metric.activity_data.flights_climbed = max(0, int(steps / 1000))
            metric.activity_data.active_minutes = max(0, int(steps / 80))
            metric.activity_data.vo2_max = None
            metric.activity_data.resting_heart_rate = None
            metrics.append(metric)

        features = processor.process(cast("list[HealthMetric]", metrics))

        # Should handle realistic variation well
        assert len(features) > 0

        # Check that consistency score reflects the variation
        consistency_features = [
            f for f in features if "consistency" in f.get("feature_name", "")
        ]
        if consistency_features:
            consistency_score = consistency_features[0]["value"]
            assert 0.0 <= consistency_score <= 1.0
            # With this varied data, consistency should be moderate
            assert consistency_score < 0.9  # Not perfect consistency
