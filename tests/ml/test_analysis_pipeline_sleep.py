"""Integration tests for SleepProcessor in AnalysisPipeline.

Tests the complete integration of sleep processing within the analysis pipeline,
ensuring proper data flow and feature fusion.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from clarity.ml.analysis_pipeline import HealthAnalysisPipeline
from clarity.ml.processors.sleep_processor import SleepFeatures
from clarity.models.health_data import (
    BiometricData,
    HealthMetric,
    HealthMetricType,
    SleepData,
    SleepStage,
)


class TestAnalysisPipelineSleepIntegration:
    """Integration tests for sleep processing in analysis pipeline."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.pipeline = HealthAnalysisPipeline()

    @pytest.fixture
    @staticmethod
    def sample_sleep_metrics() -> list[HealthMetric]:
        """Create sample sleep metrics for testing."""
        return [
            HealthMetric(
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                sleep_data=SleepData(
                    total_sleep_minutes=465,  # 7h 45min
                    sleep_efficiency=0.97,  # 97%
                    time_to_sleep_minutes=15,
                    wake_count=2,
                    sleep_stages={
                        SleepStage.AWAKE: 15,
                        SleepStage.REM: 90,
                        SleepStage.LIGHT: 285,
                        SleepStage.DEEP: 90,
                    },
                    sleep_start=datetime(2024, 6, 1, 23, 0, tzinfo=UTC),
                    sleep_end=datetime(2024, 6, 2, 7, 0, tzinfo=UTC),
                ),
                device_id="test_device",
                raw_data={"test": "data"},
                metadata={"test": "metadata"},
            )
        ]

    @pytest.mark.asyncio
    async def test_sleep_processing_in_pipeline(
        self, sample_sleep_metrics: list[HealthMetric]
    ) -> None:
        """Test sleep processing integration in analysis pipeline."""
        # Mock dependencies
        mock_firestore = AsyncMock()
        self.pipeline._firestore_client = mock_firestore

        # Process sleep data
        results = await self.pipeline.process_health_data(
            user_id="test_user", health_metrics=sample_sleep_metrics
        )

        # Verify sleep features were extracted
        assert hasattr(results, "sleep_features")
        assert results.sleep_features is not None
        assert isinstance(results.sleep_features, dict)

        # Verify key sleep metrics
        sleep_features = results.sleep_features
        assert "total_sleep_minutes" in sleep_features
        assert "sleep_efficiency" in sleep_features
        assert "overall_quality_score" in sleep_features

    @pytest.mark.asyncio
    async def test_sleep_vector_fusion(
        self, sample_sleep_metrics: list[HealthMetric]
    ) -> None:
        """Test sleep feature vector integration in modality fusion."""
        # Mock dependencies
        mock_firestore = AsyncMock()
        self.pipeline._firestore_client = mock_firestore

        # Process sleep data
        results = await self.pipeline.process_health_data(
            user_id="test_user", health_metrics=sample_sleep_metrics
        )

        # Should have fused vector including sleep modality
        assert hasattr(results, "fused_vector")
        assert len(results.fused_vector) > 0

    @pytest.mark.asyncio
    async def test_sleep_with_multiple_modalities(
        self, sample_sleep_metrics: list[HealthMetric]
    ) -> None:
        """Test sleep processing with other health modalities."""
        # Add some cardio data to test multi-modal fusion
        multi_modal_metrics = [
            *sample_sleep_metrics,
            HealthMetric(
                metric_type=HealthMetricType.HEART_RATE,
                biometric_data=BiometricData(heart_rate=72.0),
                device_id="test_device",
                raw_data={"hr": 72},
                metadata={},
            ),
        ]

        # Mock dependencies
        mock_firestore = AsyncMock()
        self.pipeline._firestore_client = mock_firestore

        # Process multi-modal data
        results = await self.pipeline.process_health_data(
            user_id="test_user", health_metrics=multi_modal_metrics
        )

        # Should have both sleep and cardio features
        assert hasattr(results, "sleep_features")
        assert hasattr(results, "cardio_features")
        assert results.sleep_features is not None
        assert len(results.cardio_features) > 0

        # Should have fused vector
        assert len(results.fused_vector) > 0

    @pytest.mark.asyncio
    async def test_sleep_features_validation(
        self, sample_sleep_metrics: list[HealthMetric]
    ) -> None:
        """Test that sleep features are properly validated."""
        # Mock dependencies
        mock_firestore = AsyncMock()
        self.pipeline._firestore_client = mock_firestore

        # Process sleep data
        results = await self.pipeline.process_health_data(
            user_id="test_user", health_metrics=sample_sleep_metrics
        )

        # Validate sleep features structure
        sleep_features = results.sleep_features
        expected_fields = [
            "total_sleep_minutes",
            "sleep_efficiency",
            "sleep_latency",
            "waso_minutes",
            "awakenings_count",
            "rem_percentage",
            "deep_percentage",
            "light_percentage",
            "consistency_score",
            "overall_quality_score",
        ]

        for field in expected_fields:
            assert field in sleep_features, f"Missing field: {field}"
            assert isinstance(
                sleep_features[field], (int, float)
            ), f"Field {field} should be numeric"

        # Validate reasonable ranges
        assert 0 <= sleep_features["sleep_efficiency"] <= 1
        assert 0 <= sleep_features["rem_percentage"] <= 1
        assert 0 <= sleep_features["deep_percentage"] <= 1
        assert 0 <= sleep_features["overall_quality_score"] <= 1

    @pytest.mark.asyncio
    async def test_empty_sleep_data_handling(self) -> None:
        """Test pipeline handles empty sleep data gracefully."""
        # Mock dependencies
        mock_firestore = AsyncMock()
        self.pipeline._firestore_client = mock_firestore

        # Process empty data
        results = await self.pipeline.process_health_data(
            user_id="test_user", health_metrics=[]
        )

        # Should handle gracefully without sleep features
        assert not hasattr(results, "sleep_features") or not results.sleep_features

    @staticmethod
    def test_sleep_vector_conversion() -> None:
        """Test sleep features to vector conversion."""
        # Create test sleep features
        test_features = SleepFeatures(
            total_sleep_minutes=420.0,  # 7 hours
            sleep_efficiency=0.85,
            sleep_latency=20.0,
            waso_minutes=30.0,
            awakenings_count=3.0,
            rem_percentage=0.20,
            deep_percentage=0.15,
            light_percentage=0.65,
            consistency_score=0.8,
            overall_quality_score=0.75,
        )

        # Test conversion
        vector = HealthAnalysisPipeline._convert_sleep_features_to_vector(
            test_features
        )

        # Verify vector structure
        assert len(vector) == 8
        assert all(isinstance(v, float) for v in vector)

        # Verify normalization (values should be roughly 0-1 range)
        assert 0 <= vector[0] <= 1.5  # total_sleep normalized by 8 hours
        assert 0 <= vector[1] <= 1  # efficiency already 0-1
        assert 0 <= vector[2] <= 1  # latency normalized by 1 hour
