"""Integration tests for SleepProcessor in AnalysisPipeline.

Tests the complete integration of sleep processing within the analysis pipeline,
ensuring proper data flow and feature fusion.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from clarity.ml.analysis_pipeline import AnalysisResults, HealthAnalysisPipeline
from clarity.models.health_data import (
    BiometricData,
    HealthMetric,
    HealthMetricType,
    SleepData,
    SleepStage,
)


class TestAnalysisPipelineSleepIntegration:
    """Integration tests for sleep processing in analysis pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = HealthAnalysisPipeline()

    @pytest.fixture
    def sample_sleep_metrics(self):
        """Create sample sleep metrics for testing."""
        return [
            HealthMetric(
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                sleep_data=SleepData(
                    total_sleep_minutes=450,
                    sleep_efficiency=0.9,
                    time_to_sleep_minutes=12,
                    wake_count=2,
                    sleep_stages={
                        SleepStage.AWAKE: 30,
                        SleepStage.REM: 90,
                        SleepStage.LIGHT: 270,
                        SleepStage.DEEP: 90
                    },
                    sleep_start=datetime(2024, 6, 1, 23, 0, tzinfo=UTC),
                    sleep_end=datetime(2024, 6, 2, 7, 0, tzinfo=UTC)
                ),
                device_id="test_device",
                raw_data={"test": "sleep_data"},
                metadata={"test": "metadata"}
            )
        ]

    @pytest.mark.asyncio
    async def test_sleep_processing_in_pipeline(self, sample_sleep_metrics):
        """Test sleep processing integration in analysis pipeline."""
        # Mock dependencies
        self.pipeline.gemini_service = AsyncMock()
        self.pipeline.gemini_service.generate_insights.return_value = {
            "insights": ["Good sleep quality detected"],
            "recommendations": ["Maintain consistent bedtime"]
        }

        # Process metrics through pipeline
        results = await self.pipeline.process_health_data(sample_sleep_metrics)

        # Verify sleep features were processed
        assert isinstance(results, AnalysisResults)
        assert hasattr(results, 'sleep_features')
        assert results.sleep_features is not None

        # Verify sleep features content
        sleep_features = results.sleep_features
        assert sleep_features['total_sleep_minutes'] == 450
        assert sleep_features['sleep_efficiency'] == 0.9
        assert sleep_features['rem_percentage'] > 0.0
        assert sleep_features['deep_percentage'] > 0.0

    @pytest.mark.asyncio
    async def test_sleep_vector_fusion(self, sample_sleep_metrics):
        """Test sleep feature vector integration in modality fusion."""
        # Mock dependencies
        self.pipeline.gemini_service = AsyncMock()
        self.pipeline.gemini_service.generate_insights.return_value = {
            "insights": ["Sleep analysis complete"],
            "recommendations": ["Continue good sleep habits"]
        }

        # Process metrics
        results = await self.pipeline.process_health_data(sample_sleep_metrics)

        # Verify fused vector includes sleep modality
        assert hasattr(results, 'fused_vector')
        assert len(results.fused_vector) > 0

        # The fused vector should contain sleep feature contributions
        # (exact values depend on fusion implementation)
        assert all(isinstance(val, (int, float)) for val in results.fused_vector)

    @pytest.mark.asyncio
    async def test_sleep_with_multiple_modalities(self, sample_sleep_metrics):
        """Test sleep processing with other health modalities."""
        # Add some cardio data to test multi-modal fusion
        multi_modal_metrics = sample_sleep_metrics + [
            HealthMetric(
                metric_type=HealthMetricType.HEART_RATE,
                biometric_data=BiometricData(heart_rate=72.0),
                device_id="test_device",
                raw_data={"hr": 72},
                metadata={}
            )
        ]

        # Mock dependencies
        self.pipeline.gemini_service = AsyncMock()
        self.pipeline.gemini_service.generate_insights.return_value = {
            "insights": ["Multi-modal health analysis"],
            "recommendations": ["Maintain healthy lifestyle"]
        }

        # Process multi-modal metrics
        results = await self.pipeline.process_health_data(multi_modal_metrics)

        # Verify both sleep and cardio were processed
        assert hasattr(results, 'sleep_features')
        assert hasattr(results, 'cardio_features')
        assert results.sleep_features is not None
        assert len(results.cardio_features) > 0

        # Verify fusion includes both modalities
        assert len(results.fused_vector) > 0

    @pytest.mark.asyncio
    async def test_sleep_features_validation(self, sample_sleep_metrics):
        """Test that sleep features are properly validated."""
        # Mock dependencies
        self.pipeline.gemini_service = AsyncMock()
        self.pipeline.gemini_service.generate_insights.return_value = {
            "insights": ["Sleep validation test"],
            "recommendations": ["Test recommendation"]
        }

        # Process metrics
        results = await self.pipeline.process_health_data(sample_sleep_metrics)

        # Verify sleep features structure and ranges
        sleep_features = results.sleep_features

        # Check required fields exist
        required_fields = [
            'total_sleep_minutes', 'sleep_efficiency', 'sleep_latency',
            'waso_minutes', 'awakenings_count', 'rem_percentage',
            'deep_percentage', 'consistency_score'
        ]
        for field in required_fields:
            assert field in sleep_features, f"Missing field: {field}"

        # Validate ranges
        assert 0 <= sleep_features['sleep_efficiency'] <= 1.0
        assert 0 <= sleep_features['rem_percentage'] <= 1.0
        assert 0 <= sleep_features['deep_percentage'] <= 1.0
        assert 0 <= sleep_features['consistency_score'] <= 1.0
        assert sleep_features['total_sleep_minutes'] >= 0
        assert sleep_features['awakenings_count'] >= 0

    @pytest.mark.asyncio
    async def test_empty_sleep_data_handling(self):
        """Test pipeline handles empty sleep data gracefully."""
        # Mock dependencies
        self.pipeline.gemini_service = AsyncMock()
        self.pipeline.gemini_service.generate_insights.return_value = {
            "insights": ["No sleep data available"],
            "recommendations": ["Ensure sleep tracking is enabled"]
        }

        # Process empty metrics
        results = await self.pipeline.process_health_data([])

        # Verify pipeline completes without error
        assert isinstance(results, AnalysisResults)
        # Sleep features should not be present or should be empty
        assert not hasattr(results, 'sleep_features') or not results.sleep_features

    def test_sleep_vector_conversion(self):
        """Test sleep features to vector conversion."""
        from clarity.ml.processors.sleep_processor import SleepFeatures

        # Create test sleep features
        sleep_features = SleepFeatures(
            total_sleep_minutes=420,
            sleep_efficiency=0.85,
            sleep_latency=15.0,
            waso_minutes=30.0,
            awakenings_count=3,
            rem_percentage=0.22,
            deep_percentage=0.18,
            consistency_score=0.75
        )

        # Test conversion to vector
        vector = HealthAnalysisPipeline._convert_sleep_features_to_vector(sleep_features)

        # Verify vector properties
        assert isinstance(vector, list)
        assert len(vector) == 8  # Should have 8 normalized features
        assert all(isinstance(val, float) for val in vector)

        # Verify normalization (values should be roughly 0-1 range)
        assert all(0.0 <= val <= 2.0 for val in vector), f"Vector values out of range: {vector}"

    @pytest.mark.asyncio
    async def test_sleep_processing_with_missing_stages(self):
        """Test sleep processing when stage data is missing."""
        # Create sleep metric without stage data
        sleep_metric_no_stages = HealthMetric(
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            sleep_data=SleepData(
                total_sleep_minutes=420,
                sleep_efficiency=0.88,
                time_to_sleep_minutes=10,
                wake_count=1,
                sleep_stages=None,  # No stage breakdown
                sleep_start=datetime(2024, 6, 1, 23, 30, tzinfo=UTC),
                sleep_end=datetime(2024, 6, 2, 7, 30, tzinfo=UTC)
            ),
            device_id="test_device",
            raw_data={"test": "data"},
            metadata={}
        )

        # Mock dependencies
        self.pipeline.gemini_service = AsyncMock()
        self.pipeline.gemini_service.generate_insights.return_value = {
            "insights": ["Basic sleep analysis without stages"],
            "recommendations": ["Enable detailed sleep tracking"]
        }

        # Process metric
        results = await self.pipeline.process_health_data([sleep_metric_no_stages])

        # Verify processing completes and handles missing stages
        assert hasattr(results, 'sleep_features')
        sleep_features = results.sleep_features

        # REM and deep percentages should be 0 when stages are missing
        assert sleep_features['rem_percentage'] == 0.0
        assert sleep_features['deep_percentage'] == 0.0
        # Other metrics should still be calculated
        assert sleep_features['total_sleep_minutes'] == 420
        assert sleep_features['sleep_efficiency'] == 0.88
