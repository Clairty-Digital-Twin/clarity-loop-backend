"""Comprehensive test suite for PAT (Pretrained Actigraphy Transformer) Service.

This test suite provides thorough coverage of the PAT service functionality,
including model loading, weight conversion, inference, and clinical analysis.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import logging

import numpy as np
import pytest
import torch

from clarity.ml.pat_service import (
    PAT_CONFIGS,
    ActigraphyAnalysis,
    ActigraphyInput,
    PATEncoder,
    PATForMentalHealthClassification,
    PATModelService,
    PATMultiHeadAttention,
    get_pat_service,
)
from clarity.ml.preprocessing import ActigraphyDataPoint
from clarity.services.health_data_service import MLPredictionError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize random generator
rng = np.random.default_rng()


class TestPATModelService:
    """Test the core PAT model service functionality."""

    @pytest.fixture
    async def pat_service(self) -> PATModelService:
        """Create a PAT service instance for testing."""
        service = PATModelService(model_size="medium")
        await service.load_model()
        return service

    @pytest.fixture
    def sample_actigraphy_data(self) -> list[ActigraphyDataPoint]:
        """Generate sample actigraphy data for testing."""
        data_points = []
        base_time = datetime.now(UTC)

        # Generate 24 hours of minute-level data (1440 points)
        for minute in range(1440):
            timestamp = base_time + timedelta(minutes=minute)

            # Simulate realistic activity pattern
            hour = minute // 60
            if 6 <= hour <= 22:  # Daytime
                activity = 30 + 20 * np.sin(2 * np.pi * (hour - 6) / 16)
            else:  # Nighttime
                activity = 5

            # Add some noise
            activity += rng.normal(0, 5)
            activity = max(0, activity)

            data_points.append(
                ActigraphyDataPoint(timestamp=timestamp, value=float(activity))
            )

        return data_points

    @pytest.fixture
    def week_actigraphy_data(self) -> list[ActigraphyDataPoint]:
        """Generate a full week of actigraphy data (10080 points)."""
        data_points = []
        base_time = datetime.now(UTC)

        for minute in range(10080):  # 7 days * 24 hours * 60 minutes
            timestamp = base_time + timedelta(minutes=minute)

            # Simulate weekly pattern
            hour_of_day = (minute // 60) % 24
            day_of_week = minute // (24 * 60)

            # Weekend vs weekday patterns
            is_weekend = day_of_week >= 5

            if 6 <= hour_of_day <= 22:  # Daytime
                base_activity = 40 if not is_weekend else 25
                activity = base_activity + 15 * np.sin(
                    2 * np.pi * (hour_of_day - 6) / 16
                )
            else:  # Nighttime
                activity = 3 if not is_weekend else 7

            activity += rng.normal(0, 8)
            activity = max(0, activity)

            data_points.append(
                ActigraphyDataPoint(timestamp=timestamp, value=float(activity))
            )

        return data_points

    @staticmethod
    async def test_service_initialization() -> None:
        """Test PAT service initialization."""
        # Test default initialization
        service = PATModelService()
        assert service.model_size == "medium"
        assert not service.is_loaded
        assert service.model is None

        # Test custom initialization
        service = PATModelService(model_size="small", device="cpu")
        assert service.model_size == "small"
        assert service.device == "cpu"

    @staticmethod
    async def test_model_loading(pat_service: PATModelService) -> None:
        """Test PAT model loading with real weights."""
        assert pat_service.is_loaded
        assert pat_service.model is not None
        assert isinstance(pat_service.model, PATForMentalHealthClassification)

        # Test encoder architecture
        if pat_service.model is not None:
            encoder = pat_service.model.encoder
            assert isinstance(encoder, PATEncoder)
            assert encoder.embed_dim == 96
            assert encoder.input_size == 10080
            assert encoder.patch_size == 18
            assert len(encoder.transformer_layers) == 2  # PAT-M
        else:
            pytest.skip("Model not loaded")

    @staticmethod
    async def test_weight_conversion(pat_service: PATModelService) -> None:
        """Test that TensorFlow weights were converted correctly."""
        if pat_service.model is None:
            pytest.skip("Model not loaded")
        encoder = pat_service.model.encoder  # type: ignore[union-attr]

        # Check if weights are loaded (not random)
        first_layer = encoder.transformer_layers[0]
        attention = first_layer.attention

        # Check query projection weights
        first_query = attention.query_projections[0]  # type: ignore[union-attr,index]
        weight_std = first_query.weight.std().item()  # type: ignore[union-attr]

        # Real weights should have reasonable std (not too small/large)
        assert 0.01 < weight_std < 1.0, f"Weight std {weight_std} suggests random init"

        # Check patch embedding weights
        patch_weight_std = encoder.patch_embedding.weight.std().item()
        assert 0.01 < patch_weight_std < 1.0

    @staticmethod
    async def test_attention_mechanism(pat_service: PATModelService) -> None:
        """Test the custom PAT attention mechanism."""
        if pat_service.model is None:
            pytest.skip("Model not loaded")
        encoder = pat_service.model.encoder  # type: ignore[union-attr]
        attention = encoder.transformer_layers[0].attention

        assert isinstance(attention, PATMultiHeadAttention)
        assert attention.num_heads == 12
        assert attention.head_dim == 96
        assert attention.embed_dim == 96

        # Test attention forward pass
        batch_size, seq_len, embed_dim = 2, 560, 96
        x = torch.randn(batch_size, seq_len, embed_dim)

        output, attn_weights = attention(x, x, x)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

    @staticmethod
    async def test_inference_24h(
        pat_service: PATModelService, sample_actigraphy_data: list[ActigraphyDataPoint]
    ) -> None:
        """Test PAT inference with 24-hour data."""
        actigraphy_input = ActigraphyInput(
            user_id="test_user_24h",
            data_points=sample_actigraphy_data,
            sampling_rate=1.0,
            duration_hours=24,
        )

        analysis = await pat_service.analyze_actigraphy(actigraphy_input)

        # Validate analysis structure
        assert isinstance(analysis, ActigraphyAnalysis)
        assert analysis.user_id == "test_user_24h"

        # Validate metrics ranges
        assert 0 <= analysis.sleep_efficiency <= 100
        assert 0 <= analysis.sleep_onset_latency <= 300  # Max 5 hours
        assert 0 <= analysis.total_sleep_time <= 24
        assert 0 <= analysis.circadian_rhythm_score <= 1
        assert 0 <= analysis.depression_risk_score <= 1
        assert 0 <= analysis.confidence_score <= 1

        # Check clinical insights
        assert len(analysis.clinical_insights) > 0
        assert all(isinstance(insight, str) for insight in analysis.clinical_insights)

    @staticmethod
    async def test_inference_week(
        pat_service: PATModelService, week_actigraphy_data: list[ActigraphyDataPoint]
    ) -> None:
        """Test PAT inference with full week data."""
        actigraphy_input = ActigraphyInput(
            user_id="test_user_week",
            data_points=week_actigraphy_data,
            sampling_rate=1.0,
            duration_hours=168,
        )

        analysis = await pat_service.analyze_actigraphy(actigraphy_input)

        assert isinstance(analysis, ActigraphyAnalysis)
        assert analysis.user_id == "test_user_week"

        # Week-long data should provide more stable metrics
        assert 0 <= analysis.sleep_efficiency <= 100
        assert 0 <= analysis.circadian_rhythm_score <= 1

    @staticmethod
    async def test_clinical_insights_generation(pat_service: PATModelService) -> None:
        """Test clinical insights generation for different scenarios."""
        # Test high sleep efficiency scenario
        insights_high = pat_service._generate_clinical_insights(
            sleep_efficiency=90.0, circadian_score=0.9, depression_risk=0.2
        )

        assert any("excellent" in insight.lower() for insight in insights_high)
        assert any("strong" in insight.lower() for insight in insights_high)
        assert any("healthy" in insight.lower() for insight in insights_high)

        # Test poor sleep scenario
        insights_poor = pat_service._generate_clinical_insights(
            sleep_efficiency=60.0, circadian_score=0.3, depression_risk=0.8
        )

        assert any("poor" in insight.lower() for insight in insights_poor)
        assert any("irregular" in insight.lower() for insight in insights_poor)
        assert any("elevated" in insight.lower() for insight in insights_poor)

    @staticmethod
    async def test_preprocessing_pipeline(
        pat_service: PATModelService, sample_actigraphy_data: list[ActigraphyDataPoint]
    ) -> None:
        """Test the actigraphy preprocessing pipeline."""
        # Test normal length data (1440 = 24h)
        tensor = pat_service._preprocess_actigraphy_data(
            sample_actigraphy_data, target_length=1440
        )

        assert tensor.shape == (1440,)
        assert tensor.dtype == torch.float32

        # Test with different target length
        tensor_week = pat_service._preprocess_actigraphy_data(
            sample_actigraphy_data, target_length=10080
        )

        assert tensor_week.shape == (10080,)

    @staticmethod
    async def test_health_check(pat_service: PATModelService) -> None:
        """Test service health check."""
        health = await pat_service.health_check()

        assert health["service"] == "PAT Model Service"
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert "device" in health

    @staticmethod
    async def test_model_not_loaded_error() -> None:
        """Test error handling when model is not loaded."""
        service = PATModelService()

        sample_data = ActigraphyInput(
            user_id="test",
            data_points=[ActigraphyDataPoint(timestamp=datetime.now(UTC), value=1.0)],
            sampling_rate=1.0,
            duration_hours=24,
        )

        with pytest.raises(MLPredictionError) as excinfo:
            await service.analyze_actigraphy(sample_data)

        assert "PAT model not loaded" in str(excinfo.value)
        assert excinfo.value.model_name == "PAT"
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    @staticmethod
    async def test_singleton_service() -> None:
        """Test the global singleton PAT service."""
        service1 = await get_pat_service()
        service2 = await get_pat_service()

        # Should return the same instance
        assert service1 is service2

    @staticmethod
    async def test_different_model_sizes() -> None:
        """Test loading different PAT model sizes."""
        for model_size in ["small", "medium", "large"]:
            service = PATModelService(model_size=model_size)
            config = service.config

            assert config["embed_dim"] == 96  # All models use 96
            assert config["patch_size"] in {9, 18}  # Valid patch sizes
            assert config["num_heads"] in {6, 12}  # Valid head counts

            if model_size == "small":
                assert config["num_layers"] == 1
                assert config["num_heads"] == 6
            elif model_size == "medium":
                assert config["num_layers"] == 2
                assert config["num_heads"] == 12
            elif model_size == "large":
                assert config["num_layers"] == 4
                assert config["num_heads"] == 12


class TestPATArchitecture:
    """Test the PAT model architecture components."""

    @staticmethod
    def test_pat_configs() -> None:
        """Test PAT configuration validity."""
        for config in PAT_CONFIGS.values():
            assert "num_layers" in config
            assert "num_heads" in config
            assert "embed_dim" in config
            assert config["embed_dim"] == 96  # All models use 96
            assert config["ff_dim"] == 256
            assert config["patch_size"] in {9, 18}
            assert config["input_size"] == 10080

    @staticmethod
    def test_encoder_forward_pass() -> None:
        """Test PAT encoder forward pass."""
        encoder = PATEncoder(
            input_size=10080,
            patch_size=18,
            embed_dim=96,
            num_layers=2,
            num_heads=12,
            ff_dim=256,
        )

        batch_size = 2
        seq_len = 10080
        x = torch.randn(batch_size, seq_len)

        output = encoder(x)

        expected_patches = seq_len // 18  # 560 patches
        assert output.shape == (batch_size, expected_patches, 96)

    @staticmethod
    def test_classification_head() -> None:
        """Test the classification head."""
        encoder = PATEncoder(embed_dim=96, num_layers=1, num_heads=6)
        model = PATForMentalHealthClassification(encoder, num_classes=18)

        batch_size = 2
        seq_len = 10080
        x = torch.randn(batch_size, seq_len)

        outputs = model(x)

        assert "raw_logits" in outputs
        assert "sleep_metrics" in outputs
        assert "circadian_score" in outputs
        assert "depression_risk" in outputs
        assert "embeddings" in outputs

        assert outputs["raw_logits"].shape == (batch_size, 18)
        assert outputs["sleep_metrics"].shape == (batch_size, 8)
        assert outputs["circadian_score"].shape == (batch_size, 1)


class TestPATIntegration:
    """Integration tests for PAT service with other components."""

    @staticmethod
    async def test_end_to_end_pipeline() -> None:
        """Test complete end-to-end PAT analysis pipeline."""
        # Create service
        service = PATModelService(model_size="medium")
        await service.load_model()

        # Generate realistic data
        data_points = []
        base_time = datetime.now(UTC)

        for hour in range(24):
            for minute in range(60):
                timestamp = base_time + timedelta(hours=hour, minutes=minute)

                # Sleep pattern: low activity 10pm-6am, high activity 6am-10pm
                if hour >= 22 or hour <= 6:
                    activity = rng.exponential(2)  # Sleep
                else:
                    activity = rng.exponential(20)  # Awake

                data_points.append(
                    ActigraphyDataPoint(timestamp=timestamp, value=float(activity))
                )

        # Create input
        actigraphy_input = ActigraphyInput(
            user_id="integration_test",
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24,
        )

        # Run analysis
        analysis = await service.analyze_actigraphy(actigraphy_input)

        # Validate results
        assert analysis.user_id == "integration_test"
        assert len(analysis.clinical_insights) > 0
        assert analysis.confidence_score > 0

        # Check timestamp format
        parsed_time = datetime.fromisoformat(analysis.analysis_timestamp)
        assert isinstance(parsed_time, datetime)

    @staticmethod
    async def test_concurrent_analysis() -> None:
        """Test concurrent PAT analysis requests."""
        service = PATModelService(model_size="medium")
        await service.load_model()

        # Create multiple analysis tasks
        async def run_analysis(user_id: str) -> ActigraphyAnalysis:
            data_points = [
                ActigraphyDataPoint(
                    timestamp=datetime.now(UTC) + timedelta(minutes=i),
                    value=float(rng.exponential(10)),
                )
                for i in range(1440)
            ]

            actigraphy_input = ActigraphyInput(
                user_id=user_id,
                data_points=data_points,
                sampling_rate=1.0,
                duration_hours=24,
            )

            return await service.analyze_actigraphy(actigraphy_input)

        # Run multiple analyses concurrently
        tasks = [run_analysis(f"user_{i}") for i in range(3)]
        results = await asyncio.gather(*tasks)

        # Validate all results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.user_id == f"user_{i}"
            assert isinstance(result, ActigraphyAnalysis)


def run_quick_test() -> None:
    """Run a quick test to verify PAT functionality."""

    async def _async_quick_test() -> None:
        """Async wrapper for quick test."""
        logger.info("🧪 Running PAT Service Quick Test...")

        service = PATModelService(model_size="medium")
        await service.load_model()

        # Generate test data
        data_points = [
            ActigraphyDataPoint(
                timestamp=datetime.now(UTC) + timedelta(minutes=i),
                value=float(10 + 5 * np.sin(2 * np.pi * i / 1440)),
            )
            for i in range(1440)
        ]

        actigraphy_input = ActigraphyInput(
            user_id="quick_test",
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24,
        )

        analysis = await service.analyze_actigraphy(actigraphy_input)

        logger.info("✅ Analysis complete for user: %s", analysis.user_id)
        logger.info("   Sleep Efficiency: %.1f%%", analysis.sleep_efficiency)
        logger.info("   Confidence: %.3f", analysis.confidence_score)
        logger.info("   Insights: %d generated", len(analysis.clinical_insights))
        logger.info("🎯 PAT Service Test PASSED!")

    asyncio.run(_async_quick_test())


if __name__ == "__main__":
    run_quick_test()
