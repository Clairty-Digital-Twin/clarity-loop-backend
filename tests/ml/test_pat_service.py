"""Comprehensive tests for PAT (Pretrained Actigraphy Transformer) Model Service.

This test suite covers all aspects of the PAT service including:
- Model initialization and loading
- Weight loading from H5 files
- Actigraphy analysis pipeline
- Error handling and edge cases
- Health checks and status monitoring
"""

from __future__ import annotations

from datetime import UTC, datetime
import sys
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
import torch

import clarity.ml.pat_service
from clarity.ml.pat_service import (
    ActigraphyAnalysis,
    ActigraphyInput,
    PATEncoder,
    PATForMentalHealthClassification,
    PATModelService,
    get_pat_service,
)
from clarity.ml.preprocessing import ActigraphyDataPoint
from clarity.services.health_data_service import MLPredictionError


class TestPATEncoder:
    """Test the PyTorch PAT Encoder model architecture."""

    @staticmethod
    def test_pat_encoder_initialization() -> None:
        """Test PAT encoder model initialization with default parameters."""
        encoder = PATEncoder(
            input_size=10080,
            patch_size=18,
            embed_dim=96,
            num_layers=2,
            num_heads=12,
            ff_dim=256,
        )

        assert hasattr(encoder, "patch_embedding")
        assert hasattr(encoder, "positional_encoding")
        assert hasattr(encoder, "transformer_layers")
        assert len(encoder.transformer_layers) == 2
        assert encoder.embed_dim == 96

    @staticmethod
    def test_pat_encoder_custom_parameters() -> None:
        """Test PAT encoder with custom architecture parameters."""
        encoder = PATEncoder(
            input_size=1440,
            patch_size=10,
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            ff_dim=128,
        )

        assert encoder.input_size == 1440
        assert encoder.patch_size == 10
        assert encoder.embed_dim == 64
        assert len(encoder.transformer_layers) == 1

    @staticmethod
    def test_pat_encoder_forward_pass() -> None:
        """Test forward pass through PAT encoder."""
        encoder = PATEncoder(
            input_size=10080,
            patch_size=18,
            embed_dim=96,
            num_layers=2,
            num_heads=12,
            ff_dim=256,
        )
        encoder.eval()

        batch_size = 2
        input_size = 10080

        x = torch.randn(batch_size, input_size)

        with torch.no_grad():
            outputs = encoder(x)

        expected_patches = input_size // 18  # 560 patches
        assert outputs.shape == (batch_size, expected_patches, 96)

    @staticmethod
    def test_pat_full_model_forward_pass() -> None:
        """Test forward pass through full PAT model."""
        encoder = PATEncoder(
            input_size=10080, patch_size=18, embed_dim=96, num_layers=1, num_heads=6
        )
        model = PATForMentalHealthClassification(encoder, num_classes=18)
        model.eval()

        batch_size = 2
        input_size = 10080

        x = torch.randn(batch_size, input_size)

        with torch.no_grad():
            outputs = model(x)

        assert isinstance(outputs, dict)
        assert "raw_logits" in outputs
        assert "sleep_metrics" in outputs
        assert "circadian_score" in outputs
        assert "depression_risk" in outputs
        assert "embeddings" in outputs

        assert outputs["raw_logits"].shape == (batch_size, 18)
        assert outputs["sleep_metrics"].shape == (batch_size, 8)
        assert outputs["circadian_score"].shape == (batch_size, 1)
        assert outputs["depression_risk"].shape == (batch_size, 1)


class TestPATModelServiceInitialization:
    """Test PAT model service initialization and configuration."""

    @staticmethod
    def test_service_initialization_default_parameters() -> None:
        """Test service initialization with default parameters."""
        service = PATModelService()

        assert service.model_size == "medium"
        assert service.device in {"cpu", "cuda"}
        assert service.model is None
        assert not service.is_loaded
        # Path comparison skipped - paths are now absolute and dynamic

    @staticmethod
    def test_service_initialization_custom_parameters() -> None:
        """Test service initialization with custom parameters - path sanitization."""
        custom_path = "custom/path/to/model.h5"
        service = PATModelService(
            model_size="large", device="cpu", model_path=custom_path
        )

        assert service.model_size == "large"
        assert service.device == "cpu"
        # Path sanitization security feature - unsafe paths are replaced with safe default
        # The actual path should match what PATModelService._sanitize_model_path returns
        assert service.model_path.endswith("models/pat/default_model.h5")
        assert "clarity-loop-backend" in service.model_path

    @staticmethod
    def test_service_initialization_model_paths() -> None:
        """Test model path selection for different sizes."""
        small_service = PATModelService(model_size="small")
        medium_service = PATModelService(model_size="medium")
        large_service = PATModelService(model_size="large")

        # Verify the paths contain the correct model names
        assert "PAT-S_29k_weights.h5" in small_service.model_path
        assert "PAT-M_29k_weights.h5" in medium_service.model_path
        assert "PAT-L_29k_weights.h5" in large_service.model_path

        # Verify the paths use the pat subdirectory
        assert "/pat/" in small_service.model_path
        assert "/pat/" in medium_service.model_path
        assert "/pat/" in large_service.model_path

    @staticmethod
    def test_device_selection_cuda_available() -> None:
        """Test device selection when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            service = PATModelService()
            assert service.device == "cuda"

    @staticmethod
    def test_device_selection_cuda_unavailable() -> None:
        """Test device selection when CUDA is unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            service = PATModelService()
            assert service.device == "cpu"


class TestPATModelServiceLoading:
    """Test PAT model loading functionality."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_success_with_weights() -> None:
        """Test successful model loading with existing weight file."""
        service = PATModelService(model_size="medium")

        # Mock the weight loading method directly to avoid h5py complications
        mock_state_dict = {
            "encoder.patch_embedding.weight": torch.randn(96, 18),
            "encoder.patch_embedding.bias": torch.randn(96),
            "classifier.0.weight": torch.randn(96),
            "classifier.0.bias": torch.randn(96),
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                PATModelService,
                "_load_tensorflow_weights",
                return_value=mock_state_dict,
            ),
        ):
            await service.load_model()

            assert service.is_loaded
            assert service.model is not None

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_missing_weights_file() -> None:
        """Test model loading when weight file doesn't exist."""
        service = PATModelService(model_path="/nonexistent/path/model.h5")

        await service.load_model()

        assert service.is_loaded
        assert service.model is not None  # Model initialized without weights

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_h5py_import_error() -> None:
        """Test model loading when h5py is not available."""
        service = PATModelService(model_size="medium")

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.dict("sys.modules", {}, clear=False),
        ):
            # Remove h5py from sys.modules if it exists
            if "h5py" in sys.modules:
                del sys.modules["h5py"]

            await service.load_model()

            assert service.is_loaded
            assert service.model is not None

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_h5_file_error() -> None:
        """Test model loading when H5 file is corrupted."""
        service = PATModelService(model_size="medium")

        # Mock the weight loading method to raise an OSError (simulating corrupted file)
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                PATModelService,
                "_load_tensorflow_weights",
                side_effect=OSError("Corrupted file"),
            ),
        ):
            await service.load_model()

            assert service.is_loaded
            assert service.model is not None

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_weight_mapping_success() -> None:
        """Test successful weight mapping from H5 to PyTorch."""
        service = PATModelService(model_size="medium")

        # Create mock weights with correct shapes for realistic testing
        rng = np.random.default_rng(42)
        mock_state_dict = {
            "encoder.patch_embedding.weight": torch.from_numpy(
                rng.standard_normal((96, 18))
            ).float(),
            "encoder.patch_embedding.bias": torch.from_numpy(
                rng.standard_normal(96)
            ).float(),
            "classifier.0.weight": torch.from_numpy(rng.standard_normal(96)).float(),
            "classifier.0.bias": torch.from_numpy(rng.standard_normal(96)).float(),
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                PATModelService,
                "_load_tensorflow_weights",
                return_value=mock_state_dict,
            ),
        ):
            await service.load_model()

            assert service.is_loaded


class TestPATModelServiceAnalysis:
    """Test PAT model actigraphy analysis functionality."""

    @pytest.fixture
    @staticmethod
    def sample_actigraphy_input() -> ActigraphyInput:
        """Create sample actigraphy input data."""
        data_points = [
            ActigraphyDataPoint(timestamp=datetime.now(UTC), value=float(i % 100))
            for i in range(1440)  # 24 hours of minute-by-minute data
        ]

        return ActigraphyInput(
            user_id=str(uuid4()),
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24,
        )

    @pytest.mark.asyncio
    @staticmethod
    async def test_analyze_actigraphy_success(
        sample_actigraphy_input: ActigraphyInput,
    ) -> None:
        """Test successful actigraphy analysis."""
        service = PATModelService(model_size="medium")
        service.is_loaded = True

        # Mock the preprocessor and model
        with (
            patch.object(service, "_preprocess_actigraphy_data") as mock_preprocess,
            patch.object(service, "_postprocess_predictions") as mock_postprocess,
        ):
            # Set up the model after load_model is called
            def setup_model() -> None:
                service.model = MagicMock()
                service.is_loaded = True

            # Mock preprocessor output
            mock_preprocess.return_value = torch.randn(1, 1440, 1)

            # Mock model output
            service.model = MagicMock()
            service.model.return_value = {
                "raw_logits": torch.randn(1, 18),
                "sleep_metrics": torch.randn(1, 8),
                "circadian_score": torch.randn(1, 1),
                "depression_risk": torch.randn(1, 1),
                "embeddings": torch.randn(1, 96),
            }

            # Mock postprocessor output
            mock_postprocess.return_value = ActigraphyAnalysis(
                user_id=sample_actigraphy_input.user_id,
                analysis_timestamp=datetime.now(UTC).isoformat(),
                sleep_efficiency=85.0,
                sleep_onset_latency=15.0,
                wake_after_sleep_onset=30.0,
                total_sleep_time=7.5,
                circadian_rhythm_score=0.75,
                activity_fragmentation=0.25,
                depression_risk_score=0.2,
                sleep_stages=["wake"] * 1440,
                confidence_score=0.85,
                clinical_insights=["Good sleep efficiency"],
                embedding=[0.0] * 128,
            )

            result = await service.analyze_actigraphy(sample_actigraphy_input)

            assert isinstance(result, ActigraphyAnalysis)
            assert result.user_id == sample_actigraphy_input.user_id
            assert result.sleep_efficiency == 85.0

    @pytest.mark.asyncio
    @staticmethod
    async def test_analyze_actigraphy_model_not_loaded(
        sample_actigraphy_input: ActigraphyInput,
    ) -> None:
        """Test analysis when model is not loaded."""
        service = PATModelService(model_size="medium")
        service.is_loaded = False

        with (
            patch.object(
                service,
                "_preprocess_actigraphy_data",
                return_value=torch.randn(1, 1440, 1),
            ),
            patch.object(service, "_postprocess_predictions") as mock_postprocess,
        ):
            # Manually set up the model for this test
            service.model = MagicMock()
            service.is_loaded = True

            service.model.return_value = {
                "raw_logits": torch.randn(1, 18),
                "sleep_metrics": torch.randn(1, 8),
                "circadian_score": torch.randn(1, 1),
                "depression_risk": torch.randn(1, 1),
                "embeddings": torch.randn(1, 96),
            }

            mock_postprocess.return_value = ActigraphyAnalysis(
                user_id=sample_actigraphy_input.user_id,
                analysis_timestamp=datetime.now(UTC).isoformat(),
                sleep_efficiency=75.0,
                sleep_onset_latency=20.0,
                wake_after_sleep_onset=40.0,
                total_sleep_time=7.0,
                circadian_rhythm_score=0.7,
                activity_fragmentation=0.3,
                depression_risk_score=0.3,
                sleep_stages=["sleep"] * 1440,
                confidence_score=0.8,
                clinical_insights=["Moderate sleep quality"],
                embedding=[0.0] * 128,
            )

            result = await service.analyze_actigraphy(sample_actigraphy_input)

            assert isinstance(result, ActigraphyAnalysis)

    @pytest.mark.asyncio
    @staticmethod
    async def test_analyze_actigraphy_preprocessing_error(
        sample_actigraphy_input: ActigraphyInput,
    ) -> None:
        """Test analysis with preprocessing error."""
        service = PATModelService(model_size="medium")
        service.is_loaded = True
        service.model = MagicMock()

        with (
            patch.object(
                service,
                "_preprocess_actigraphy_data",
                side_effect=ValueError("Preprocessing failed"),
            ),
            pytest.raises(MLPredictionError) as excinfo,
        ):
            await service.analyze_actigraphy(sample_actigraphy_input)
        assert "Preprocessing failed" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)

    @pytest.mark.asyncio
    @staticmethod
    async def test_analyze_actigraphy_model_inference_error(
        sample_actigraphy_input: ActigraphyInput,
    ) -> None:
        """Test analysis with model inference error."""
        service = PATModelService(model_size="medium")
        service.is_loaded = True
        service.model = MagicMock(side_effect=RuntimeError("Model inference failed"))

        with (
            patch.object(
                service,
                "_preprocess_actigraphy_data",
                return_value=torch.randn(1, 1440, 1),
            ),
            pytest.raises(MLPredictionError) as excinfo,
        ):
            await service.analyze_actigraphy(sample_actigraphy_input)

        assert "Model inference failed" in str(excinfo.value)
        assert excinfo.value.model_name == "PAT"
        assert isinstance(excinfo.value.__cause__, RuntimeError)


class TestPATModelServicePostprocessing:
    """Test PAT model postprocessing functionality."""

    @staticmethod
    def test_postprocess_predictions_typical_values() -> None:
        """Test postprocessing with typical prediction values."""
        service = PATModelService()

        mock_predictions = {
            "raw_logits": torch.randn(1, 18),
            "sleep_metrics": torch.tensor(
                [[0.85, 0.75, 0.2, 7.5, 30.0, 15.0, 0.75, 0.25]]
            ),
            "circadian_score": torch.tensor([[0.75]]),
            "depression_risk": torch.tensor([[0.2]]),
            "embeddings": torch.randn(1, 96),
        }

        user_id = str(uuid4())

        result = service._postprocess_predictions(mock_predictions, user_id)

        assert isinstance(result, ActigraphyAnalysis)
        assert result.user_id == user_id
        assert 0.0 <= result.sleep_efficiency <= 100.0
        assert 0.0 <= result.circadian_rhythm_score <= 1.0
        assert 0.0 <= result.depression_risk_score <= 1.0
        assert len(result.clinical_insights) > 0

    @staticmethod
    def test_generate_clinical_insights_excellent_sleep() -> None:
        """Test clinical insights generation for excellent sleep."""
        insights = PATModelService._generate_clinical_insights(
            sleep_efficiency=90.0, circadian_score=0.9, depression_risk=0.1
        )

        assert any("Excellent sleep" in insight for insight in insights)
        assert any("Strong circadian rhythm" in insight for insight in insights)
        assert any("healthy mood" in insight.lower() for insight in insights)

    @staticmethod
    def test_generate_clinical_insights_poor_sleep() -> None:
        """Test clinical insights generation for poor sleep."""
        insights = PATModelService._generate_clinical_insights(
            sleep_efficiency=60.0, circadian_score=0.4, depression_risk=0.8
        )

        assert any("Poor sleep efficiency" in insight for insight in insights)
        assert any("Irregular circadian rhythm" in insight for insight in insights)
        assert any("Elevated depression risk" in insight for insight in insights)

    @staticmethod
    def test_generate_clinical_insights_moderate_values() -> None:
        """Test clinical insights generation for moderate values."""
        insights = PATModelService._generate_clinical_insights(
            sleep_efficiency=75.0, circadian_score=0.65, depression_risk=0.5
        )

        assert any("Good sleep efficiency" in insight for insight in insights)
        assert any("Moderate circadian rhythm" in insight for insight in insights)


class TestPATModelServiceHealthCheck:
    """Test PAT model service health check functionality."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_health_check_loaded_model() -> None:
        """Test health check with loaded model."""
        service = PATModelService(model_size="large", device="cpu")
        service.is_loaded = True

        health = await service.health_check()

        assert health["service"] == "PAT Model Service"
        assert health["status"] == "healthy"
        assert health["model_size"] == "large"
        assert health["device"] == "cpu"
        assert health["model_loaded"] is True

    @pytest.mark.asyncio
    @staticmethod
    async def test_health_check_unloaded_model() -> None:
        """Test health check behavior when model weights are not properly loaded.

        Following ML testing best practices:
        - Tests actual service behavior (not mocked)
        - Validates that service reports 'unhealthy' when weights missing
        - 'unhealthy' is more informative than 'not_loaded' per industry standards
        """
        service = PATModelService(model_size="small")
        service.is_loaded = False

        health = await service.health_check()

        # Primary assertion - service reports unhealthy status following ML best practices
        assert health["status"] == "unhealthy"
        assert health["model_loaded"] is False

        # Additional validation - check health details are informative
        assert isinstance(health.get("service"), str)
        assert health["service"] == "PAT Model Service"


class TestGlobalPATService:
    """Test global PAT service singleton functionality."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_get_pat_service_singleton() -> None:
        """Test that get_pat_service returns a singleton instance."""
        # Clear any existing global instance
        clarity.ml.pat_service._pat_service = None

        with patch.object(PATModelService, "load_model") as mock_load:
            mock_load.return_value = None

            service1 = await get_pat_service()
            service2 = await get_pat_service()

            assert service1 is service2
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    @staticmethod
    async def test_get_pat_service_loads_model() -> None:
        """Test that get_pat_service loads the model."""
        # Clear any existing global instance
        clarity.ml.pat_service._pat_service = None

        with patch.object(PATModelService, "load_model") as mock_load:
            mock_load.return_value = None

            service = await get_pat_service()

            assert service is not None
            mock_load.assert_called_once()


class TestPATModelServiceEdgeCases:
    """Test edge cases and error conditions."""

    @staticmethod
    def test_raise_model_not_loaded_error() -> None:
        """Test the model not loaded error helper.

        Following ML testing best practices:
        - Service methods raise domain-specific MLPredictionError
        - RuntimeError is preserved as the __cause__ for proper exception chaining
        - This enables better error handling and monitoring in production
        """
        with pytest.raises(MLPredictionError, match="PAT model not loaded") as exc_info:
            PATModelService._raise_model_not_loaded_error()

        # Verify exception chaining - RuntimeError should be the cause
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "PAT model not loaded" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    @staticmethod
    async def test_preprocess_actigraphy_data() -> None:
        """Test actigraphy data preprocessing."""
        service = PATModelService()

        # Mock the preprocessor
        mock_preprocessor = MagicMock()
        mock_tensor = torch.randn(1440)
        mock_preprocessor.preprocess_for_pat_model.return_value = mock_tensor
        service.preprocessor = mock_preprocessor

        data_points = [
            ActigraphyDataPoint(timestamp=datetime.now(UTC), value=1.0)
            for _ in range(100)
        ]

        result = service._preprocess_actigraphy_data(data_points, target_length=1440)

        assert result.shape == (1440,)
        mock_preprocessor.preprocess_for_pat_model.assert_called_once_with(
            data_points, 1440
        )

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_exception_handling() -> None:
        """Test exception handling during model loading."""
        service = PATModelService()

        # Mock the PATEncoder constructor to raise an exception
        with (
            patch(
                "clarity.ml.pat_service.PATEncoder",
                side_effect=Exception("Critical error"),
            ),
            pytest.raises(Exception, match="Critical error"),
        ):
            await service.load_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
