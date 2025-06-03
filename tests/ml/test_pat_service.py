"""Comprehensive tests for PAT (Pretrained Actigraphy Transformer) Model Service.

This test suite covers all aspects of the PAT service including:
- Model initialization and loading
- Weight loading from H5 files
- Actigraphy analysis pipeline
- Error handling and edge cases
- Health checks and status monitoring
"""

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
    PATModelService,
    PATTransformer,
    get_pat_service,
)
from clarity.ml.preprocessing import ActigraphyDataPoint


class TestPATTransformer:
    """Test the PyTorch PAT Transformer model architecture."""

    @staticmethod
    def test_pat_transformer_initialization() -> None:
        """Test PAT transformer model initialization with default parameters."""
        model = PATTransformer()

        assert hasattr(model, "positional_encoding")
        assert hasattr(model, "transformer")
        assert hasattr(model, "sleep_stage_head")
        assert hasattr(model, "sleep_metrics_head")
        assert hasattr(model, "depression_head")
        assert isinstance(model.sleep_stage_head, torch.nn.Linear)

    @staticmethod
    def test_pat_transformer_custom_parameters() -> None:
        """Test PAT transformer with custom architecture parameters."""
        model = PATTransformer(
            input_dim=2,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            sequence_length=720,
            num_classes=3
        )

        assert model.input_dim == 2
        assert model.hidden_dim == 128
        assert model.sleep_stage_head.out_features == 3

    @staticmethod
    def test_pat_transformer_forward_pass() -> None:
        """Test forward pass through PAT transformer."""
        model = PATTransformer()
        model.eval()

        batch_size = 2
        seq_length = 1440
        input_features = 1

        x = torch.randn(batch_size, seq_length, input_features)

        with torch.no_grad():
            outputs = model(x)

        assert isinstance(outputs, dict)
        assert "sleep_stages" in outputs
        assert "sleep_metrics" in outputs
        assert "depression_risk" in outputs

        assert outputs["sleep_stages"].shape == (batch_size, seq_length, 4)
        assert outputs["sleep_metrics"].shape == (batch_size, 8)
        assert outputs["depression_risk"].shape == (batch_size, 1)

    @staticmethod
    def test_pat_transformer_different_sequence_lengths() -> None:
        """Test PAT transformer with different input sequence lengths."""
        model = PATTransformer()
        model.eval()

        # Only test sequence lengths within the positional encoding bounds
        for seq_len in [720, 1440]:  # 12h, 24h (removed 2880 which exceeds bounds)
            x = torch.randn(1, seq_len, 1)

            with torch.no_grad():
                outputs = model(x)

            assert outputs["sleep_stages"].shape == (1, seq_len, 4)
            assert outputs["sleep_metrics"].shape == (1, 8)


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
        assert service.model_path == "models/PAT-M_29k_weights.h5"

    @staticmethod
    def test_service_initialization_custom_parameters() -> None:
        """Test service initialization with custom parameters."""
        custom_path = "custom/path/to/model.h5"
        service = PATModelService(
            model_size="large",
            device="cpu",
            model_path=custom_path
        )

        assert service.model_size == "large"
        assert service.device == "cpu"
        assert service.model_path == custom_path

    @staticmethod
    def test_service_initialization_model_paths() -> None:
        """Test model path selection for different sizes."""
        small_service = PATModelService(model_size="small")
        medium_service = PATModelService(model_size="medium")
        large_service = PATModelService(model_size="large")

        assert small_service.model_path == "models/PAT-S_29k_weights.h5"
        assert medium_service.model_path == "models/PAT-M_29k_weights.h5"
        assert large_service.model_path == "models/PAT-L_29k_weights.h5"

    @staticmethod
    def test_device_selection_cuda_available() -> None:
        """Test device selection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            service = PATModelService()
            assert service.device == "cuda"

    @staticmethod
    def test_device_selection_cuda_unavailable() -> None:
        """Test device selection when CUDA is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            service = PATModelService()
            assert service.device == "cpu"


class TestPATModelServiceLoading:
    """Test PAT model loading functionality."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_success_with_weights() -> None:
        """Test successful model loading with existing weight file."""
        service = PATModelService(model_size="medium")

        with patch('pathlib.Path.exists', return_value=True):
            # Create a more realistic h5py mock
            mock_h5py = MagicMock()
            mock_file = MagicMock()
            
            # Set up context manager behavior
            mock_h5py.File.return_value.__enter__.return_value = mock_file
            mock_h5py.File.return_value.__exit__.return_value = None
            
            # Mock keys() to return a simple list (not a KeysView)
            mock_file.keys.return_value = ['inputs', 'dense']
            
            # Mock nested access for weight groups
            mock_inputs = MagicMock()
            mock_inputs.__contains__ = lambda x: x in ['kernel:0', 'bias:0']
            mock_inputs.__getitem__ = lambda x: {
                'kernel:0': np.ones((1, 256)),
                'bias:0': np.ones(256)
            }[x]
            
            mock_dense = MagicMock()
            mock_dense.__contains__ = lambda x: x in ['kernel:0', 'bias:0']
            mock_dense.__getitem__ = lambda x: {
                'kernel:0': np.ones((256, 4)),
                'bias:0': np.ones(4)
            }[x]
            
            mock_file.__getitem__ = lambda x: {
                'inputs': mock_inputs,
                'dense': mock_dense
            }[x]
            mock_file.__contains__ = lambda x: x in ['inputs', 'dense']

            with patch.dict('sys.modules', {'h5py': mock_h5py}):
                await service.load_model()

                assert service.is_loaded
                assert service.model is not None

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_missing_weights_file() -> None:
        """Test model loading when weight file doesn't exist."""
        service = PATModelService(model_path="nonexistent/path.h5")

        await service.load_model()

        assert service.is_loaded
        assert service.model is not None  # Model initialized without weights

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_h5py_import_error() -> None:
        """Test model loading when h5py is not available."""
        service = PATModelService(model_size="medium")

        with patch('pathlib.Path.exists', return_value=True):
            # Simulate h5py not being available by not adding it to sys.modules
            with patch.dict('sys.modules', {}, clear=False):
                # Remove h5py from sys.modules if it exists
                if 'h5py' in sys.modules:
                    del sys.modules['h5py']

                await service.load_model()

                assert service.is_loaded
                assert service.model is not None

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_h5_file_error() -> None:
        """Test model loading when H5 file is corrupted."""
        service = PATModelService(model_size="medium")

        with patch('pathlib.Path.exists', return_value=True):
            # Mock h5py to raise an exception without trying to import the real h5py
            mock_h5py = MagicMock()
            mock_h5py.File.side_effect = Exception("Corrupted file")

            with patch.dict('sys.modules', {'h5py': mock_h5py}):
                await service.load_model()

                assert service.is_loaded
                assert service.model is not None

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_weight_mapping_success() -> None:
        """Test successful weight mapping from H5 to PyTorch."""
        service = PATModelService(model_size="medium")

        # Create mock weights with correct shapes
        rng = np.random.default_rng(42)
        mock_input_weight = rng.standard_normal((1, 256))  # (input_dim, hidden_dim)
        mock_input_bias = rng.standard_normal(256)

        with patch('pathlib.Path.exists', return_value=True):
            mock_h5py = MagicMock()
            mock_file = MagicMock()
            mock_h5py.File.return_value.__enter__.return_value = mock_file
            mock_file.keys.return_value = ['input_layer']
            mock_file.__getitem__.return_value = {
                'weight': mock_input_weight,
                'bias': mock_input_bias
            }

            with patch.dict('sys.modules', {'h5py': mock_h5py}):
                await service.load_model()

                assert service.is_loaded


class TestPATModelServiceAnalysis:
    """Test PAT model actigraphy analysis functionality."""

    @pytest.fixture
    @staticmethod
    def sample_actigraphy_input() -> ActigraphyInput:
        """Create sample actigraphy input data."""
        data_points = [
            ActigraphyDataPoint(
                timestamp=datetime.now(UTC),
                value=float(i % 100)
            )
            for i in range(1440)  # 24 hours of minute-by-minute data
        ]

        return ActigraphyInput(
            user_id=str(uuid4()),
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24
        )

    @pytest.mark.asyncio
    @staticmethod
    async def test_analyze_actigraphy_success(sample_actigraphy_input: ActigraphyInput) -> None:
        """Test successful actigraphy analysis."""
        service = PATModelService(model_size="medium")
        service.is_loaded = True

        # Mock the preprocessor and model
        with patch.object(service, '_preprocess_actigraphy_data') as mock_preprocess, \
             patch.object(service, '_postprocess_predictions') as mock_postprocess:

            # Set up the model after load_model is called
            def setup_model() -> None:
                service.model = MagicMock()
                service.is_loaded = True

            mock_load = setup_model

            # Mock preprocessor output
            mock_preprocess.return_value = torch.randn(1, 1440, 1)

            # Mock model output
            service.model = MagicMock()
            service.model.return_value = {
                "sleep_stages": torch.randn(1, 1440, 4),
                "sleep_metrics": torch.randn(1, 8),
                "depression_risk": torch.randn(1, 1)
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
                clinical_insights=["Good sleep efficiency"]
            )

            result = await service.analyze_actigraphy(sample_actigraphy_input)

            assert isinstance(result, ActigraphyAnalysis)
            assert result.user_id == sample_actigraphy_input.user_id
            assert result.sleep_efficiency == 85.0

    @pytest.mark.asyncio
    @staticmethod
    async def test_analyze_actigraphy_model_not_loaded(sample_actigraphy_input: ActigraphyInput) -> None:
        """Test analysis when model is not loaded."""
        service = PATModelService(model_size="medium")
        service.is_loaded = False

        with patch.object(service, 'load_model') as mock_load_model:

            # Set up the model after load_model is called
            def setup_model() -> None:
                service.model = MagicMock()
                service.is_loaded = True

            mock_load_model.side_effect = setup_model

            with patch.object(service, '_preprocess_actigraphy_data', return_value=torch.randn(1, 1440, 1)), \
                 patch.object(service, '_postprocess_predictions') as mock_postprocess:

                service.model = MagicMock()
                service.model.return_value = {
                    "sleep_stages": torch.randn(1, 1440, 4),
                    "sleep_metrics": torch.randn(1, 8),
                    "depression_risk": torch.randn(1, 1)
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
                    clinical_insights=["Moderate sleep quality"]
                )

                result = await service.analyze_actigraphy(sample_actigraphy_input)

                mock_load_model.assert_called_once()
                assert isinstance(result, ActigraphyAnalysis)

    @pytest.mark.asyncio
    @staticmethod
    async def test_analyze_actigraphy_preprocessing_error(sample_actigraphy_input: ActigraphyInput) -> None:
        """Test analysis with preprocessing error."""
        service = PATModelService(model_size="medium")
        service.is_loaded = True
        service.model = MagicMock()

        with patch.object(service, '_preprocess_actigraphy_data', side_effect=ValueError("Preprocessing failed")), \
             pytest.raises(ValueError, match="Preprocessing failed"):
            await service.analyze_actigraphy(sample_actigraphy_input)

    @pytest.mark.asyncio
    @staticmethod
    async def test_analyze_actigraphy_model_inference_error(sample_actigraphy_input: ActigraphyInput) -> None:
        """Test analysis with model inference error."""
        service = PATModelService(model_size="medium")
        service.is_loaded = True
        service.model = MagicMock()
        service.model.side_effect = RuntimeError("Model inference failed")

        with patch.object(service, '_preprocess_actigraphy_data', return_value=torch.randn(1, 1440, 1)), \
             pytest.raises(RuntimeError, match="Model inference failed"):
            await service.analyze_actigraphy(sample_actigraphy_input)


class TestPATModelServicePostprocessing:
    """Test PAT model postprocessing functionality."""

    @staticmethod
    def test_postprocess_predictions_typical_values() -> None:
        """Test postprocessing with typical prediction values."""
        service = PATModelService()

        mock_predictions = {
            "sleep_stages": torch.randn(1, 1440, 4),
            "sleep_metrics": torch.tensor([[0.85, 0.75, 0.2, 7.5, 30.0, 15.0, 0.75, 0.25]]),
            "circadian_score": torch.tensor([[0.75]]),  # Added missing key
            "depression_risk": torch.tensor([[0.2]])
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
            sleep_efficiency=90.0,
            circadian_score=0.9,
            depression_risk=0.1
        )

        assert any("Excellent sleep" in insight for insight in insights)
        assert any("Strong circadian rhythm" in insight for insight in insights)
        assert any("Low depression risk" in insight for insight in insights)

    @staticmethod
    def test_generate_clinical_insights_poor_sleep() -> None:
        """Test clinical insights generation for poor sleep."""
        insights = PATModelService._generate_clinical_insights(
            sleep_efficiency=60.0,
            circadian_score=0.4,
            depression_risk=0.8
        )

        assert any("Poor sleep efficiency" in insight for insight in insights)
        assert any("Irregular circadian rhythm" in insight for insight in insights)
        assert any("Elevated depression risk" in insight for insight in insights)

    @staticmethod
    def test_generate_clinical_insights_moderate_values() -> None:
        """Test clinical insights generation for moderate values."""
        insights = PATModelService._generate_clinical_insights(
            sleep_efficiency=75.0,
            circadian_score=0.65,
            depression_risk=0.5
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
        """Test health check with unloaded model."""
        service = PATModelService(model_size="small")
        service.is_loaded = False

        health = await service.health_check()

        assert health["status"] == "not_loaded"
        assert health["model_loaded"] is False


class TestGlobalPATService:
    """Test global PAT service singleton functionality."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_get_pat_service_singleton() -> None:
        """Test that get_pat_service returns a singleton instance."""
        # Clear any existing global instance
        clarity.ml.pat_service._pat_service = None

        with patch.object(PATModelService, 'load_model') as mock_load:
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

        with patch.object(PATModelService, 'load_model') as mock_load:
            mock_load.return_value = None

            service = await get_pat_service()

            assert service is not None
            mock_load.assert_called_once()


class TestPATModelServiceEdgeCases:
    """Test edge cases and error conditions."""

    @staticmethod
    def test_raise_model_not_loaded_error() -> None:
        """Test the model not loaded error helper."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            PATModelService._raise_model_not_loaded_error()

    @pytest.mark.asyncio
    @staticmethod
    async def test_preprocess_actigraphy_data() -> None:
        """Test actigraphy data preprocessing."""
        service = PATModelService()

        # Mock the preprocessor
        mock_preprocessor = MagicMock()
        mock_tensor = torch.randn(1, 1440, 1)
        mock_preprocessor.preprocess_for_pat_model.return_value = mock_tensor
        service.preprocessor = mock_preprocessor

        data_points = [
            ActigraphyDataPoint(timestamp=datetime.now(UTC), value=1.0)
            for _ in range(100)
        ]

        result = service._preprocess_actigraphy_data(data_points, target_length=1440)

        assert result.shape == (1, 1440, 1)
        mock_preprocessor.preprocess_for_pat_model.assert_called_once_with(data_points, 1440)

    @pytest.mark.asyncio
    @staticmethod
    async def test_load_model_exception_handling() -> None:
        """Test exception handling during model loading."""
        service = PATModelService()

        # Mock the PATTransformer constructor to raise an exception
        with patch('clarity.ml.pat_service.PATTransformer', side_effect=Exception("Critical error")), \
             pytest.raises(Exception, match="Critical error"):
            await service.load_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
