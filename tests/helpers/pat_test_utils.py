"""Test utilities for PAT service tests."""

from contextlib import contextmanager
from unittest.mock import MagicMock, Mock, patch


@contextmanager
def mock_pat_integrity_check():
    """Mock PAT service integrity checks to always pass.

    This is needed because test environments don't have the actual
    model files that match the production checksums.
    """
    with patch(
        "clarity.ml.pat_service.PATModelService._verify_model_integrity",
        return_value=True,
    ), patch("pathlib.Path.exists", return_value=True), patch(
        "clarity.ml.pat_service.PATModelService._load_pretrained_weights",
        return_value=True,
    ):
        # Also mock the actual model loading to prevent file I/O
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        with patch(
            "clarity.ml.pat_service.PATForMentalHealthClassification",
            return_value=mock_model,
        ):
            yield mock_model


@contextmanager
def mock_pat_service_for_testing():
    """Comprehensive mock for PAT service in integration tests.

    This provides a fully functional mock PAT service that:
    - Bypasses integrity checks
    - Doesn't require actual model files
    - Returns realistic predictions
    - Is singleton-safe
    """
    # Clear any existing singleton
    import clarity.ml.pat_service

    clarity.ml.pat_service._pat_service = None

    with patch(
        "clarity.ml.pat_service.PATModelService._verify_model_integrity",
        return_value=True,
    ), patch(
        "clarity.ml.pat_service.PATModelService._load_pretrained_weights",
        return_value=True,
    ), patch("pathlib.Path.exists", return_value=True):

        # Mock the model itself
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.to = Mock(return_value=mock_model)

        # Mock model forward pass to return proper output dictionary
        import numpy as np
        import torch

        # Create a mock that returns the expected output structure
        def mock_forward(*args, **kwargs):
            # PAT model returns a dictionary with specific keys
            return {
                "sleep_metrics": torch.tensor(
                    np.random.rand(1, 10).astype(np.float32)
                ),  # 10 sleep metrics
                "circadian_score": torch.tensor(
                    np.random.rand(1, 1).astype(np.float32)
                ),
                "depression_risk": torch.tensor(
                    np.random.rand(1, 1).astype(np.float32)
                ),
                "embeddings": torch.tensor(
                    np.random.rand(1, 96).astype(np.float32)
                ),  # 96-dim embeddings
            }

        mock_model.side_effect = mock_forward

        with patch(
            "clarity.ml.pat_service.PATForMentalHealthClassification",
            return_value=mock_model,
        ):
            yield
