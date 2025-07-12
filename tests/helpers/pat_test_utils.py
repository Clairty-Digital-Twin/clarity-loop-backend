"""Test utilities for PAT service tests."""

from unittest.mock import patch, MagicMock, Mock
from contextlib import contextmanager
import os


@contextmanager
def mock_pat_integrity_check():
    """Mock PAT service integrity checks to always pass.
    
    This is needed because test environments don't have the actual
    model files that match the production checksums.
    """
    with patch("clarity.ml.pat_service.PATModelService._verify_model_integrity", return_value=True):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("clarity.ml.pat_service.PATModelService._load_pretrained_weights", return_value=True):
                # Also mock the actual model loading to prevent file I/O
                mock_model = MagicMock()
                mock_model.eval = MagicMock()
                mock_model.to = MagicMock(return_value=mock_model)
                
                with patch("clarity.ml.pat_service.PATForMentalHealthClassification", return_value=mock_model):
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
    
    with patch("clarity.ml.pat_service.PATModelService._verify_model_integrity", return_value=True):
        with patch("clarity.ml.pat_service.PATModelService._load_pretrained_weights", return_value=True):
            with patch("pathlib.Path.exists", return_value=True):
                
                # Mock the model itself
                mock_model = Mock()
                mock_model.eval = Mock(return_value=mock_model)
                mock_model.to = Mock(return_value=mock_model)
                
                # Mock model forward pass
                import torch
                mock_output = torch.randn(1, 18)  # 18 classes
                mock_model.return_value = mock_output
                
                with patch("clarity.ml.pat_service.PATForMentalHealthClassification", return_value=mock_model):
                    yield