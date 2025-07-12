"""Test utilities for PAT service tests."""

from unittest.mock import patch, MagicMock
from contextlib import contextmanager


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