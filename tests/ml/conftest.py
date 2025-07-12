"""ML test fixtures and configuration."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from clarity.ml.pat_service import PATModelService


@pytest.fixture
def mock_pat_service():
    """Mock PAT service that bypasses model loading and integrity checks."""
    with patch(
        "clarity.ml.pat_service.PATModelService._verify_model_integrity",
        return_value=True,
    ), patch(
        "clarity.ml.pat_service.PATModelService._load_pretrained_weights",
        return_value=True,
    ):
        service = PATModelService(model_size="medium")

        # Mock the model
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)

        # Mock prediction output
        mock_output = MagicMock()
        mock_output.squeeze.return_value.cpu.return_value.numpy.return_value = [
            0.1
        ] * 18
        mock_model.return_value = mock_output

        service.model = mock_model
        service.is_loaded = True

        yield service


@pytest.fixture(autouse=True)
def mock_pat_integrity_for_integration_tests(request):
    """Automatically mock PAT integrity checks for integration tests that don't test security."""
    # Only apply to specific test files that need it
    test_files_to_mock = [
        "test_analysis_pipeline_mania_integration.py",
        "test_pat_service_production.py",
    ]

    current_file = request.node.parent.name
    if any(filename in current_file for filename in test_files_to_mock):
        # Skip if the test specifically tests integrity
        if "integrity" in request.node.name or "security" in request.node.name:
            yield
        else:
            with patch(
                "clarity.ml.pat_service.PATModelService._verify_model_integrity",
                return_value=True,
            ):
                yield
    else:
        yield
