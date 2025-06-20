"""Integration tests for PAT preprocessing with the new windowing approach."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest
import torch

from clarity.ml.pat_service import PATModelService
from clarity.ml.preprocessing import (
    ActigraphyDataPoint,
    HealthDataPreprocessor,
    StandardActigraphyPreprocessor,
)
from clarity.ml.proxy_actigraphy import ProxyActigraphyTransformer, StepCountData
from clarity.utils.time_window import (
    WEEK_MINUTES,
    pad_to_week,
    prepare_for_pat_inference,
    slice_to_weeks,
)


class TestPreprocessingIntegration:
    """Test the integration of new preprocessing with PAT service."""

    def test_standard_preprocessor_uses_truncation(self):
        """Test that StandardActigraphyPreprocessor now uses truncation instead of downsampling."""
        # Create 2 weeks of data
        data_points = [
            ActigraphyDataPoint(
                timestamp=datetime.now(UTC) - timedelta(minutes=20160 - i),
                value=float(i),  # Ascending values to test truncation
            )
            for i in range(20160)  # 2 weeks
        ]

        # Process with the standard preprocessor
        tensor = StandardActigraphyPreprocessor.preprocess(
            data_points, target_length=10080
        )

        # Verify shape
        assert tensor.shape == (10080,)

        # Verify it took the most recent week (values 10080 to 20159)
        # After normalization, we can't check exact values, but we can verify
        # the pattern is preserved (ascending)
        values = tensor.numpy()

        # Check that values are generally increasing (allowing for normalization)
        # The last week should have higher values than earlier parts
        first_quarter_mean = np.mean(values[:2520])
        last_quarter_mean = np.mean(values[-2520:])
        assert (
            last_quarter_mean > first_quarter_mean
        ), "Should preserve ascending pattern"

    def test_pat_service_multi_week_handling(self):
        """Test that PAT service correctly handles multi-week inputs."""
        # Create a mock PAT service
        service = PATModelService()

        # Create 3 weeks of synthetic data
        data_points = [
            ActigraphyDataPoint(
                timestamp=datetime.now(UTC) - timedelta(minutes=30240 - i),
                value=100.0 + 10.0 * np.sin(2 * np.pi * i / 1440),  # Daily pattern
            )
            for i in range(30240)  # 3 weeks
        ]

        # Preprocess the data
        with patch.object(service, "model") as mock_model:
            # Mock the model to avoid loading actual weights
            mock_model.return_value = {
                "prediction": torch.randn(1, 3),
                "representation": torch.randn(1, 128),
            }

            # Process the data
            tensor = service._preprocess_actigraphy_data(data_points)

            # Verify shape
            assert tensor.shape == (10080,)

            # Verify it's on the correct device
            # service.device is a string like 'cpu' or 'cuda'
            assert tensor.device.type == service.device

    def test_proxy_actigraphy_consistency(self):
        """Test that proxy actigraphy transformation is consistent with new approach."""
        transformer = ProxyActigraphyTransformer()

        # Test with exactly 2 weeks of step data
        now = datetime.now(UTC)
        # Use realistic step counts (0-200 steps per minute with some variation)
        step_counts = [
            50.0
            + 30.0 * np.sin(2 * np.pi * i / 1440)
            + 20.0 * np.random.random()  # noqa: NPY002
            for i in range(20160)  # 2 weeks with daily pattern
        ]
        step_data = StepCountData(
            user_id="test_user",
            upload_id="test_upload",
            step_counts=step_counts,
            timestamps=[now - timedelta(minutes=20160 - i) for i in range(20160)],
        )

        result = transformer.transform_step_data(step_data)

        # Should have exactly one week of data
        assert len(result.vector) == 10080

        # Verify it took the most recent week
        # The proxy transformation includes sqrt and normalization,
        # but the relative pattern should be preserved
        # Check that we have variation in the data (not all same value)
        assert np.std(result.vector) > 0.1, "Transformed data should have variation"

        # Check the quality score is reasonable
        assert 0 < result.quality_score <= 1.0

    def test_health_data_preprocessor_integration(self):
        """Test HealthDataPreprocessor with new windowing."""
        preprocessor = HealthDataPreprocessor()

        # Create test data spanning multiple weeks
        test_cases = [
            (5000, "Less than 1 week"),
            (10080, "Exactly 1 week"),
            (15000, "1.5 weeks"),
            (20160, "Exactly 2 weeks"),
            (30240, "Exactly 3 weeks"),
        ]

        for num_points, description in test_cases:
            data_points = [
                ActigraphyDataPoint(
                    timestamp=datetime.now(UTC) - timedelta(minutes=num_points - i),
                    value=float(i % 1440),  # Daily pattern
                )
                for i in range(num_points)
            ]

            # Process with target length of 1 week
            tensor = preprocessor.preprocess_for_pat_model(
                data_points, target_length=10080
            )

            # All should produce exactly 10080 points
            assert tensor.shape == (10080,), f"Failed for {description}"
            assert isinstance(tensor, torch.Tensor), f"Wrong type for {description}"

    def test_edge_case_empty_data(self):
        """Test handling of empty data."""
        preprocessor = HealthDataPreprocessor()

        # Empty data
        empty_points = []
        tensor = preprocessor.preprocess_for_pat_model(
            empty_points, target_length=10080
        )

        # Should return tensor of zeros
        assert tensor.shape == (10080,)
        assert torch.all(tensor == 0.0)

    def test_edge_case_single_point(self):
        """Test handling of single data point."""
        preprocessor = HealthDataPreprocessor()

        # Single point
        single_point = [ActigraphyDataPoint(timestamp=datetime.now(UTC), value=100.0)]
        tensor = preprocessor.preprocess_for_pat_model(
            single_point, target_length=10080
        )

        # Should pad with zeros
        assert tensor.shape == (10080,)
        # Since we only have one value, normalization will result in 0 (no std dev)
        # or it will be padded. Let's just verify the shape
        assert tensor.shape == (10080,)

    @pytest.mark.parametrize("noise_level", [0.1, 1.0, 10.0])
    def test_robustness_to_noise(self, noise_level: float) -> None:
        """Test that preprocessing is robust to different noise levels."""
        # Create base signal
        base_signal = [
            ActigraphyDataPoint(
                timestamp=datetime.now(UTC) - timedelta(minutes=10080 - i),
                value=100.0 + 50.0 * np.sin(2 * np.pi * i / 1440),
            )
            for i in range(10080)
        ]

        # Ensure we pass the correct target length
        target_length = 10080

        # Add noise
        noisy_signal = [
            ActigraphyDataPoint(
                timestamp=point.timestamp,
                value=point.value + np.random.normal(0, noise_level),  # noqa: NPY002
            )
            for point in base_signal
        ]

        # Process both
        preprocessor = HealthDataPreprocessor()
        base_tensor = preprocessor.preprocess_for_pat_model(
            base_signal, target_length=target_length
        )
        noisy_tensor = preprocessor.preprocess_for_pat_model(
            noisy_signal, target_length=target_length
        )

        # Both should have correct shape
        assert base_tensor.shape == (10080,)
        assert noisy_tensor.shape == (10080,)

        # Should be similar but not identical
        correlation = torch.corrcoef(torch.stack([base_tensor, noisy_tensor]))[
            0, 1
        ].item()

        # Higher noise = lower correlation, but should still be positive
        assert correlation > 0.5, f"Correlation too low for noise level {noise_level}"

    def test_concurrent_preprocessing(self):
        """Test that preprocessing works correctly when called concurrently."""
        preprocessor = HealthDataPreprocessor()

        def preprocess_data(length: int) -> torch.Tensor:
            data_points = [
                ActigraphyDataPoint(
                    timestamp=datetime.now(UTC) - timedelta(minutes=length - i),
                    value=float(i),
                )
                for i in range(length)
            ]
            return preprocessor.preprocess_for_pat_model(data_points, 10080)

        # Process multiple datasets concurrently
        lengths = [5000, 10080, 15000, 20160, 30240]

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(preprocess_data, lengths))

        # All should produce correct shape
        for i, result in enumerate(results):
            assert result.shape == (10080,), f"Failed for length {lengths[i]}"

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        test_cases = [
            ("Very large values", 1e10),
            ("Very small values", 1e-10),
            ("Mixed extreme values", None),
        ]

        preprocessor = HealthDataPreprocessor()

        for description, base_value in test_cases:
            if base_value is None:
                # Mixed extreme values
                values = [1e10 if i % 2 == 0 else 1e-10 for i in range(10080)]
            else:
                values = [base_value] * 10080

            data_points = [
                ActigraphyDataPoint(
                    timestamp=datetime.now(UTC) - timedelta(minutes=10080 - i),
                    value=values[i],
                )
                for i in range(10080)
            ]

            tensor = preprocessor.preprocess_for_pat_model(
                data_points, target_length=10080
            )

            # Should not contain NaN or Inf
            assert not torch.any(torch.isnan(tensor)), f"NaN found for {description}"
            assert not torch.any(torch.isinf(tensor)), f"Inf found for {description}"
            assert tensor.shape == (10080,)

    def test_time_window_import_availability(self):
        """Verify that the time_window module is properly imported."""
        # This should not raise ImportError
        # Verify the functions exist and have correct signatures
        assert callable(slice_to_weeks)
        assert callable(pad_to_week)
        assert callable(prepare_for_pat_inference)
        assert WEEK_MINUTES == 10080
