"""Tests for Proxy Actigraphy Transformation Module.

This module tests the conversion of Apple HealthKit step count data
into proxy actigraphy signals for PAT model analysis.
"""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
from pydantic_core import ValidationError
import pytest

from clarity.ml.proxy_actigraphy import (
    DEFAULT_NHANES_STATS,
    MAX_REALISTIC_STEPS_PER_MINUTE,
    MINUTES_PER_DAY,
    MINUTES_PER_WEEK,
    ProxyActigraphyResult,
    ProxyActigraphyTransformer,
    StepCountData,
    create_proxy_actigraphy_transformer,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestStepCountData:
    """Test StepCountData model."""

    def test_step_count_data_creation(self) -> None:
        """Test creating a valid StepCountData."""
        timestamps = [datetime.now(UTC) for _ in range(3)]
        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=[100.0, 150.0, 80.0],
            timestamps=timestamps,
            user_metadata={"age": 25, "sex": "male"},
        )

        assert step_data.user_id == "user123"
        assert step_data.upload_id == "upload456"
        assert len(step_data.step_counts) == 3
        assert len(step_data.timestamps) == 3
        assert step_data.user_metadata["age"] == 25

    def test_step_count_data_validation(self) -> None:
        """Test StepCountData validation."""
        timestamps = [datetime.now(UTC)]

        # Valid data
        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=[100.0],
            timestamps=timestamps,
        )
        assert len(step_data.step_counts) == 1

    def test_step_count_data_empty_list(self) -> None:
        """Test StepCountData with empty step counts."""
        with pytest.raises(ValidationError, match="List should have at least 1 item"):
            StepCountData(
                user_id="user123",
                upload_id="upload456",
                step_counts=[],  # Empty list should fail validation
                timestamps=[],
            )


class TestProxyActigraphyResult:
    """Test ProxyActigraphyResult model."""

    def test_result_creation(self) -> None:
        """Test creating a valid ProxyActigraphyResult."""
        result = ProxyActigraphyResult(
            user_id="user123",
            upload_id="upload456",
            vector=[1.0, 2.0, 3.0],
            quality_score=0.85,
            transformation_stats={
                "input_length": 3,
                "output_length": 3,
                "mean_steps_per_min": 100.0,
            },
            nhanes_reference={"year": 2025, "mean": 1.2, "std": 0.8},
        )

        assert result.user_id == "user123"
        assert result.upload_id == "upload456"
        assert len(result.vector) == 3
        assert result.quality_score == 0.85
        assert result.transformation_stats["input_length"] == 3
        assert result.nhanes_reference["year"] == 2025

    def test_result_quality_score_validation(self) -> None:
        """Test ProxyActigraphyResult quality score validation."""
        # Valid quality score
        result = ProxyActigraphyResult(
            user_id="user123",
            upload_id="upload456",
            vector=[1.0],
            quality_score=0.5,
            transformation_stats={},
            nhanes_reference={},
        )
        assert result.quality_score == 0.5


class TestProxyActigraphyTransformer:
    """Test ProxyActigraphyTransformer functionality."""

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_transformer_initialization(self, mock_lookup_stats: MagicMock) -> None:
        """Test transformer initialization."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        transformer = ProxyActigraphyTransformer()
        assert transformer is not None
        assert transformer.reference_year == 2025
        assert transformer.cache_enabled is True
        assert transformer.nhanes_mean == 1.2
        assert transformer.nhanes_std == 0.8

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_transformer_initialization_with_params(
        self, mock_lookup_stats: MagicMock
    ) -> None:
        """Test transformer initialization with custom parameters."""
        mock_lookup_stats.return_value = (1.15, 0.75)

        transformer = ProxyActigraphyTransformer(
            reference_year=2024, cache_enabled=False
        )
        assert transformer.reference_year == 2024
        assert transformer.cache_enabled is False
        assert transformer.nhanes_mean == 1.15
        assert transformer.nhanes_std == 0.75

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_steps_to_movement_proxy(self, mock_lookup_stats: MagicMock) -> None:
        """Test step count to movement proxy conversion."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        transformer = ProxyActigraphyTransformer()

        # Test with simple step data
        steps = np.array([100.0, 200.0, 50.0, 0.0])
        proxy_values = transformer.steps_to_movement_proxy(steps)

        assert isinstance(proxy_values, np.ndarray)
        assert len(proxy_values) == len(steps)
        assert all(isinstance(val, (int, float, np.number)) for val in proxy_values)

        # Values should be clipped to reasonable range
        assert all(-5.0 <= val <= 5.0 for val in proxy_values)

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_transform_step_data_basic(self, mock_lookup_stats: MagicMock) -> None:
        """Test basic step data transformation."""
        mock_lookup_stats.return_value = (3.2, 1.8)  # Updated to match new NHANES stats

        transformer = ProxyActigraphyTransformer()

        # Create sample step count data
        timestamps = [datetime.now(UTC) for _ in range(3)]
        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=[100.0, 150.0, 80.0],
            timestamps=timestamps,
        )

        result = transformer.transform_step_data(step_data)

        assert isinstance(result, ProxyActigraphyResult)
        assert result.user_id == "user123"
        assert result.upload_id == "upload456"

        # With automatic padding to full week, expect 10,080 values
        week_size = 10080  # MINUTES_PER_WEEK
        assert len(result.vector) == week_size
        assert 0 <= result.quality_score <= 1
        assert "input_length" in result.transformation_stats
        assert "output_length" in result.transformation_stats

        # Verify transformation stats show padding
        stats = result.transformation_stats
        assert stats["input_length"] == 3  # Original input size
        assert stats["output_length"] == week_size  # Padded output size
        assert stats["padding_length"] == week_size - 3  # Amount of padding
        assert stats["padding_percentage"] > 99.0  # Almost all padding

        # The last 3 values should correspond to our original input
        # (since padding is added at the beginning)
        actual_values = result.vector[-3:]  # Last 3 values
        assert len(actual_values) == 3

        # Values should be varied (not all identical) and within clipping range
        assert all(-4.0 <= val <= 4.0 for val in actual_values)
        unique_values = set(actual_values)
        assert (
            len(unique_values) >= 2
        )  # Should have some variation from different step counts

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_transform_with_caching(self, mock_lookup_stats: MagicMock) -> None:
        """Test transformation with caching enabled."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        transformer = ProxyActigraphyTransformer(cache_enabled=True)

        timestamps = [datetime.now(UTC) for _ in range(3)]
        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=[100.0, 150.0, 80.0],
            timestamps=timestamps,
        )

        # First transformation
        result1 = transformer.transform_step_data(step_data)

        # Second transformation (should use cache)
        result2 = transformer.transform_step_data(step_data)

        # Results should be identical
        assert result1.vector == result2.vector
        assert result1.quality_score == result2.quality_score

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_transform_without_caching(self, mock_lookup_stats: MagicMock) -> None:
        """Test transformation with caching disabled."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        transformer = ProxyActigraphyTransformer(cache_enabled=False)

        timestamps = [datetime.now(UTC)]
        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=[100.0],
            timestamps=timestamps,
        )

        result = transformer.transform_step_data(step_data)
        assert isinstance(result, ProxyActigraphyResult)

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_transform_large_dataset(self, mock_lookup_stats: MagicMock) -> None:
        """Test transformation with large dataset."""
        mock_lookup_stats.return_value = (3.2, 1.8)  # Updated to match new NHANES stats

        transformer = ProxyActigraphyTransformer()

        # Create large dataset - use exactly one week to avoid padding issues
        # MINUTES_PER_WEEK = 10080 (7 days * 24 hours * 60 minutes)
        week_size = 10080
        timestamps = [datetime.now(UTC) for _ in range(week_size)]
        # Create realistic step counts that will produce varied transformation results
        # Use values that work well with the updated NHANES normalization (mean=3.2, std=1.8)
        # Create more variety by using modulo 100 instead of 10 to get more unique values
        step_counts = [max(0.0, 1.0 + (i % 100) * 0.02) for i in range(week_size)]

        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=step_counts,
            timestamps=timestamps,
        )

        result = transformer.transform_step_data(step_data)

        assert isinstance(result, ProxyActigraphyResult)
        assert len(result.vector) == week_size
        assert result.transformation_stats["input_length"] == week_size
        assert result.transformation_stats["output_length"] == week_size

        # Verify the transformation produces varied values (not all the same)
        unique_values = set(result.vector)
        assert len(unique_values) > 10  # Should have many different values

        # Verify values are within expected range
        assert all(-5.0 <= val <= 5.0 for val in result.vector)

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_transform_with_padding(self, mock_lookup_stats: MagicMock) -> None:
        """Test transformation with small dataset that requires padding."""
        mock_lookup_stats.return_value = (3.2, 1.8)  # Updated to match new NHANES stats

        transformer = ProxyActigraphyTransformer()

        # Create small dataset that will trigger padding
        small_size = 100
        timestamps = [datetime.now(UTC) for _ in range(small_size)]
        step_counts = [max(0.0, 1.0 + (i % 20) * 0.1) for i in range(small_size)]

        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=step_counts,
            timestamps=timestamps,
        )

        result = transformer.transform_step_data(step_data)

        assert isinstance(result, ProxyActigraphyResult)
        # Output should be padded to full week
        week_size = 10080  # MINUTES_PER_WEEK
        assert len(result.vector) == week_size
        assert result.transformation_stats["input_length"] == small_size
        assert result.transformation_stats["output_length"] == week_size

        # Check padding behavior
        padded_values = result.vector[:-small_size]  # First 9980 values
        actual_values = result.vector[-small_size:]  # Last 100 values

        # With circadian-aware padding, values should be varied but in realistic range
        # They should be generally negative (rest period values) but not identical
        assert all(-4.0 <= val <= 4.0 for val in padded_values)  # Within clipping range

        # Padded values should have some variation (not all identical)
        unique_padded_values = set(padded_values)
        assert (
            len(unique_padded_values) > 10
        )  # Should have some variation due to circadian padding

        # Most padded values should be in the rest range (negative, around rest_value)
        rest_range_count = sum(1 for val in padded_values if -3.0 <= val <= -1.0)
        assert rest_range_count > len(padded_values) * 0.7  # At least 70% in rest range

        # Actual values should be varied and generally higher than padding
        unique_actual_values = set(actual_values)
        assert len(unique_actual_values) > 5  # Should have varied values

        # Verify padding statistics are tracked
        stats = result.transformation_stats
        assert stats["padding_length"] == week_size - small_size
        assert stats["padding_percentage"] > 99.0  # Almost all padding

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_quality_score_calculation(self, mock_lookup_stats: MagicMock) -> None:
        """Test quality score calculation."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        transformer = ProxyActigraphyTransformer()

        # Create data with known characteristics
        timestamps = [datetime.now(UTC) for _ in range(4)]
        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=[100.0, 110.0, 90.0, 105.0],  # Consistent data
            timestamps=timestamps,
        )

        result = transformer.transform_step_data(step_data)

        # Quality score should be reasonable for consistent data
        assert 0 <= result.quality_score <= 1

        # Check transformation stats
        stats = result.transformation_stats
        assert "zero_step_percentage" in stats
        assert "mean_steps_per_min" in stats
        assert "max_steps_per_min" in stats
        assert "total_steps" in stats

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_transform_with_zeros(self, mock_lookup_stats: MagicMock) -> None:
        """Test transformation with zero step counts."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        transformer = ProxyActigraphyTransformer()

        timestamps = [datetime.now(UTC) for _ in range(3)]
        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=[0.0, 0.0, 0.0],
            timestamps=timestamps,
        )

        result = transformer.transform_step_data(step_data)

        assert isinstance(result, ProxyActigraphyResult)
        # With padding to full week, zero percentage will be much lower
        assert (
            result.transformation_stats["zero_step_percentage"] < 1.0
        )  # Much lower due to padding
        # With circadian-aware padding, total_steps will be > 0 due to padding values
        assert (
            result.transformation_stats["total_steps"] > 0.0
        )  # Padding adds non-zero values

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_transform_with_extreme_values(self, mock_lookup_stats: MagicMock) -> None:
        """Test transformation with extreme step values."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        transformer = ProxyActigraphyTransformer()

        timestamps = [datetime.now(UTC) for _ in range(3)]
        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=[0.0, 10000.0, 100.0],  # Include extreme value
            timestamps=timestamps,
        )

        result = transformer.transform_step_data(step_data)

        assert isinstance(result, ProxyActigraphyResult)
        # Extreme values should be clipped
        assert all(-5.0 <= val <= 5.0 for val in result.vector)


class TestHelperFunctions:
    """Test helper functions and utilities."""

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_create_proxy_actigraphy_transformer(
        self, mock_lookup_stats: MagicMock
    ) -> None:
        """Test transformer factory function."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        transformer = create_proxy_actigraphy_transformer()
        assert isinstance(transformer, ProxyActigraphyTransformer)
        assert transformer.reference_year == 2025
        assert transformer.cache_enabled is True

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_create_transformer_with_params(self, mock_lookup_stats: MagicMock) -> None:
        """Test transformer factory with custom parameters."""
        mock_lookup_stats.return_value = (1.15, 0.75)

        transformer = create_proxy_actigraphy_transformer(
            reference_year=2024, cache_enabled=False
        )
        assert transformer.reference_year == 2024
        assert transformer.cache_enabled is False


class TestConstants:
    """Test module constants."""

    def test_constants_exist(self) -> None:
        """Test that required constants are defined."""
        assert isinstance(MINUTES_PER_WEEK, int)
        assert isinstance(MAX_REALISTIC_STEPS_PER_MINUTE, int)
        assert isinstance(MINUTES_PER_DAY, int)
        assert isinstance(DEFAULT_NHANES_STATS, dict)

    def test_constants_values(self) -> None:
        """Test that constants have reasonable values."""
        assert MINUTES_PER_WEEK == 7 * 24 * 60
        assert MINUTES_PER_DAY == 24 * 60
        assert MAX_REALISTIC_STEPS_PER_MINUTE > 0

        # Check DEFAULT_NHANES_STATS structure
        for year, stats in DEFAULT_NHANES_STATS.items():
            assert isinstance(year, str)
            assert "mean" in stats
            assert "std" in stats
            assert "source" in stats
            assert isinstance(stats["mean"], (int, float))
            assert isinstance(stats["std"], (int, float))
            assert stats["std"] > 0

    def test_performance_characteristics(self) -> None:
        """Test performance characteristics of the transformer."""
        with patch("clarity.ml.proxy_actigraphy.lookup_norm_stats") as mock_lookup:
            mock_lookup.return_value = (1.2, 0.8)

            transformer = ProxyActigraphyTransformer()

            # Test with different sizes
            small_size = 100
            large_size = 10080  # Full week

            # Small dataset test
            timestamps_small = [datetime.now(UTC) for _ in range(small_size)]
            step_data_small = StepCountData(
                user_id="user123",
                upload_id="upload456",
                step_counts=[float(i % 10) for i in range(small_size)],
                timestamps=timestamps_small,
            )

            start_time = time.time()
            result_small = transformer.transform_step_data(step_data_small)
            small_time = time.time() - start_time

            # Large dataset test
            timestamps_large = [datetime.now(UTC) for _ in range(large_size)]
            step_data_large = StepCountData(
                user_id="user123",
                upload_id="upload456",
                step_counts=[float(i % 100) for i in range(large_size)],
                timestamps=timestamps_large,
            )

            start_time = time.time()
            result_large = transformer.transform_step_data(step_data_large)
            large_time = time.time() - start_time

            # Basic performance checks
            assert small_time < 5.0  # Should complete in under 5 seconds
            assert large_time < 10.0  # Should complete in under 10 seconds

            # Verify results are valid
            assert isinstance(result_small, ProxyActigraphyResult)
            assert isinstance(result_large, ProxyActigraphyResult)
            assert len(result_small.vector) == large_size  # Padded to full week
            assert len(result_large.vector) == large_size


class TestIntegrationProxyActigraphy:
    """Integration tests for proxy actigraphy functionality."""

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_end_to_end_transformation(self, mock_lookup_stats: MagicMock) -> None:
        """Test complete end-to-end transformation workflow."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        # Create realistic step count data for a day
        base_time = datetime.now(UTC)
        timestamps = [base_time for _ in range(24)]  # Hourly data

        # Simulate realistic step patterns
        step_counts = []
        for hour in range(24):
            base_steps = 50 + (hour - 6) * 5 if 6 <= hour <= 22 else 5
            step_counts.append(float(base_steps))

        step_data = StepCountData(
            user_id="test_user",
            upload_id="test_upload",
            step_counts=step_counts,
            timestamps=timestamps,
            user_metadata={"age": 30, "sex": "female"},
        )

        transformer = ProxyActigraphyTransformer(reference_year=2025)
        result = transformer.transform_step_data(step_data)

        # Verify result structure
        assert isinstance(result, ProxyActigraphyResult)
        assert result.user_id == "test_user"
        assert result.upload_id == "test_upload"
        assert len(result.vector) == 10080  # Always padded to full week

        # Verify quality metrics
        assert 0 <= result.quality_score <= 1

        # Verify transformation stats
        stats = result.transformation_stats
        assert stats["input_length"] == 24
        assert stats["output_length"] == 10080  # Always padded to full week
        assert stats["total_steps"] > 0

        # Verify NHANES reference
        nhanes_ref = result.nhanes_reference
        assert nhanes_ref["year"] == 2025
        assert nhanes_ref["mean"] == 1.2
        assert nhanes_ref["std"] == 0.8

    def test_performance_characteristics(self) -> None:
        """Test performance characteristics of the transformer."""
        with patch("clarity.ml.proxy_actigraphy.lookup_norm_stats") as mock_lookup:
            mock_lookup.return_value = (1.2, 0.8)

            transformer = ProxyActigraphyTransformer()

            # Create moderately large dataset
            timestamps = [datetime.now(UTC) for _ in range(500)]
            step_counts = [100.0 + i for i in range(500)]

            step_data = StepCountData(
                user_id="perf_test",
                upload_id="perf_upload",
                step_counts=step_counts,
                timestamps=timestamps,
            )

            start_time = time.time()
            result = transformer.transform_step_data(step_data)
            end_time = time.time()

            # Should complete in reasonable time
            assert (end_time - start_time) < 1.0
            assert isinstance(result, ProxyActigraphyResult)
            assert len(result.vector) == 10080  # Always padded to full week


class TestEdgeCasesProxyActigraphy:
    """Test edge cases and boundary conditions."""

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_single_data_point(self, mock_lookup_stats: MagicMock) -> None:
        """Test transformation with single data point."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        transformer = ProxyActigraphyTransformer()

        timestamps = [datetime.now(UTC)]
        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=[100.0],
            timestamps=timestamps,
        )

        result = transformer.transform_step_data(step_data)

        assert isinstance(result, ProxyActigraphyResult)
        # Should be padded to full week size
        assert len(result.vector) == 10080
        assert result.transformation_stats["input_length"] == 1
        assert result.transformation_stats["output_length"] == 10080

    @patch("clarity.ml.proxy_actigraphy.lookup_norm_stats")
    def test_negative_step_counts(self, mock_lookup_stats: MagicMock) -> None:
        """Test transformation handles negative step counts."""
        mock_lookup_stats.return_value = (1.2, 0.8)

        transformer = ProxyActigraphyTransformer()

        timestamps = [datetime.now(UTC) for _ in range(3)]
        step_data = StepCountData(
            user_id="user123",
            upload_id="upload456",
            step_counts=[-10.0, 50.0, -5.0],  # Include negative values
            timestamps=timestamps,
        )

        result = transformer.transform_step_data(step_data)

        assert isinstance(result, ProxyActigraphyResult)
        # Should handle negative values gracefully
        assert len(result.vector) == 10080  # Padded to full week
        assert all(
            -5.0 <= val <= 5.0 for val in result.vector
        )  # Values should be clipped
