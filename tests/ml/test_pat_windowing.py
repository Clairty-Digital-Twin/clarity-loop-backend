"""Tests for PAT model time window handling.

This module tests the canonical time window slicing implementation
to ensure consistent behavior across all PAT data paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from clarity.utils.time_window import (
    WEEK_MINUTES,
    pad_to_week,
    prepare_for_pat_inference,
    slice_to_weeks,
)


class TestSliceToWeeks:
    """Test the slice_to_weeks function."""

    def test_empty_array(self):
        """Test handling of empty array."""
        arr = np.array([])
        chunks = slice_to_weeks(arr)
        assert len(chunks) == 0

    def test_less_than_one_week(self):
        """Test array shorter than one week."""
        arr = np.arange(5000)  # Less than 10,080
        chunks = slice_to_weeks(arr, keep="all")
        assert len(chunks) == 0

    def test_exactly_one_week(self):
        """Test array of exactly one week."""
        arr = np.arange(WEEK_MINUTES)

        # Test "latest" mode
        chunks = slice_to_weeks(arr, keep="latest")
        assert len(chunks) == 1
        assert chunks[0].shape == (WEEK_MINUTES,)
        np.testing.assert_array_equal(chunks[0], arr)

        # Test "all" mode
        chunks = slice_to_weeks(arr, keep="all")
        assert len(chunks) == 1
        assert chunks[0].shape == (WEEK_MINUTES,)

    def test_multi_week_latest(self):
        """Test multi-week data with keep='latest'."""
        # 3 weeks of data
        arr = np.arange(30_240)
        chunks = slice_to_weeks(arr, keep="latest")

        assert len(chunks) == 1
        assert chunks[0].shape == (10_080,)
        # Should get the last week (20160 to 30239)
        expected = np.arange(20_160, 30_240)
        np.testing.assert_array_equal(chunks[0], expected)

    def test_multi_week_all(self):
        """Test multi-week data with keep='all'."""
        # 3 weeks of data
        arr = np.arange(30_240)
        chunks = slice_to_weeks(arr, keep="all")

        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert chunk.shape == (10_080,)
            expected_start = i * 10_080
            expected = np.arange(expected_start, expected_start + 10_080)
            np.testing.assert_array_equal(chunk, expected)

    def test_partial_week_trimming(self):
        """Test that partial weeks are trimmed from the beginning."""
        # 2.5 weeks of data (25,200 minutes)
        arr = np.arange(25_200)
        chunks = slice_to_weeks(arr, keep="all")

        assert len(chunks) == 2
        # First chunk should start at index 5040 (0.5 week trimmed)
        expected_first = np.arange(5_040, 15_120)
        np.testing.assert_array_equal(chunks[0], expected_first)

        # Second chunk
        expected_second = np.arange(15_120, 25_200)
        np.testing.assert_array_equal(chunks[1], expected_second)

    def test_2d_array(self):
        """Test handling of 2D arrays (multi-channel data)."""
        # 2 weeks of 3-channel data
        arr = np.random.rand(20_160, 3)  # noqa: NPY002
        chunks = slice_to_weeks(arr, keep="latest")

        assert len(chunks) == 1
        assert chunks[0].shape == (10_080, 3)

    def test_invalid_keep_parameter(self):
        """Test invalid keep parameter raises error."""
        arr = np.arange(10_080)
        with pytest.raises(ValueError, match="Invalid keep parameter"):
            slice_to_weeks(arr, keep="invalid")

    def test_high_dimensional_array_error(self):
        """Test that 3D+ arrays raise error."""
        arr = np.zeros((100, 10, 5))
        with pytest.raises(ValueError, match="Expected 1-D or 2-D array"):
            slice_to_weeks(arr)


class TestPadToWeek:
    """Test the pad_to_week function."""

    def test_pad_short_array_left(self):
        """Test padding short array on the left."""
        arr = np.ones(5000)
        padded = pad_to_week(arr, pad_side="left")

        assert padded.shape == (10_080,)
        assert np.all(padded[:5080] == 0.0)  # Padding
        assert np.all(padded[5080:] == 1.0)  # Original data

    def test_pad_short_array_right(self):
        """Test padding short array on the right."""
        arr = np.ones(5000)
        padded = pad_to_week(arr, pad_side="right")

        assert padded.shape == (10_080,)
        assert np.all(padded[:5000] == 1.0)  # Original data
        assert np.all(padded[5000:] == 0.0)  # Padding

    def test_pad_custom_value(self):
        """Test padding with custom value."""
        arr = np.ones(5000)
        padded = pad_to_week(arr, pad_value=-1.0)

        assert padded.shape == (10_080,)
        assert np.all(padded[:5080] == -1.0)  # Custom padding
        assert np.all(padded[5080:] == 1.0)  # Original data

    def test_no_padding_needed(self):
        """Test array that doesn't need padding."""
        arr = np.arange(10_080)
        padded = pad_to_week(arr)

        assert padded.shape == (10_080,)
        np.testing.assert_array_equal(padded, arr.astype(np.float32))

    def test_array_too_long_error(self):
        """Test that arrays longer than target raise error."""
        arr = np.arange(15_000)
        with pytest.raises(ValueError, match="exceeds target length"):
            pad_to_week(arr)

    def test_2d_array_padding(self):
        """Test padding of 2D arrays."""
        arr = np.ones((5000, 3))
        padded = pad_to_week(arr, pad_side="left")

        assert padded.shape == (10_080, 3)
        assert np.all(padded[:5080, :] == 0.0)  # Padding
        assert np.all(padded[5080:, :] == 1.0)  # Original data


class TestPrepareForPATInference:
    """Test the main prepare_for_pat_inference function."""

    def test_short_array_padded(self):
        """Test short arrays are padded."""
        arr = np.ones(5000)
        prepared = prepare_for_pat_inference(arr)

        assert prepared.shape == (10_080,)
        assert prepared.dtype == np.float32
        # Should be left-padded with zeros
        assert np.all(prepared[:5080] == 0.0)
        assert np.all(prepared[5080:] == 1.0)

    def test_exact_length_unchanged(self):
        """Test arrays of exact length are unchanged."""
        arr = np.arange(10_080, dtype=np.float64)
        prepared = prepare_for_pat_inference(arr)

        assert prepared.shape == (10_080,)
        assert prepared.dtype == np.float32
        np.testing.assert_array_equal(prepared, arr.astype(np.float32))

    def test_long_array_truncated(self):
        """Test long arrays are truncated to most recent data."""
        # 2 weeks of data
        arr = np.arange(20_160)
        prepared = prepare_for_pat_inference(arr)

        assert prepared.shape == (10_080,)
        assert prepared.dtype == np.float32
        # Should get the last week
        expected = np.arange(10_080, 20_160).astype(np.float32)
        np.testing.assert_array_equal(prepared, expected)

    def test_very_long_array(self):
        """Test very long arrays (3+ weeks)."""
        # 3.5 weeks of data
        arr = np.arange(35_280)
        prepared = prepare_for_pat_inference(arr)

        assert prepared.shape == (10_080,)
        # Should get the last complete week
        expected = np.arange(25_200, 35_280).astype(np.float32)
        np.testing.assert_array_equal(prepared, expected)

    def test_custom_target_length(self):
        """Test with custom target length."""
        arr = np.arange(2000)
        prepared = prepare_for_pat_inference(arr, target_length=1440)

        assert prepared.shape == (1440,)
        # Should get the last 1440 samples
        expected = np.arange(560, 2000).astype(np.float32)
        np.testing.assert_array_equal(prepared, expected)


class TestIntegrationWithPAT:
    """Integration tests simulating real PAT usage scenarios."""

    def test_preprocessing_consistency(self):
        """Test that all preprocessing paths produce consistent results."""
        # Create test data of various lengths
        test_cases = [
            (5000, "Short data (< 1 week)"),
            (10_080, "Exact 1 week"),
            (15_000, "1.5 weeks"),
            (20_160, "Exact 2 weeks"),
            (30_240, "Exact 3 weeks"),
            (35_000, "~3.5 weeks"),
        ]

        for length, description in test_cases:
            arr = np.random.rand(length)  # noqa: NPY002
            prepared = prepare_for_pat_inference(arr)

            # All should produce exactly 10,080 samples
            assert prepared.shape == (10_080,), f"Failed for {description}"
            assert prepared.dtype == np.float32, f"Wrong dtype for {description}"

            # Verify data integrity
            if length >= 10_080:
                # Should contain the most recent 10,080 samples
                expected_start = length - 10_080
                expected = arr[expected_start:].astype(np.float32)
                np.testing.assert_allclose(
                    prepared, expected, err_msg=f"Data mismatch for {description}"
                )

    def test_pat_model_shape_requirement(self):
        """Test that output always matches PAT model requirements."""
        # PAT model expects input shape (batch_size, 10080)
        # This test verifies our preprocessing produces the right shape

        test_lengths = [1000, 10_080, 20_000, 50_000]

        for length in test_lengths:
            arr = np.random.rand(length)  # noqa: NPY002
            prepared = prepare_for_pat_inference(arr)
            # Simulate adding batch dimension as done in pat_service.py
            input_tensor = np.expand_dims(prepared, 0)

            assert input_tensor.shape == (1, 10_080)
            assert input_tensor.dtype == np.float32

    def test_multi_week_analysis_scenario(self):
        """Test scenario where we might want to analyze multiple weeks."""
        # 3 weeks of data
        data = np.random.rand(30_240)  # noqa: NPY002
        # Get all weeks
        all_weeks = slice_to_weeks(data, keep="all")
        assert len(all_weeks) == 3

        # Each week should be ready for PAT
        for i, week_data in enumerate(all_weeks):
            assert week_data.shape == (10_080,)

            # Verify it's the correct week
            expected_start = i * 10_080
            expected = data[expected_start : expected_start + 10_080]
            np.testing.assert_array_equal(week_data, expected)

    def test_realtime_streaming_scenario(self):
        """Test scenario of continuous data streaming."""
        # Simulate receiving data over time
        accumulated_data = []

        # Day 1-5: Not enough data
        for _day in range(5):
            daily_data = np.random.rand(1440)  # 1 day  # noqa: NPY002
            accumulated_data.extend(daily_data)

            arr = np.array(accumulated_data)
            prepared = prepare_for_pat_inference(arr)

            # Should be padded
            assert prepared.shape == (10_080,)
            assert np.sum(prepared == 0.0) > 0  # Has padding

        # Day 8: More than 1 week
        for _day in range(5, 8):
            daily_data = np.random.rand(1440)  # noqa: NPY002
            accumulated_data.extend(daily_data)

        arr = np.array(accumulated_data)
        prepared = prepare_for_pat_inference(arr)

        # Should truncate to most recent week
        assert prepared.shape == (10_080,)
        expected = arr[-10_080:].astype(np.float32)
        np.testing.assert_array_equal(prepared, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_arrays(self):
        """Test handling of very small arrays."""
        for size in [0, 1, 10, 100]:
            arr = np.ones(size)
            if size == 0:
                chunks = slice_to_weeks(arr)
                assert len(chunks) == 0
            else:
                prepared = prepare_for_pat_inference(arr)
                assert prepared.shape == (10_080,)
                assert np.sum(prepared == 1.0) == size
                assert np.sum(prepared == 0.0) == 10_080 - size

    def test_data_types(self):
        """Test handling of different data types."""
        test_data = np.arange(10_080)

        for dtype in [np.int32, np.int64, np.float32, np.float64]:
            arr = test_data.astype(dtype)
            prepared = prepare_for_pat_inference(arr)

            assert prepared.dtype == np.float32
            np.testing.assert_array_equal(prepared, test_data.astype(np.float32))

    def test_negative_values(self):
        """Test handling of negative values."""
        arr = np.arange(-5000, 5080)
        prepared = prepare_for_pat_inference(arr)

        assert prepared.shape == (10_080,)
        assert prepared.dtype == np.float32
        np.testing.assert_array_equal(prepared, arr.astype(np.float32))

    def test_nan_inf_values(self):
        """Test that NaN and inf values are preserved."""
        arr = np.ones(10_080)
        arr[100] = np.nan
        arr[200] = np.inf
        arr[300] = -np.inf

        prepared = prepare_for_pat_inference(arr)

        assert np.isnan(prepared[100])
        assert np.isinf(prepared[200])
        assert prepared[200] > 0
        assert np.isinf(prepared[300])
        assert prepared[300] < 0
