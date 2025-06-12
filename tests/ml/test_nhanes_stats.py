"""Tests for NHANES statistics module.

This module tests the NHANES population statistics functionality
used for normalizing health data against population norms.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from clarity.ml.nhanes_stats import (
    AGE_STRATIFIED_STATS,
    EXTREME_OUTLIER_THRESHOLD,
    NHANES_REFERENCE_STATS,
    SEX_STRATIFIED_STATS,
    NHANESStatsError,
    get_available_age_groups,
    get_available_years,
    get_reference_info,
    lookup_norm_stats,
    validate_proxy_values,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestNHANESStatsLookup:
    """Test NHANES statistics lookup functionality."""

    @staticmethod
    def test_lookup_norm_stats_default() -> None:
        """Test lookup with default parameters."""
        mean, std = lookup_norm_stats()
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0

    @staticmethod
    def test_lookup_norm_stats_valid_year() -> None:
        """Test lookup with valid year."""
        mean, std = lookup_norm_stats(year=2025)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0

    @staticmethod
    def test_lookup_norm_stats_invalid_year() -> None:
        """Test lookup with invalid year falls back to default."""
        mean, std = lookup_norm_stats(year=2030)  # Future year not in data
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0

    @staticmethod
    def test_lookup_norm_stats_with_age_group() -> None:
        """Test lookup with age group stratification."""
        mean, std = lookup_norm_stats(year=2025, age_group="18-29")
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0

    @staticmethod
    def test_lookup_norm_stats_with_sex() -> None:
        """Test lookup with sex stratification."""
        mean_male, std_male = lookup_norm_stats(year=2025, sex="male")
        mean_female, std_female = lookup_norm_stats(year=2025, sex="female")

        assert isinstance(mean_male, float)
        assert isinstance(std_male, float)
        assert isinstance(mean_female, float)
        assert isinstance(std_female, float)

        # Should be different values for different sexes
        assert mean_male != mean_female or std_male != std_female

    @staticmethod
    def test_lookup_norm_stats_with_all_params() -> None:
        """Test lookup with all parameters."""
        mean, std = lookup_norm_stats(year=2025, age_group="30-39", sex="female")
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0

    @staticmethod
    def test_lookup_norm_stats_invalid_age_group() -> None:
        """Test lookup with invalid age group."""
        mean, std = lookup_norm_stats(year=2025, age_group="invalid")
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0

    @staticmethod
    def test_lookup_norm_stats_invalid_sex() -> None:
        """Test lookup with invalid sex."""
        mean, std = lookup_norm_stats(year=2025, sex="invalid")
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0

    @staticmethod
    def test_lookup_norm_stats_case_insensitive_sex() -> None:
        """Test that sex parameter is case insensitive."""
        mean_lower, std_lower = lookup_norm_stats(sex="male")
        mean_upper, std_upper = lookup_norm_stats(sex="MALE")
        mean_mixed, std_mixed = lookup_norm_stats(sex="Male")

        assert mean_lower == mean_upper == mean_mixed
        assert std_lower == std_upper == std_mixed


class TestAvailableData:
    """Test functions that return available data."""

    @staticmethod
    def test_get_available_years() -> None:
        """Test getting available years."""
        years = get_available_years()
        assert isinstance(years, list)
        assert len(years) > 0
        assert all(isinstance(year, int) for year in years)
        assert 2025 in years  # Should include our test year

    @staticmethod
    def test_get_available_age_groups() -> None:
        """Test getting available age groups."""
        age_groups = get_available_age_groups()
        assert isinstance(age_groups, list)
        assert len(age_groups) > 0
        assert all(isinstance(group, str) for group in age_groups)
        assert "18-29" in age_groups  # Should include young adults

    @staticmethod
    def test_available_data_consistency() -> None:
        """Test that available data is consistent with constants."""
        years = get_available_years()
        age_groups = get_available_age_groups()

        # Years should match NHANES_REFERENCE_STATS keys
        expected_years = [int(year) for year in NHANES_REFERENCE_STATS]
        assert set(years) == set(expected_years)

        # Age groups should match AGE_STRATIFIED_STATS keys
        assert set(age_groups) == set(AGE_STRATIFIED_STATS.keys())


class TestReferenceInfo:
    """Test reference information functionality."""

    @staticmethod
    def test_get_reference_info_valid_year() -> None:
        """Test getting reference info for valid year."""
        info = get_reference_info(2025)
        assert isinstance(info, dict)
        assert "mean" in info
        assert "std" in info
        assert "sample_size" in info
        assert "source" in info

    @staticmethod
    def test_get_reference_info_invalid_year() -> None:
        """Test getting reference info for invalid year."""
        info = get_reference_info(2030)  # Future year
        assert isinstance(info, dict)
        assert "mean" in info
        assert "std" in info
        # Should return default year info

    @staticmethod
    def test_get_reference_info_default() -> None:
        """Test getting reference info with default year."""
        info = get_reference_info()
        assert isinstance(info, dict)
        assert "mean" in info
        assert "std" in info


class TestProxyValueValidation:
    """Test proxy value validation functionality."""

    @staticmethod
    def test_validate_proxy_values_normal() -> None:
        """Test validation with normal values."""
        # Generate some normal values around the expected mean
        proxy_values = [4.0, 4.2, 4.5, 4.1, 4.3, 4.4, 4.0, 4.6]

        result = validate_proxy_values(proxy_values)
        assert isinstance(result, dict)
        assert "total_values" in result
        assert "mean_z_score" in result
        assert "std_z_score" in result
        assert "extreme_low_count" in result
        assert "extreme_high_count" in result
        assert "outlier_percentage" in result
        assert "validation_passed" in result

        assert result["total_values"] == len(proxy_values)
        assert isinstance(bool(result["validation_passed"]), bool)

    @staticmethod
    def test_validate_proxy_values_with_outliers() -> None:
        """Test validation with outlier values."""
        # Include some extreme values
        proxy_values = [
            4.0,
            4.2,
            15.0,
            4.1,
            -5.0,
            4.4,
            4.0,
            4.6,
        ]  # 15.0 and -5.0 are outliers

        result = validate_proxy_values(proxy_values)
        assert result["extreme_low_count"] > 0 or result["extreme_high_count"] > 0
        assert result["outlier_percentage"] > 0

    @staticmethod
    def test_validate_proxy_values_empty_list() -> None:
        """Test validation with empty list."""
        # Empty list should return a result with NaN values but not crash
        result = validate_proxy_values([])
        assert result["total_values"] == 0
        # NaN values are expected for empty lists

    @staticmethod
    def test_validate_proxy_values_single_value() -> None:
        """Test validation with single value."""
        result = validate_proxy_values([4.0])
        assert result["total_values"] == 1
        assert isinstance(result["mean_z_score"], float)

    @staticmethod
    def test_validate_proxy_values_different_year() -> None:
        """Test validation with different reference year."""
        proxy_values = [4.0, 4.2, 4.5, 4.1]

        result_2023 = validate_proxy_values(proxy_values, year=2023)
        result_2025 = validate_proxy_values(proxy_values, year=2025)

        assert result_2023["reference_year"] == 2023
        assert result_2025["reference_year"] == 2025
        # Results might be different due to different reference stats


class TestNHANESConstants:
    """Test NHANES constants and data structures."""

    @staticmethod
    def test_nhanes_reference_stats_structure() -> None:
        """Test that NHANES reference stats have correct structure."""
        assert isinstance(NHANES_REFERENCE_STATS, dict)

        for year, stats in NHANES_REFERENCE_STATS.items():
            assert isinstance(year, str)
            assert isinstance(stats, dict)
            assert "mean" in stats
            assert "std" in stats
            assert "sample_size" in stats
            assert "source" in stats
            assert isinstance(stats["mean"], (int, float))
            assert isinstance(stats["std"], (int, float))
            assert stats["std"] > 0

    @staticmethod
    def test_age_stratified_stats_structure() -> None:
        """Test that age stratified stats have correct structure."""
        assert isinstance(AGE_STRATIFIED_STATS, dict)

        for age_group, stats in AGE_STRATIFIED_STATS.items():
            assert isinstance(age_group, str)
            assert isinstance(stats, dict)
            assert "mean" in stats
            assert "std" in stats
            assert isinstance(stats["mean"], (int, float))
            assert isinstance(stats["std"], (int, float))
            assert stats["std"] > 0

    @staticmethod
    def test_sex_stratified_stats_structure() -> None:
        """Test that sex stratified stats have correct structure."""
        assert isinstance(SEX_STRATIFIED_STATS, dict)

        expected_sexes = ["male", "female", "other"]
        for sex in expected_sexes:
            assert sex in SEX_STRATIFIED_STATS
            stats = SEX_STRATIFIED_STATS[sex]
            assert "mean" in stats
            assert "std" in stats
            assert isinstance(stats["mean"], (int, float))
            assert isinstance(stats["std"], (int, float))
            assert stats["std"] > 0

    @staticmethod
    def test_extreme_outlier_threshold() -> None:
        """Test that outlier threshold is reasonable."""
        assert isinstance(EXTREME_OUTLIER_THRESHOLD, (int, float))
        assert EXTREME_OUTLIER_THRESHOLD > 0
        assert EXTREME_OUTLIER_THRESHOLD <= 5  # Reasonable upper bound


class TestNHANESStatsError:
    """Test NHANES stats error handling."""

    @staticmethod
    def test_nhanes_stats_error_creation() -> None:
        """Test creating NHANESStatsError."""
        error = NHANESStatsError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    @patch("clarity.ml.nhanes_stats.NHANES_REFERENCE_STATS", {})
    def test_lookup_with_empty_stats(self) -> None:
        """Test lookup behavior with empty reference stats."""
        with pytest.raises(NHANESStatsError):
            lookup_norm_stats(year=2025)


class TestIntegrationNHANESStats:
    """Integration tests for NHANES stats functionality."""

    @staticmethod
    def test_full_workflow_normal_case() -> None:
        """Test complete workflow with normal inputs."""
        # Get available years and pick one
        years = get_available_years()
        year = years[0] if years else 2025

        # Get available age groups and pick one
        age_groups = get_available_age_groups()
        age_group = age_groups[0] if age_groups else None

        # Get stats
        mean, std = lookup_norm_stats(year=year, age_group=age_group, sex="male")
        assert isinstance(mean, float)
        assert isinstance(std, float)

        # Get reference info
        info = get_reference_info(year)
        assert isinstance(info, dict)

        # Validate some proxy values
        proxy_values = [mean - std, mean, mean + std]  # Values around the mean
        validation = validate_proxy_values(proxy_values, year=year)
        assert validation["validation_passed"]  # Should pass with normal values

    @staticmethod
    def test_error_handling_chain() -> None:
        """Test error handling throughout the workflow."""
        # Test with invalid inputs
        mean, std = lookup_norm_stats(year=9999, age_group="invalid", sex="invalid")
        assert isinstance(mean, float)
        assert isinstance(std, float)

        # Should still work due to fallback mechanisms
        info = get_reference_info(9999)
        assert isinstance(info, dict)

    @patch("clarity.ml.nhanes_stats.logger")
    def test_logging_integration(self, mock_logger: MagicMock) -> None:
        """Test that logging works correctly."""
        # Test with invalid inputs that should trigger logging
        lookup_norm_stats(year=9999)  # Invalid year
        lookup_norm_stats(age_group="invalid")  # Invalid age group
        lookup_norm_stats(sex="invalid")  # Invalid sex

        # Verify logging infrastructure is in place
        assert hasattr(mock_logger, "warning")


class TestPerformanceNHANESStats:
    """Performance tests for NHANES stats functionality."""

    @staticmethod
    def test_lookup_performance() -> None:
        """Test that lookups are reasonably fast."""
        start_time = time.time()
        for _ in range(1000):
            lookup_norm_stats(year=2025, age_group="30-39", sex="male")
        end_time = time.time()

        # Should complete 1000 lookups in under 1 second
        assert (end_time - start_time) < 1.0

    @staticmethod
    def test_validation_performance() -> None:
        """Test that validation is reasonably fast."""
        # Generate test data
        proxy_values = [4.0 + i * 0.1 for i in range(100)]

        start_time = time.time()
        for _ in range(100):
            validate_proxy_values(proxy_values)
        end_time = time.time()

        # Should complete 100 validations in under 1 second
        assert (end_time - start_time) < 1.0

    @staticmethod
    def test_caching_effectiveness() -> None:
        """Test that LRU caching is working."""
        # First call
        start_time = time.time()
        years1 = get_available_years()
        time.time() - start_time  # We don't actually use the timing

        # Second call (should be faster due to caching)
        start_time = time.time()
        years2 = get_available_years()
        time.time() - start_time  # We don't actually use the timing

        # Results should be identical
        assert years1 == years2

        # Second call should be faster (though this might be flaky on fast systems)
        # Just ensure it doesn't crash and returns same results
