"""Tests for Proxy Actigraphy Transformation Module.

This module tests the conversion of Apple HealthKit step count data
into proxy actigraphy signals for PAT model analysis.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from typing import List

from clarity.ml.proxy_actigraphy import (  # type: ignore[import-untyped]
    ProxyActigraphyTransformer,
    ProxyActigraphyResult,
    TransformationQualityScore,
    ActigraphyDataPoint,
    ProxyActigraphyError,
)


class TestActigraphyDataPoint:
    """Test ActigraphyDataPoint model."""

    def test_actigraphy_data_point_creation(self):
        """Test creating a valid ActigraphyDataPoint."""
        timestamp = datetime.now(timezone.utc)
        point = ActigraphyDataPoint(
            timestamp=timestamp,
            value=100.5,
            quality_score=0.95
        )
        
        assert point.timestamp == timestamp
        assert point.value == 100.5
        assert point.quality_score == 0.95

    def test_actigraphy_data_point_validation(self):
        """Test ActigraphyDataPoint validation."""
        timestamp = datetime.now(timezone.utc)
        
        # Valid quality score
        point = ActigraphyDataPoint(
            timestamp=timestamp,
            value=100.5,
            quality_score=0.5
        )
        assert point.quality_score == 0.5
        
        # Test boundary values
        point_min = ActigraphyDataPoint(
            timestamp=timestamp,
            value=0.0,
            quality_score=0.0
        )
        assert point_min.quality_score == 0.0
        
        point_max = ActigraphyDataPoint(
            timestamp=timestamp,
            value=1000.0,
            quality_score=1.0
        )
        assert point_max.quality_score == 1.0


class TestTransformationQualityScore:
    """Test TransformationQualityScore model."""

    def test_quality_score_creation(self):
        """Test creating a valid TransformationQualityScore."""
        score = TransformationQualityScore(
            overall_score=0.85,
            data_completeness=0.90,
            temporal_consistency=0.80,
            outlier_ratio=0.05,
            normalization_quality=0.95
        )
        
        assert score.overall_score == 0.85
        assert score.data_completeness == 0.90
        assert score.temporal_consistency == 0.80
        assert score.outlier_ratio == 0.05
        assert score.normalization_quality == 0.95

    def test_quality_score_validation(self):
        """Test TransformationQualityScore validation."""
        # Test boundary values
        score = TransformationQualityScore(
            overall_score=1.0,
            data_completeness=1.0,
            temporal_consistency=1.0,
            outlier_ratio=0.0,
            normalization_quality=1.0
        )
        assert score.overall_score == 1.0


class TestProxyActigraphyResult:
    """Test ProxyActigraphyResult model."""

    def test_result_creation(self):
        """Test creating a valid ProxyActigraphyResult."""
        timestamp = datetime.now(timezone.utc)
        data_points = [
            ActigraphyDataPoint(timestamp=timestamp, value=100.0, quality_score=0.9)
        ]
        quality_score = TransformationQualityScore(
            overall_score=0.85,
            data_completeness=0.90,
            temporal_consistency=0.80,
            outlier_ratio=0.05,
            normalization_quality=0.95
        )
        
        result = ProxyActigraphyResult(
            proxy_actigraphy_data=data_points,
            quality_score=quality_score,
            transformation_metadata={
                "source": "apple_healthkit",
                "algorithm_version": "1.0"
            }
        )
        
        assert len(result.proxy_actigraphy_data) == 1
        assert result.quality_score.overall_score == 0.85
        assert result.transformation_metadata["source"] == "apple_healthkit"


class TestProxyActigraphyTransformer:
    """Test ProxyActigraphyTransformer functionality."""

    def test_transformer_initialization(self):
        """Test transformer initialization."""
        transformer = ProxyActigraphyTransformer()
        assert transformer is not None

    def test_transformer_initialization_with_params(self):
        """Test transformer initialization with custom parameters."""
        transformer = ProxyActigraphyTransformer(
            normalization_method="z_score",
            outlier_threshold=3.0,
            min_data_points=100
        )
        assert transformer is not None

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_transform_step_counts_basic(self, mock_lookup_stats):
        """Test basic step count transformation."""
        # Mock NHANES stats lookup
        mock_lookup_stats.return_value = (4.5, 2.0)  # mean, std
        
        transformer = ProxyActigraphyTransformer()
        
        # Create sample step count data
        step_counts = [
            {"timestamp": datetime.now(timezone.utc), "value": 1000},
            {"timestamp": datetime.now(timezone.utc), "value": 1500},
            {"timestamp": datetime.now(timezone.utc), "value": 800},
        ]
        
        result = transformer.transform_step_counts(
            step_counts=step_counts,
            user_age=25,
            user_sex="male"
        )
        
        assert isinstance(result, ProxyActigraphyResult)
        assert len(result.proxy_actigraphy_data) == len(step_counts)
        assert result.quality_score.overall_score > 0
        mock_lookup_stats.assert_called_once()

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_transform_empty_data(self, mock_lookup_stats):
        """Test transformation with empty data."""
        mock_lookup_stats.return_value = (4.5, 2.0)
        
        transformer = ProxyActigraphyTransformer()
        
        with pytest.raises(ProxyActigraphyError):
            transformer.transform_step_counts(
                step_counts=[],
                user_age=25,
                user_sex="male"
            )

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_transform_invalid_age(self, mock_lookup_stats):
        """Test transformation with invalid age."""
        mock_lookup_stats.return_value = None  # Invalid age returns None
        
        transformer = ProxyActigraphyTransformer()
        
        step_counts = [
            {"timestamp": datetime.now(timezone.utc), "value": 1000}
        ]
        
        with pytest.raises(ProxyActigraphyError):
            transformer.transform_step_counts(
                step_counts=step_counts,
                user_age=10,  # Invalid age
                user_sex="male"
            )

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_transform_with_outliers(self, mock_lookup_stats):
        """Test transformation with outlier data."""
        mock_lookup_stats.return_value = (4.5, 2.0)
        
        transformer = ProxyActigraphyTransformer(outlier_threshold=2.0)
        
        # Include some extreme values
        step_counts = [
            {"timestamp": datetime.now(timezone.utc), "value": 1000},
            {"timestamp": datetime.now(timezone.utc), "value": 50000},  # Extreme outlier
            {"timestamp": datetime.now(timezone.utc), "value": 800},
            {"timestamp": datetime.now(timezone.utc), "value": 0},      # Low outlier
        ]
        
        result = transformer.transform_step_counts(
            step_counts=step_counts,
            user_age=25,
            user_sex="male"
        )
        
        assert isinstance(result, ProxyActigraphyResult)
        # Quality score should be lower due to outliers
        assert result.quality_score.outlier_ratio > 0

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_transform_different_normalization_methods(self, mock_lookup_stats):
        """Test different normalization methods."""
        mock_lookup_stats.return_value = (4.5, 2.0)
        
        step_counts = [
            {"timestamp": datetime.now(timezone.utc), "value": 1000},
            {"timestamp": datetime.now(timezone.utc), "value": 1500},
        ]
        
        # Test z-score normalization
        transformer_zscore = ProxyActigraphyTransformer(normalization_method="z_score")
        result_zscore = transformer_zscore.transform_step_counts(
            step_counts=step_counts,
            user_age=25,
            user_sex="male"
        )
        
        # Test min-max normalization
        transformer_minmax = ProxyActigraphyTransformer(normalization_method="min_max")
        result_minmax = transformer_minmax.transform_step_counts(
            step_counts=step_counts,
            user_age=25,
            user_sex="male"
        )
        
        assert isinstance(result_zscore, ProxyActigraphyResult)
        assert isinstance(result_minmax, ProxyActigraphyResult)
        
        # Results should be different due to different normalization
        zscore_values = [p.value for p in result_zscore.proxy_actigraphy_data]
        minmax_values = [p.value for p in result_minmax.proxy_actigraphy_data]
        assert zscore_values != minmax_values

    def test_transform_invalid_normalization_method(self):
        """Test transformer with invalid normalization method."""
        with pytest.raises(ValueError):
            ProxyActigraphyTransformer(normalization_method="invalid_method")

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_transform_large_dataset(self, mock_lookup_stats):
        """Test transformation with large dataset."""
        mock_lookup_stats.return_value = (4.5, 2.0)
        
        transformer = ProxyActigraphyTransformer()
        
        # Create large dataset
        step_counts = []
        base_time = datetime.now(timezone.utc)
        for i in range(1000):
            step_counts.append({
                "timestamp": base_time,
                "value": 1000 + (i % 500)  # Varying values
            })
        
        result = transformer.transform_step_counts(
            step_counts=step_counts,
            user_age=25,
            user_sex="male"
        )
        
        assert isinstance(result, ProxyActigraphyResult)
        assert len(result.proxy_actigraphy_data) == 1000
        assert result.quality_score.data_completeness == 1.0  # All data present

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_quality_score_calculation(self, mock_lookup_stats):
        """Test quality score calculation components."""
        mock_lookup_stats.return_value = (4.5, 2.0)
        
        transformer = ProxyActigraphyTransformer()
        
        # Create data with known characteristics
        step_counts = [
            {"timestamp": datetime.now(timezone.utc), "value": 1000},
            {"timestamp": datetime.now(timezone.utc), "value": 1100},
            {"timestamp": datetime.now(timezone.utc), "value": 900},
            {"timestamp": datetime.now(timezone.utc), "value": 1050},
        ]
        
        result = transformer.transform_step_counts(
            step_counts=step_counts,
            user_age=25,
            user_sex="male"
        )
        
        quality = result.quality_score
        
        # Check all quality components are present and reasonable
        assert 0 <= quality.overall_score <= 1
        assert 0 <= quality.data_completeness <= 1
        assert 0 <= quality.temporal_consistency <= 1
        assert 0 <= quality.outlier_ratio <= 1
        assert 0 <= quality.normalization_quality <= 1
        
        # Data completeness should be 1.0 for complete data
        assert quality.data_completeness == 1.0

    def test_caching_functionality(self):
        """Test that caching works correctly."""
        transformer = ProxyActigraphyTransformer(enable_caching=True)
        
        # This test would need to check internal caching mechanisms
        # For now, just ensure the transformer can be created with caching enabled
        assert transformer is not None

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_metadata_generation(self, mock_lookup_stats):
        """Test transformation metadata generation."""
        mock_lookup_stats.return_value = (4.5, 2.0)
        
        transformer = ProxyActigraphyTransformer()
        
        step_counts = [
            {"timestamp": datetime.now(timezone.utc), "value": 1000}
        ]
        
        result = transformer.transform_step_counts(
            step_counts=step_counts,
            user_age=25,
            user_sex="male"
        )
        
        metadata = result.transformation_metadata
        
        # Check that metadata contains expected fields
        assert "algorithm_version" in metadata
        assert "normalization_method" in metadata
        assert "transformation_timestamp" in metadata
        assert "input_data_points" in metadata
        assert "outlier_threshold" in metadata


class TestProxyActigraphyError:
    """Test ProxyActigraphyError exception."""

    def test_error_creation(self):
        """Test creating ProxyActigraphyError."""
        error = ProxyActigraphyError("Test error message")
        assert isinstance(error, Exception)
        assert str(error) == "Test error message"

    def test_error_with_cause(self):
        """Test creating ProxyActigraphyError with cause."""
        original_error = ValueError("Original error")
        error = ProxyActigraphyError("Wrapper error") from original_error
        
        assert isinstance(error, ProxyActigraphyError)
        assert str(error) == "Wrapper error"
        assert error.__cause__ == original_error


class TestIntegrationProxyActigraphy:
    """Integration tests for proxy actigraphy functionality."""

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_end_to_end_transformation(self, mock_lookup_stats):
        """Test complete end-to-end transformation workflow."""
        mock_lookup_stats.return_value = (4.5, 2.0)
        
        # Create realistic step count data
        base_time = datetime.now(timezone.utc)
        step_counts = []
        
        # Simulate a day of step count data (24 hours, hourly readings)
        for hour in range(24):
            # Simulate realistic step patterns (more steps during day)
            if 6 <= hour <= 22:  # Daytime
                base_steps = 500 + (hour - 6) * 50
            else:  # Nighttime
                base_steps = 50
            
            step_counts.append({
                "timestamp": base_time,
                "value": base_steps + (hour % 3) * 100  # Add some variation
            })
        
        transformer = ProxyActigraphyTransformer(
            normalization_method="z_score",
            outlier_threshold=3.0,
            min_data_points=20
        )
        
        result = transformer.transform_step_counts(
            step_counts=step_counts,
            user_age=30,
            user_sex="female"
        )
        
        # Verify result structure
        assert isinstance(result, ProxyActigraphyResult)
        assert len(result.proxy_actigraphy_data) == 24
        
        # Verify quality metrics
        quality = result.quality_score
        assert quality.overall_score > 0.5  # Should be reasonable quality
        assert quality.data_completeness == 1.0  # Complete data
        
        # Verify metadata
        metadata = result.transformation_metadata
        assert metadata["input_data_points"] == 24
        assert metadata["normalization_method"] == "z_score"
        
        # Verify data points have reasonable values
        for point in result.proxy_actigraphy_data:
            assert isinstance(point.value, float)
            assert 0 <= point.quality_score <= 1

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_error_handling_workflow(self, mock_lookup_stats):
        """Test error handling throughout the workflow."""
        # Test with NHANES lookup failure
        mock_lookup_stats.return_value = None
        
        transformer = ProxyActigraphyTransformer()
        
        step_counts = [
            {"timestamp": datetime.now(timezone.utc), "value": 1000}
        ]
        
        with pytest.raises(ProxyActigraphyError):
            transformer.transform_step_counts(
                step_counts=step_counts,
                user_age=25,
                user_sex="male"
            )

    def test_performance_characteristics(self):
        """Test performance characteristics of the transformer."""
        import time
        
        transformer = ProxyActigraphyTransformer()
        
        # Create moderately large dataset
        step_counts = []
        base_time = datetime.now(timezone.utc)
        for i in range(500):
            step_counts.append({
                "timestamp": base_time,
                "value": 1000 + i
            })
        
        with patch('clarity.ml.proxy_actigraphy.lookup_norm_stats') as mock_lookup:
            mock_lookup.return_value = (4.5, 2.0)
            
            start_time = time.time()
            result = transformer.transform_step_counts(
                step_counts=step_counts,
                user_age=25,
                user_sex="male"
            )
            end_time = time.time()
            
            # Should complete in reasonable time (under 1 second for 500 points)
            assert (end_time - start_time) < 1.0
            assert isinstance(result, ProxyActigraphyResult)


class TestEdgeCasesProxyActigraphy:
    """Test edge cases and boundary conditions."""

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_single_data_point(self, mock_lookup_stats):
        """Test transformation with single data point."""
        mock_lookup_stats.return_value = (4.5, 2.0)
        
        transformer = ProxyActigraphyTransformer(min_data_points=1)
        
        step_counts = [
            {"timestamp": datetime.now(timezone.utc), "value": 1000}
        ]
        
        result = transformer.transform_step_counts(
            step_counts=step_counts,
            user_age=25,
            user_sex="male"
        )
        
        assert len(result.proxy_actigraphy_data) == 1
        assert result.quality_score.data_completeness == 1.0

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_zero_step_counts(self, mock_lookup_stats):
        """Test transformation with zero step counts."""
        mock_lookup_stats.return_value = (4.5, 2.0)
        
        transformer = ProxyActigraphyTransformer()
        
        step_counts = [
            {"timestamp": datetime.now(timezone.utc), "value": 0},
            {"timestamp": datetime.now(timezone.utc), "value": 0},
            {"timestamp": datetime.now(timezone.utc), "value": 0},
        ]
        
        result = transformer.transform_step_counts(
            step_counts=step_counts,
            user_age=25,
            user_sex="male"
        )
        
        assert isinstance(result, ProxyActigraphyResult)
        # Should handle zero values gracefully

    @patch('clarity.ml.proxy_actigraphy.lookup_norm_stats')
    def test_extreme_values(self, mock_lookup_stats):
        """Test transformation with extreme values."""
        mock_lookup_stats.return_value = (4.5, 2.0)
        
        transformer = ProxyActigraphyTransformer()
        
        step_counts = [
            {"timestamp": datetime.now(timezone.utc), "value": 0},
            {"timestamp": datetime.now(timezone.utc), "value": 100000},  # Very high
            {"timestamp": datetime.now(timezone.utc), "value": 1000},
        ]
        
        result = transformer.transform_step_counts(
            step_counts=step_counts,
            user_age=25,
            user_sex="male"
        )
        
        assert isinstance(result, ProxyActigraphyResult)
        # Should handle extreme values and flag them as outliers
        assert result.quality_score.outlier_ratio > 0 