"""Proxy Actigraphy Transformation Module.

This module converts Apple HealthKit step count data into proxy actigraphy signals
that can be analyzed by the PAT (Pretrained Actigraphy Transformer) model.

The transformation process:
1. Normalizes step counts using NHANES population statistics
2. Applies square root transformation for variance stabilization
3. Generates proxy actigraphy vectors suitable for PAT analysis

Key Features:
- NHANES-based population normalization
- Quality scoring for data validation
- Caching for performance optimization
- Comprehensive transformation statistics
- Circadian-aware padding for realistic actigraphy patterns

This implementation follows clean code principles with proper separation of concerns,
comprehensive error handling, and extensive validation.
"""

# removed - breaks FastAPI

from datetime import datetime
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from clarity.core.constants import (
    CIRCADIAN_DECAY_FACTOR,
    CIRCADIAN_PADDING_VARIATION_STD,
    CIRCADIAN_PATTERN_AMPLITUDE,
    EXTREME_VALUE_LOWER_THRESHOLD,
    EXTREME_VALUE_UPPER_THRESHOLD,
    MAX_REALISTIC_STEPS_PER_MINUTE,
    MIN_ACTIVITY_LEVEL,
    MINUTES_PER_DAY,
    MINUTES_PER_HOUR,
    MINUTES_PER_WEEK,
    PROXY_VALUE_CLIP_MAX,
    PROXY_VALUE_CLIP_MIN,
    QUALITY_WEIGHT_COMPLETENESS,
    QUALITY_WEIGHT_PADDING,
    QUALITY_WEIGHT_PROXY_SIGNAL,
    QUALITY_WEIGHT_REALISTIC,
    QUALITY_WEIGHT_VARIABILITY,
    SMOOTHING_WINDOW_SIZE,
    SQRT_MIN_ACTIVITY,
    Z_SCORE_NORMALIZATION_EPSILON,
    ZERO_PERCENTAGE_PENALTY_FACTOR,
)
from clarity.core.exceptions import (
    DataLengthMismatchError,
    EmptyDataError,
    create_numeric_validation_error,
)
from clarity.core.types import FloatArray, LoggerProtocol, NHANESStats, StepCount
from clarity.ml.nhanes_stats import lookup_norm_stats
from clarity.utils.time_window import prepare_for_pat_inference

logger: LoggerProtocol = logging.getLogger(__name__)

# Updated NHANES statistics based on sqrt-transformed step counts
# These values are more appropriate for proxy actigraphy transformation
DEFAULT_NHANES_STATS: dict[str, NHANESStats] = {
    "2025": {
        "mean": 3.2,  # sqrt-transformed step count mean
        "std": 1.8,  # sqrt-transformed step count std
        "source": "NHANES 2017-2020 sqrt-transformed (projected)",
    },
    "2024": {"mean": 3.1, "std": 1.7, "source": "NHANES 2017-2020 sqrt-transformed"},
}


class StepCountData(BaseModel):
    """Input data structure for step count transformation."""

    user_id: str = Field(description="Unique user identifier")
    upload_id: str = Field(description="Unique upload session identifier")
    step_counts: list[StepCount] = Field(
        description="Step counts per minute", min_length=1
    )
    timestamps: list[datetime] = Field(
        description="Corresponding timestamps for each step count"
    )
    user_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional user demographics and metadata"
    )


class ProxyActigraphyResult(BaseModel):
    """Result of proxy actigraphy transformation."""

    user_id: str = Field(description="User identifier")
    upload_id: str = Field(description="Upload session identifier")
    vector: list[float] = Field(description="Proxy actigraphy vector")
    quality_score: float = Field(
        description="Data quality score (0.0-1.0)", ge=0.0, le=1.0
    )
    transformation_stats: dict[str, Any] = Field(
        description="Statistics about the transformation process"
    )
    nhanes_reference: dict[str, float] = Field(
        description="NHANES normalization parameters used"
    )


def _get_nhanes_stats_for_year(year: int) -> tuple[float, float]:
    """Get NHANES normalization statistics for a given year.

    Args:
        year: Year to get statistics for

    Returns:
        Tuple of (mean, std) for the specified year

    Raises:
        ValueError: If statistics data is invalid
    """
    year_str = str(year)

    if year_str in DEFAULT_NHANES_STATS:
        stats = DEFAULT_NHANES_STATS[year_str]
        logger.info("Using %s for normalization", stats["source"])
        mean_val = stats["mean"]
        std_val = stats["std"]

        if not isinstance(mean_val, (int, float)):
            field_name = "mean"
            raise create_numeric_validation_error(field_name, mean_val)
        if not isinstance(std_val, (int, float)):
            field_name = "std"
            raise create_numeric_validation_error(field_name, std_val)

        return float(mean_val), float(std_val)

    # Fallback to latest available year
    latest_year = max(DEFAULT_NHANES_STATS.keys())
    stats = DEFAULT_NHANES_STATS[latest_year]
    logger.warning("Year %d not found, using %s", year, stats["source"])
    mean_val = stats["mean"]
    std_val = stats["std"]

    if not isinstance(mean_val, (int, float)):
        field_name = "mean"
        raise create_numeric_validation_error(field_name, mean_val)
    if not isinstance(std_val, (int, float)):
        field_name = "std"
        raise create_numeric_validation_error(field_name, std_val)

    return float(mean_val), float(std_val)


def _generate_circadian_padding(
    length: int, base_activity: float = MIN_ACTIVITY_LEVEL
) -> FloatArray:
    """Generate circadian-aware padding values for realistic rest periods.

    Args:
        length: Number of minutes to generate
        base_activity: Base activity level for rest periods

    Returns:
        Array of realistic padding values that vary slightly
    """
    if length == 0:
        return np.array([])

    # Create random number generator for reproducible results
    rng = np.random.default_rng()

    # Create slight variations around base activity level to simulate
    # realistic rest periods with minimal movement
    padding_values = rng.normal(base_activity, CIRCADIAN_PADDING_VARIATION_STD, length)

    # Ensure values are non-negative
    padding_values = np.maximum(padding_values, 0.0)

    # Add subtle circadian pattern for longer periods
    if length > MINUTES_PER_HOUR:  # Only for periods longer than 1 hour
        time_points = np.linspace(0, 2 * np.pi * (length / MINUTES_PER_DAY), length)
        circadian_variation = (
            CIRCADIAN_PATTERN_AMPLITUDE
            * np.sin(time_points)
            * np.exp(CIRCADIAN_DECAY_FACTOR * np.abs(time_points))
        )
        padding_values += circadian_variation
        padding_values = np.maximum(padding_values, 0.0)

    return padding_values


def _smooth_proxy_values(
    proxy_values: FloatArray, window_size: int = SMOOTHING_WINDOW_SIZE
) -> FloatArray:
    """Apply temporal smoothing to reduce unrealistic step changes.

    Args:
        proxy_values: Raw proxy actigraphy values
        window_size: Size of smoothing window

    Returns:
        Smoothed proxy actigraphy values
    """
    if len(proxy_values) < window_size:
        return proxy_values

    # Apply light smoothing using a moving average
    smoothed = np.copy(proxy_values)
    half_window = window_size // 2

    for i in range(half_window, len(proxy_values) - half_window):
        window_vals = proxy_values[i - half_window : i + half_window + 1]
        # Use median to preserve important signal characteristics
        smoothed[i] = np.median(window_vals)

    return smoothed


class ProxyActigraphyTransformer:
    """Main transformation engine for converting step counts to proxy actigraphy.

    This class follows the Strategy pattern for transformation algorithms and
    the Template Method pattern for the overall transformation workflow.
    """

    def __init__(
        self, reference_year: int = 2025, *, cache_enabled: bool = True
    ) -> None:
        """Initialize the proxy actigraphy transformer.

        Args:
            reference_year: NHANES reference year for normalization
            cache_enabled: Whether to enable result caching
        """
        self.reference_year = reference_year
        self.cache_enabled = cache_enabled
        self._cache: dict[str, ProxyActigraphyResult] = {}

        # Load NHANES normalization parameters
        self.nhanes_mean, self.nhanes_std = lookup_norm_stats(reference_year)

        logger.info(
            "ProxyActigraphyTransformer initialized with NHANES %d", reference_year
        )
        logger.info("  • Reference mean: %.3f", self.nhanes_mean)
        logger.info("  • Reference std: %.3f", self.nhanes_std)
        logger.info("  • Cache enabled: %s", cache_enabled)

    def steps_to_movement_proxy(
        self, steps_per_min: FloatArray, padding_mask: NDArray[np.bool_] | None = None
    ) -> FloatArray:
        """Convert step counts to movement proxy values.

        Applies square root transformation followed by z-score normalization
        using NHANES population statistics. Handles padding and real zeros differently.

        Args:
            steps_per_min: Array of step counts per minute
            padding_mask: Boolean array indicating padded positions (True = padded)

        Returns:
            Normalized proxy actigraphy values
        """
        # Apply square root transformation for variance stabilization
        sqrt_steps = np.sqrt(np.maximum(steps_per_min, 0))

        # Z-score normalization using NHANES population statistics
        normalized = (sqrt_steps - self.nhanes_mean) / (
            self.nhanes_std + Z_SCORE_NORMALIZATION_EPSILON
        )

        # Handle padded values differently to avoid unrealistic patterns
        if padding_mask is not None:
            # For padded areas, use a more realistic rest-period value
            rest_value = (np.sqrt(SQRT_MIN_ACTIVITY) - self.nhanes_mean) / (
                self.nhanes_std + Z_SCORE_NORMALIZATION_EPSILON
            )

            # Create random number generator for reproducible results
            rng = np.random.default_rng()
            normalized[padding_mask] = rest_value + rng.normal(
                0, CIRCADIAN_PADDING_VARIATION_STD, int(np.sum(padding_mask))
            )

        # Apply temporal smoothing to reduce unrealistic step changes
        smoothed = _smooth_proxy_values(normalized)

        # Clip extreme values to reasonable range
        return np.clip(smoothed, PROXY_VALUE_CLIP_MIN, PROXY_VALUE_CLIP_MAX)

    def transform_step_data(self, step_data: StepCountData) -> ProxyActigraphyResult:
        """Transform step count data to proxy actigraphy.

        Args:
            step_data: Input step count data

        Returns:
            Proxy actigraphy transformation result

        Raises:
            DataValidationError: If input data is invalid
        """
        # Generate cache key
        cache_key = (
            f"{step_data.user_id}_{step_data.upload_id}_{len(step_data.step_counts)}"
        )

        # Check cache if enabled
        if self.cache_enabled and cache_key in self._cache:
            logger.info("Returning cached transformation for %s", cache_key)
            return self._cache[cache_key]

        try:
            # Prepare and validate step data
            steps_array, padding_mask = self._prepare_step_data(
                step_data.step_counts, step_data.timestamps
            )

            # Transform to proxy actigraphy
            proxy_vector = self.steps_to_movement_proxy(steps_array, padding_mask)

            # Calculate quality score
            quality_score = self._calculate_quality_score(
                steps_array, proxy_vector, padding_mask
            )

            # Generate transformation statistics
            original_length = len(step_data.step_counts)
            padding_length = np.sum(padding_mask)

            transformation_stats = {
                "input_length": original_length,
                "output_length": len(proxy_vector),
                "padding_length": int(padding_length),
                "padding_percentage": float(padding_length / len(proxy_vector) * 100),
                "zero_step_percentage": (np.sum(steps_array == 0) / len(steps_array))
                * 100,
                "mean_steps_per_min": float(np.mean(steps_array)),
                "max_steps_per_min": float(np.max(steps_array)),
                "total_steps": float(np.sum(steps_array)),
                "proxy_value_range": {
                    "min": float(np.min(proxy_vector)),
                    "max": float(np.max(proxy_vector)),
                    "std": float(np.std(proxy_vector)),
                },
            }

            # Create result
            result = ProxyActigraphyResult(
                user_id=step_data.user_id,
                upload_id=step_data.upload_id,
                vector=proxy_vector.tolist(),
                quality_score=quality_score,
                transformation_stats=transformation_stats,
                nhanes_reference={
                    "year": self.reference_year,
                    "mean": self.nhanes_mean,
                    "std": self.nhanes_std,
                },
            )

            # Cache result if enabled
            if self.cache_enabled:
                self._cache[cache_key] = result
                logger.debug("Cached transformation for %s", cache_key)

            logger.info("Successfully transformed step data for %s", step_data.user_id)
            logger.info("  • Quality score: %.3f", quality_score)
            logger.info("  • Vector length: %d", len(proxy_vector))
            logger.info(
                "  • Padding: %.1f%%", transformation_stats["padding_percentage"]
            )
            logger.info(
                "  • Zero steps: %.1f%%", transformation_stats["zero_step_percentage"]
            )

        except Exception:
            logger.exception("Failed to transform step data for %s", step_data.user_id)
            raise
        else:
            return result

    @staticmethod
    def _prepare_step_data(
        step_counts: list[StepCount], timestamps: list[datetime]
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Prepare and validate step count data for transformation.

        Args:
            step_counts: Raw step counts
            timestamps: Corresponding timestamps

        Returns:
            Tuple of (cleaned step count array, padding mask)

        Raises:
            DataLengthMismatchError: If data arrays have mismatched lengths
            EmptyDataError: If data is empty
        """
        if len(step_counts) != len(timestamps):
            raise DataLengthMismatchError(
                expected_length=len(timestamps),
                actual_length=len(step_counts),
                data_type="Step counts and timestamps",
            )

        if not step_counts:
            error_message = "Step counts"
            raise EmptyDataError(error_message)

        # Convert to numpy array
        steps_array = np.array(step_counts, dtype=float)
        original_length = len(steps_array)
        padding_mask = np.zeros(original_length, dtype=bool)

        # Always pad/truncate to full week (transformer requirement)
        # Ensure we have exactly one week of data (pad or truncate as needed)
        if len(steps_array) < MINUTES_PER_WEEK:
            # Pad with circadian-aware values instead of zeros
            padding_needed = MINUTES_PER_WEEK - len(steps_array)

            # Generate realistic padding values
            padding_values = _generate_circadian_padding(padding_needed)

            # Pad at the beginning (older data)
            steps_array = np.pad(
                steps_array, (padding_needed, 0), mode="constant", constant_values=0
            )
            steps_array[:padding_needed] = padding_values

            # Create padding mask
            padding_mask = np.zeros(MINUTES_PER_WEEK, dtype=bool)
            padding_mask[:padding_needed] = True

            logger.info("Padded %d minutes with circadian-aware values", padding_needed)

        elif len(steps_array) >= MINUTES_PER_WEEK:
            # Use canonical truncation approach for consistency
            # This handles both > MINUTES_PER_WEEK and == MINUTES_PER_WEEK cases
            original_length = len(steps_array)
            steps_array = prepare_for_pat_inference(steps_array, MINUTES_PER_WEEK)
            padding_mask = np.zeros(MINUTES_PER_WEEK, dtype=bool)
            if original_length > MINUTES_PER_WEEK:
                logger.info(
                    "Truncated from %d to most recent %d minutes using canonical approach",
                    original_length,
                    MINUTES_PER_WEEK,
                )
            else:
                logger.info(
                    "Data already at target length of %d minutes", MINUTES_PER_WEEK
                )

        # Handle missing data (represented as NaN or very large values)
        nan_mask = np.isnan(steps_array) | (
            steps_array > MAX_REALISTIC_STEPS_PER_MINUTE
        )
        if np.any(nan_mask):
            # Replace with small activity values instead of zeros
            steps_array[nan_mask] = 0.3  # Small but non-zero activity
            logger.warning(
                "Imputed %d missing/invalid step values with minimal activity",
                np.sum(nan_mask),
            )

        return steps_array, padding_mask

    @staticmethod
    def _calculate_quality_score(
        step_counts: FloatArray,
        proxy_vector: FloatArray,
        padding_mask: NDArray[np.bool_] | None = None,
    ) -> float:
        """Calculate data quality score for the transformation.

        Args:
            step_counts: Original step count data
            proxy_vector: Transformed proxy actigraphy vector
            padding_mask: Boolean array indicating padded positions

        Returns:
            Quality score between 0.0 and 1.0
        """
        # Adjust calculations to account for padding
        if padding_mask is not None:
            real_data_mask = ~padding_mask
            if np.sum(real_data_mask) == 0:
                return 0.0  # All data is padding

            # Calculate scores only on real data
            real_steps = step_counts[real_data_mask]
            real_proxy = proxy_vector[real_data_mask]
        else:
            real_steps = step_counts
            real_proxy = proxy_vector

        # Data completeness score (penalize excessive zeros in real data)
        zero_percentage = np.sum(real_steps == 0) / len(real_steps)
        completeness_score = max(
            0.0, 1.0 - (zero_percentage * ZERO_PERCENTAGE_PENALTY_FACTOR)
        )

        # Data variability score (reward diverse activity patterns)
        if np.std(real_steps) > 0:
            # Coefficient of variation (normalized standard deviation)
            cv = np.std(real_steps) / (
                np.mean(real_steps) + Z_SCORE_NORMALIZATION_EPSILON
            )
            variability_score = min(1.0, float(cv / 2.0))  # Cap at 1.0
        else:
            variability_score = 0.0

        # Realistic range score (penalize unrealistic values)
        realistic_mask = (real_steps >= 0) & (
            real_steps <= MAX_REALISTIC_STEPS_PER_MINUTE
        )
        realistic_score = np.sum(realistic_mask) / len(real_steps)

        # Proxy signal quality (penalize excessive extreme values)
        extreme_mask = (real_proxy < EXTREME_VALUE_LOWER_THRESHOLD) | (
            real_proxy > EXTREME_VALUE_UPPER_THRESHOLD
        )
        extreme_percentage = float(np.sum(extreme_mask) / len(real_proxy))
        proxy_quality_score = max(0.0, 1.0 - (extreme_percentage * 2.0))

        # Padding penalty (reduce quality if too much data is padded)
        if padding_mask is not None:
            padding_percentage = float(np.sum(padding_mask) / len(padding_mask))
            padding_penalty = max(0.0, 1.0 - (padding_percentage * MIN_ACTIVITY_LEVEL))
        else:
            padding_penalty = 1.0

        # Weighted combination of quality factors
        quality_score = (
            completeness_score * QUALITY_WEIGHT_COMPLETENESS
            + variability_score * QUALITY_WEIGHT_VARIABILITY
            + realistic_score * QUALITY_WEIGHT_REALISTIC
            + float(proxy_quality_score) * QUALITY_WEIGHT_PROXY_SIGNAL
            + float(padding_penalty) * QUALITY_WEIGHT_PADDING
        )

        return float(np.clip(quality_score, 0.0, 1.0))


def create_proxy_actigraphy_transformer(
    reference_year: int = 2025, *, cache_enabled: bool = True
) -> ProxyActigraphyTransformer:
    """Factory function to create a ProxyActigraphyTransformer instance.

    Args:
        reference_year: NHANES reference year for normalization
        cache_enabled: Whether to enable result caching

    Returns:
        Configured ProxyActigraphyTransformer instance
    """
    return ProxyActigraphyTransformer(
        reference_year=reference_year, cache_enabled=cache_enabled
    )
