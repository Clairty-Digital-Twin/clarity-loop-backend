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
"""

from datetime import datetime
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from clarity.ml.nhanes_stats import NHANESStats

logger = logging.getLogger(__name__)

# Constants for data processing
MINUTES_PER_WEEK = 10080  # 7 days * 24 hours * 60 minutes
MAX_REALISTIC_STEPS_PER_MINUTE = 1000  # Threshold for outlier detection

# Default NHANES statistics for fallback
DEFAULT_NHANES_STATS = {
    "2025": {
        "mean": 1.2,
        "std": 0.8,
        "source": "NHANES 2017-2020 (projected)"
    },
    "2024": {
        "mean": 1.15,
        "std": 0.75,
        "source": "NHANES 2017-2020"
    }
}


class StepCountData(BaseModel):
    """Input data structure for step count transformation."""

    user_id: str = Field(description="Unique user identifier")
    upload_id: str = Field(description="Unique upload session identifier")
    step_counts: list[float] = Field(
        description="Step counts per minute",
        min_length=1
    )
    timestamps: list[datetime] = Field(
        description="Corresponding timestamps for each step count"
    )
    user_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional user demographics and metadata"
    )


class ProxyActigraphyResult(BaseModel):
    """Result of proxy actigraphy transformation."""

    user_id: str = Field(description="User identifier")
    upload_id: str = Field(description="Upload session identifier")
    vector: list[float] = Field(description="Proxy actigraphy vector")
    quality_score: float = Field(
        description="Data quality score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    transformation_stats: dict[str, Any] = Field(
        description="Statistics about the transformation process"
    )
    nhanes_reference: dict[str, float] = Field(
        description="NHANES normalization parameters used"
    )


def _get_nhanes_stats_for_year(year: int) -> tuple[float, float]:
    """Get NHANES normalization statistics for a given year."""
    year_str = str(year)
    
    if year_str in DEFAULT_NHANES_STATS:
        stats = DEFAULT_NHANES_STATS[year_str]
        logger.info("Using %s for normalization", stats['source'])
        return stats["mean"], stats["std"]

    # Fallback to latest available year
    latest_year = max(DEFAULT_NHANES_STATS.keys())
    stats = DEFAULT_NHANES_STATS[latest_year]
    logger.warning("Year %d not found, using %s", year, stats['source'])
    return stats["mean"], stats["std"]


class ProxyActigraphyTransformer:
    """Main transformation engine for converting step counts to proxy actigraphy."""

    def __init__(self, reference_year: int = 2025, *, cache_enabled: bool = True) -> None:
        """Initialize the proxy actigraphy transformer.

        Args:
            reference_year: NHANES reference year for normalization
            cache_enabled: Whether to enable result caching
        """
        self.reference_year = reference_year
        self.cache_enabled = cache_enabled
        self._cache: dict[str, ProxyActigraphyResult] = {}

        # Load NHANES normalization parameters
        self.nhanes_mean, self.nhanes_std = NHANESStats.lookup_norm_stats(reference_year)

        logger.info("ProxyActigraphyTransformer initialized with NHANES %d", reference_year)
        logger.info("  • Reference mean: %.3f", self.nhanes_mean)
        logger.info("  • Reference std: %.3f", self.nhanes_std)
        logger.info("  • Cache enabled: %s", cache_enabled)

    def steps_to_movement_proxy(self, steps_per_min: np.ndarray) -> np.ndarray:
        """Convert step counts to movement proxy values.

        Applies square root transformation followed by z-score normalization
        using NHANES population statistics.

        Args:
            steps_per_min: Array of step counts per minute

        Returns:
            Normalized proxy actigraphy values
        """
        # Apply square root transformation for variance stabilization
        sqrt_steps = np.sqrt(np.maximum(steps_per_min, 0))

        # Z-score normalization using NHANES population statistics
        normalized = (sqrt_steps - self.nhanes_mean) / self.nhanes_std

        # Clip extreme values to reasonable range
        return np.clip(normalized, -5.0, 5.0)

    def transform_step_data(self, step_data: StepCountData) -> ProxyActigraphyResult:
        """Transform step count data to proxy actigraphy.

        Args:
            step_data: Input step count data

        Returns:
            Proxy actigraphy transformation result

        Raises:
            ValueError: If input data is invalid
        """
        # Generate cache key
        cache_key = f"{step_data.user_id}_{step_data.upload_id}_{len(step_data.step_counts)}"

        # Check cache if enabled
        if self.cache_enabled and cache_key in self._cache:
            logger.info("Returning cached transformation for %s", cache_key)
            return self._cache[cache_key]

        try:
            # Prepare and validate step data
            steps_array = self._prepare_step_data(
                step_data.step_counts,
                step_data.timestamps
            )

            # Transform to proxy actigraphy
            proxy_vector = self.steps_to_movement_proxy(steps_array)

            # Calculate quality score
            quality_score = self._calculate_quality_score(steps_array, proxy_vector)

            # Generate transformation statistics
            transformation_stats = {
                "input_length": len(step_data.step_counts),
                "output_length": len(proxy_vector),
                "zero_step_percentage": (np.sum(steps_array == 0) / len(steps_array)) * 100,
                "mean_steps_per_min": float(np.mean(steps_array)),
                "max_steps_per_min": float(np.max(steps_array)),
                "total_steps": float(np.sum(steps_array))
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
                    "std": self.nhanes_std
                }
            )

            # Cache result if enabled
            if self.cache_enabled:
                self._cache[cache_key] = result
                logger.debug("Cached transformation for %s", cache_key)

            logger.info("Successfully transformed step data for %s", step_data.user_id)
            logger.info("  • Quality score: %.3f", quality_score)
            logger.info("  • Vector length: %d", len(proxy_vector))
            logger.info("  • Zero steps: %.1f%%", transformation_stats['zero_step_percentage'])

            return result

        except Exception as e:
            logger.exception("Failed to transform step data for %s", step_data.user_id)
            raise

    @staticmethod
    def _prepare_step_data(
        step_counts: list[float],
        timestamps: list[datetime]
    ) -> np.ndarray:
        """Prepare and validate step count data for transformation.

        Args:
            step_counts: Raw step counts
            timestamps: Corresponding timestamps

        Returns:
            Cleaned and validated step count array

        Raises:
            ValueError: If data validation fails
        """
        if len(step_counts) != len(timestamps):
            msg = f"Step counts ({len(step_counts)}) and timestamps ({len(timestamps)}) length mismatch"
            raise ValueError(msg)

        if not step_counts:
            msg = "Step counts cannot be empty"
            raise ValueError(msg)

        # Convert to numpy array
        steps_array = np.array(step_counts, dtype=float)

        # Ensure we have exactly one week of data (pad or truncate as needed)
        if len(steps_array) < MINUTES_PER_WEEK:
            # Pad with zeros at the beginning (older data)
            padding_needed = MINUTES_PER_WEEK - len(steps_array)
            steps_array = np.pad(steps_array, (padding_needed, 0), mode='constant', constant_values=0)
            logger.warning("Padded %d minutes with zeros", padding_needed)

        elif len(steps_array) > MINUTES_PER_WEEK:
            # Take the most recent week of data
            steps_array = steps_array[-MINUTES_PER_WEEK:]
            logger.info("Truncated to most recent %d minutes", MINUTES_PER_WEEK)

        # Handle missing data (represented as NaN or very large values)
        nan_mask = np.isnan(steps_array) | (steps_array > MAX_REALISTIC_STEPS_PER_MINUTE)
        if np.any(nan_mask):
            # Simple imputation: replace with median of surrounding values
            steps_array[nan_mask] = 0  # Conservative approach for missing data
            logger.warning("Imputed %d missing/invalid step values", np.sum(nan_mask))

        return steps_array

    @staticmethod
    def _calculate_quality_score(
        step_counts: np.ndarray,
        proxy_vector: np.ndarray  # noqa: ARG004
    ) -> float:
        """Calculate data quality score for the transformation.

        Args:
            step_counts: Original step count data
            proxy_vector: Transformed proxy actigraphy vector

        Returns:
            Quality score between 0.0 and 1.0
        """
        # Data completeness score (penalize missing/zero data)
        zero_percentage = np.sum(step_counts == 0) / len(step_counts)
        completeness_score = max(0.0, 1.0 - (zero_percentage * 2))  # Penalize heavily

        # Data variability score (reward diverse activity patterns)
        if np.std(step_counts) > 0:
            # Coefficient of variation (normalized standard deviation)
            cv = np.std(step_counts) / (np.mean(step_counts) + 1e-6)
            variability_score = min(1.0, cv / 2.0)  # Cap at 1.0
        else:
            variability_score = 0.0

        # Realistic range score (penalize unrealistic values)
        realistic_mask = (step_counts >= 0) & (step_counts <= MAX_REALISTIC_STEPS_PER_MINUTE)
        realistic_score = np.sum(realistic_mask) / len(step_counts)

        # Circadian pattern score (reward day/night activity differences)
        if len(step_counts) >= 1440:  # At least 24 hours
            try:
                # Reshape to days x minutes_per_day for analysis
                days = len(step_counts) // 1440
                daily_data = step_counts[:days * 1440].reshape(days, 1440)

                # Calculate day vs night activity
                day_hours = daily_data[:, 360:1200]  # 6 AM to 8 PM
                night_hours = daily_data[:, np.r_[0:360, 1200:1440]]  # 8 PM to 6 AM

                day_activity = np.mean(day_hours)
                night_activity = np.mean(night_hours)

                if day_activity + night_activity > 0:
                    circadian_score = min(1.0, (day_activity - night_activity) / (day_activity + night_activity + 1e-6))
                    circadian_score = max(0.0, circadian_score)  # Ensure non-negative
                else:
                    circadian_score = 0.5  # Neutral if can't calculate
            except Exception:  # noqa: BLE001
                circadian_score = 0.5
        else:
            circadian_score = 0.5  # Neutral for short data

        # Weighted combination of quality factors
        quality_score = (
            completeness_score * 0.4 +
            variability_score * 0.3 +
            realistic_score * 0.2 +
            circadian_score * 0.1
        )

        return float(np.clip(quality_score, 0.0, 1.0))


def create_proxy_actigraphy_transformer(
    reference_year: int = 2025,
    *,
    cache_enabled: bool = True
) -> ProxyActigraphyTransformer:
    """Factory function to create a ProxyActigraphyTransformer instance.

    Args:
        reference_year: NHANES reference year for normalization
        cache_enabled: Whether to enable result caching

    Returns:
        Configured ProxyActigraphyTransformer instance
    """
    return ProxyActigraphyTransformer(
        reference_year=reference_year,
        cache_enabled=cache_enabled
    )
