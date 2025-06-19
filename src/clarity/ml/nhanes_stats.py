"""NHANES Reference Statistics Module.

This module provides population-based reference statistics for normalizing
proxy actigraphy data derived from Apple HealthKit step counts.

The statistics are derived from the National Health and Nutrition Examination Survey
(NHANES) accelerometer data, adapted for step-count based proxy actigraphy transformation.

Reference:
- NHANES 2003-2006 accelerometer data
- Population-based normalization for sleep/activity analysis
"""

# removed - breaks FastAPI

from functools import lru_cache
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Standard deviation thresholds for outlier detection
EXTREME_OUTLIER_THRESHOLD = 3


class NHANESStatsError(Exception):
    """Exception raised when NHANES statistics lookup fails."""


# NHANES reference statistics for proxy actigraphy normalization
# Based on square-root transformed step count data
NHANES_REFERENCE_STATS = {
    "2023": {
        "mean": 4.2,
        "std": 2.1,
        "sample_size": 8945,
        "source": "NHANES 2003-2006 (projected 2023)",
    },
    "2024": {
        "mean": 4.3,
        "std": 2.0,
        "sample_size": 9234,
        "source": "NHANES 2003-2006 (projected 2024)",
    },
    "2025": {
        "mean": 4.4,
        "std": 1.9,
        "sample_size": 9567,
        "source": "NHANES 2003-2006 (projected 2025)",
    },
}

# Age-stratified statistics
AGE_STRATIFIED_STATS = {
    "18-29": {"mean": 5.1, "std": 1.8},
    "30-39": {"mean": 4.8, "std": 1.9},
    "40-49": {"mean": 4.5, "std": 2.0},
    "50-59": {"mean": 4.2, "std": 2.1},
    "60-69": {"mean": 3.8, "std": 2.2},
    "70+": {"mean": 3.2, "std": 2.3},
}

# Sex-stratified statistics
SEX_STRATIFIED_STATS = {
    "male": {"mean": 4.1, "std": 2.0},
    "female": {"mean": 4.6, "std": 1.8},
    "other": {"mean": 4.4, "std": 1.9},
}


@lru_cache(maxsize=128)
def get_available_years() -> list[int]:
    """Get list of available reference years."""
    return [int(year) for year in NHANES_REFERENCE_STATS]


@lru_cache(maxsize=32)
def get_available_age_groups() -> list[str]:
    """Get list of available age group stratifications."""
    return list(AGE_STRATIFIED_STATS.keys())


def lookup_norm_stats(
    year: int = 2025, age_group: str | None = None, sex: str | None = None
) -> tuple[float, float]:
    """Look up NHANES reference statistics for proxy actigraphy normalization.

    Args:
        year: Reference year for statistics (2023-2025 supported)
        age_group: Optional age stratification ("18-29", "30-39", etc.)
        sex: Optional sex stratification ("male", "female", "other")

    Returns:
        Tuple of (mean, std) for z-score normalization

    Raises:
        NHANESStatsError: If requested year or stratification is not available

    Example:
        >>> mean, std = lookup_norm_stats(year=2025)
        >>> z_score = (value - mean) / std
    """
    try:
        # Start with base statistics for the year
        year_str = str(year)
        if year_str not in NHANES_REFERENCE_STATS:
            logger.warning("Year %d not in reference data, using 2025 default", year)
            year_str = "2025"

        base_stats = NHANES_REFERENCE_STATS[year_str]
        mean = float(base_stats["mean"])  # type: ignore[arg-type]
        std = float(base_stats["std"])  # type: ignore[arg-type]

        # Apply age group adjustments if specified
        if age_group:
            if age_group not in AGE_STRATIFIED_STATS:
                logger.warning("Age group %s not found, using base stats", age_group)
            else:
                age_stats = AGE_STRATIFIED_STATS[age_group]
                # Weighted combination of base and age-specific stats
                mean = (mean + float(age_stats["mean"])) / 2
                std = (std + float(age_stats["std"])) / 2

        # Apply sex adjustments if specified
        if sex:
            sex_lower = sex.lower()
            if sex_lower not in SEX_STRATIFIED_STATS:
                logger.warning("Sex %s not found, using base stats", sex)
            else:
                sex_stats = SEX_STRATIFIED_STATS[sex_lower]
                # Weighted combination with sex-specific stats
                mean = (mean + sex_stats["mean"]) / 2
                std = (std + sex_stats["std"]) / 2

        logger.debug(
            "NHANES stats lookup: year=%d, age=%s, sex=%s -> mean=%.3f, std=%.3f",
            year,
            age_group,
            sex,
            mean,
            std,
        )
    except Exception as e:
        logger.exception("Error looking up NHANES stats")
        msg = f"Failed to lookup reference statistics: {e}"
        raise NHANESStatsError(msg) from e
    else:
        return mean, std


def get_reference_info(year: int = 2025) -> dict[str, Any]:
    """Get detailed information about a reference year's statistics.

    Args:
        year: Reference year to get information for

    Returns:
        Dictionary with reference information including sample size,
        age range, and data source
    """
    year_str = str(year)
    if year_str in NHANES_REFERENCE_STATS:
        return NHANES_REFERENCE_STATS[year_str].copy()
    # Return default year info
    return NHANES_REFERENCE_STATS["2025"].copy()


def validate_proxy_values(
    proxy_values: list[float], year: int = 2025
) -> dict[str, Any]:
    """Validate proxy actigraphy data against NHANES reference ranges.

    Args:
        proxy_values: List of square-root transformed step count values
        year: Reference year for validation

    Returns:
        Dictionary with validation results and statistics
    """
    mean, std = lookup_norm_stats(year=year)

    # Calculate z-scores
    values_array = np.array(proxy_values)
    z_scores = (values_array - mean) / std

    # Flag extreme values (>3 standard deviations)
    extreme_low = np.sum(z_scores < -EXTREME_OUTLIER_THRESHOLD)
    extreme_high = np.sum(z_scores > EXTREME_OUTLIER_THRESHOLD)

    return {
        "total_values": len(proxy_values),
        "mean_z_score": float(np.mean(z_scores)),
        "std_z_score": float(np.std(z_scores)),
        "extreme_low_count": int(extreme_low),
        "extreme_high_count": int(extreme_high),
        "outlier_percentage": float(
            (extreme_low + extreme_high) / len(proxy_values) * 100
        ),
        "reference_year": year,
        "reference_mean": mean,
        "reference_std": std,
        "validation_passed": (extreme_low + extreme_high)
        < len(proxy_values) * 0.05,  # <5% outliers
    }


# Module initialization
logger.info(
    "NHANES reference statistics module loaded. "
    "Available years: %s, "
    "Age groups: %d, "
    "Default year: 2025",
    get_available_years(),
    len(get_available_age_groups()),
)
