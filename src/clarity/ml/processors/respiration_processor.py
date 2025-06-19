"""RespirationProcessor - Respiratory Rate and SpO2 Analysis.

Extracts respiratory features from breathing rate and oxygen saturation data.
Implements domain-specific preprocessing and feature extraction for respiratory health.
"""

# removed - breaks FastAPI

from datetime import datetime
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass  # Only for type stubs now

logger = logging.getLogger(__name__)

# Constants for respiratory analysis
MIN_RESPIRATORY_RATE = 5  # breaths/min - minimum physiological RR
MAX_RESPIRATORY_RATE = 60  # breaths/min - maximum physiological RR
MIN_SPO2 = 80  # % - minimum SpO2 for analysis
MAX_SPO2 = 100  # % - maximum physiological SpO2

# Default healthy values
DEFAULT_RR = 16.0  # breaths/min - normal respiratory rate
DEFAULT_RESTING_RR = 14.0  # breaths/min - normal resting RR
DEFAULT_RR_VARIABILITY = 2.0  # breaths/min - normal RR variability
DEFAULT_SPO2 = 98.0  # % - normal oxygen saturation
DEFAULT_MIN_SPO2 = 95.0  # % - normal minimum SpO2
DEFAULT_SPO2_VARIABILITY = 1.0  # % - normal SpO2 variability
DEFAULT_STABILITY_SCORE = 0.5  # neutral stability score
DEFAULT_EFFICIENCY_SCORE = 0.8  # good oxygenation score

# Processing parameters
RR_RESAMPLE_INTERVAL = "5T"  # 5-minute intervals for RR resampling
SPO2_RESAMPLE_INTERVAL = "10T"  # 10-minute intervals for SpO2 resampling
RR_INTERPOLATION_LIMIT = 3  # periods - max gap to interpolate for RR
SPO2_INTERPOLATION_LIMIT = 2  # periods - max gap to interpolate for SpO2
SMOOTHING_WINDOW = 3  # periods - rolling window for smoothing
PERCENTILE_25 = 25  # 25th percentile for resting rate calculation

# Stability analysis parameters
MIN_STABILITY_DATA_HOURS = 1  # minimum hours of data for stability analysis
STABILITY_DATA_POINTS = 12  # data points needed (1 hour at 5-min intervals)
OXYGENATION_DATA_POINTS = 6  # data points needed (1 hour at 10-min intervals)


class RespirationFeatures(BaseModel):
    """Respiratory features extracted from RR/SpO2 data."""

    avg_respiratory_rate: float = Field(
        description="Average respiratory rate (breaths/min)"
    )
    resting_respiratory_rate: float = Field(description="Resting respiratory rate")
    respiratory_variability: float = Field(description="Respiratory rate variability")
    avg_spo2: float = Field(description="Average oxygen saturation (%)")
    min_spo2: float = Field(description="Minimum oxygen saturation (%)")
    spo2_variability: float = Field(description="SpO2 standard deviation")
    respiratory_stability_score: float = Field(
        description="Breathing pattern stability (0-1)"
    )
    oxygenation_efficiency_score: float = Field(
        description="Oxygen efficiency score (0-1)"
    )


class RespirationProcessor:
    """Extract respiratory features from breathing rate and SpO2 time series.

    Processes respiratory rate and oxygen saturation data to extract meaningful
    respiratory health indicators including breathing efficiency and oxygenation.
    """

    def __init__(self) -> None:
        """Initialize RespirationProcessor."""
        self.logger = logging.getLogger(__name__)

    def process(
        self,
        rr_timestamps: list[datetime] | None = None,
        rr_values: list[float] | None = None,
        spo2_timestamps: list[datetime] | None = None,
        spo2_values: list[float] | None = None,
    ) -> list[float]:
        """Process respiratory rate and SpO2 data to extract respiratory features.

        Args:
            rr_timestamps: Optional list of timestamps for RR samples
            rr_values: Optional list of respiratory rate values (breaths/min)
            spo2_timestamps: Optional list of timestamps for SpO2 samples
            spo2_values: Optional list of SpO2 values (%)

        Returns:
            List of 8 respiratory features
        """
        try:
            self.logger.info(
                "Processing respiratory data: %d RR samples, %d SpO2 samples",
                len(rr_values) if rr_values else 0,
                len(spo2_values) if spo2_values else 0,
            )

            # Preprocess respiratory rate data
            rr_clean = None
            if rr_timestamps and rr_values:
                rr_clean = self._preprocess_respiratory_rate(rr_timestamps, rr_values)

            # Preprocess SpO2 data
            spo2_clean = None
            if spo2_timestamps and spo2_values:
                spo2_clean = self._preprocess_spo2(spo2_timestamps, spo2_values)

            # Extract features
            features = self._extract_features(rr_clean, spo2_clean)

            self.logger.info(
                "Extracted respiratory features: avg_rr=%.1f, avg_spo2=%.1f",
                features.avg_respiratory_rate,
                features.avg_spo2,
            )

        except Exception:
            self.logger.exception("Error processing respiratory data")
            # Return default values on error
            return [
                DEFAULT_RR,
                DEFAULT_RESTING_RR,
                DEFAULT_RR_VARIABILITY,
                DEFAULT_SPO2,
                DEFAULT_MIN_SPO2,
                DEFAULT_SPO2_VARIABILITY,
                DEFAULT_STABILITY_SCORE,
                DEFAULT_EFFICIENCY_SCORE,
            ]
        else:
            # Return as list for fusion layer
            return [
                features.avg_respiratory_rate,
                features.resting_respiratory_rate,
                features.respiratory_variability,
                features.avg_spo2,
                features.min_spo2,
                features.spo2_variability,
                features.respiratory_stability_score,
                features.oxygenation_efficiency_score,
            ]

    @staticmethod
    def _preprocess_respiratory_rate(
        timestamps: list[datetime], values: list[float]
    ) -> pd.Series:
        """Clean and normalize respiratory rate time series."""
        if not timestamps or not values:
            return pd.Series(dtype=float)

        # Create pandas Series for resampling
        ts = pd.Series(values, index=pd.to_datetime(timestamps))

        # Resample to 5-minute frequency (RR is typically less frequent than HR)
        rr_resampled = ts.resample(RR_RESAMPLE_INTERVAL).mean()

        # Remove outliers (RR outside physiological range)
        rr_resampled = rr_resampled.mask(
            (rr_resampled <= MIN_RESPIRATORY_RATE)
            | (rr_resampled > MAX_RESPIRATORY_RATE),
            np.nan,
        )

        # Fill short gaps by interpolation (up to 3 periods = 15 minutes)
        rr_interpolated = rr_resampled.interpolate(
            limit=RR_INTERPOLATION_LIMIT, limit_direction="forward"
        )

        # Apply light smoothing (3-period moving average)
        rr_smoothed = rr_interpolated.rolling(
            window=SMOOTHING_WINDOW, min_periods=1, center=True
        ).mean()

        # Fill remaining NaNs
        return rr_smoothed.ffill().bfill()

    @staticmethod
    def _preprocess_spo2(timestamps: list[datetime], values: list[float]) -> pd.Series:
        """Clean and normalize SpO2 time series."""
        if not timestamps or not values:
            return pd.Series(dtype=float)

        # Create pandas Series
        ts = pd.Series(values, index=pd.to_datetime(timestamps))

        # Resample to 10-minute frequency (SpO2 is often periodic)
        spo2_resampled = ts.resample(SPO2_RESAMPLE_INTERVAL).mean()

        # Remove outliers (SpO2 outside physiological range)
        spo2_resampled = spo2_resampled.mask(
            (spo2_resampled <= MIN_SPO2) | (spo2_resampled > MAX_SPO2), np.nan
        )

        # Interpolate short gaps
        spo2_interpolated = spo2_resampled.interpolate(limit=SPO2_INTERPOLATION_LIMIT)

        # Fill remaining NaNs
        return spo2_interpolated.ffill().bfill()

    def _extract_features(
        self,
        rr_series: pd.Series | None,
        spo2_series: pd.Series | None,
    ) -> RespirationFeatures:
        """Extract respiratory features from cleaned time series."""
        # Respiratory rate statistics
        if rr_series is not None and len(rr_series) > 0:
            avg_respiratory_rate = float(np.nanmean(rr_series))
            resting_respiratory_rate = float(np.nanpercentile(rr_series, PERCENTILE_25))
            respiratory_variability = float(np.nanstd(rr_series))
        else:
            avg_respiratory_rate = DEFAULT_RR
            resting_respiratory_rate = DEFAULT_RESTING_RR
            respiratory_variability = DEFAULT_RR_VARIABILITY

        # SpO2 statistics
        if spo2_series is not None and len(spo2_series) > 0:
            avg_spo2 = float(np.nanmean(spo2_series))
            min_spo2 = float(np.nanmin(spo2_series))
            spo2_variability = float(np.nanstd(spo2_series))
        else:
            avg_spo2 = DEFAULT_SPO2
            min_spo2 = DEFAULT_MIN_SPO2
            spo2_variability = DEFAULT_SPO2_VARIABILITY

        # Advanced features
        respiratory_stability_score = self._calculate_stability_score(rr_series)
        oxygenation_efficiency_score = self._calculate_oxygenation_score(spo2_series)

        return RespirationFeatures(
            avg_respiratory_rate=avg_respiratory_rate,
            resting_respiratory_rate=resting_respiratory_rate,
            respiratory_variability=respiratory_variability,
            avg_spo2=avg_spo2,
            min_spo2=min_spo2,
            spo2_variability=spo2_variability,
            respiratory_stability_score=respiratory_stability_score,
            oxygenation_efficiency_score=oxygenation_efficiency_score,
        )

    @staticmethod
    def _calculate_stability_score(rr_series: pd.Series | None) -> float:
        """Calculate respiratory stability score (0-1, higher is better)."""
        if rr_series is None or len(rr_series) < STABILITY_DATA_POINTS:
            return 0.5  # Neutral score

        try:
            # Calculate coefficient of variation (CV = std/mean)
            mean_rr = np.nanmean(rr_series)
            std_rr = np.nanstd(rr_series)

            if mean_rr == 0:
                return 0.5

            cv = std_rr / mean_rr

            # Lower CV indicates more stable breathing
            # Typical healthy CV for RR is 0.1-0.3
            stability_score = 1.0 - np.clip(cv / 0.4, 0.0, 1.0)

            return float(stability_score)

        except (ValueError, ZeroDivisionError, TypeError):
            return 0.5

    @staticmethod
    def _calculate_oxygenation_score(spo2_series: pd.Series | None) -> float:
        """Calculate oxygenation efficiency score (0-1, higher is better)."""
        if spo2_series is None or len(spo2_series) < OXYGENATION_DATA_POINTS:
            return 0.8  # Default good score

        try:
            mean_spo2 = np.nanmean(spo2_series)
            min_spo2 = np.nanmin(spo2_series)

            # Score based on average SpO2 and minimum SpO2
            # Excellent: avg >98%, min >95%
            # Good: avg >96%, min >92%
            # Fair: avg >94%, min >90%

            avg_score = np.clip((mean_spo2 - 94) / 4, 0.0, 1.0)  # 94-98% maps to 0-1
            min_score = np.clip((min_spo2 - 90) / 8, 0.0, 1.0)  # 90-98% maps to 0-1

            # Weighted combination (average is more important than minimum)
            oxygenation_score = 0.7 * avg_score + 0.3 * min_score

            return float(oxygenation_score)

        except (ValueError, ZeroDivisionError, TypeError):
            return 0.8
