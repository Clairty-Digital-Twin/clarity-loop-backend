"""RespirationProcessor - Respiratory Rate and SpO2 Analysis.

Extracts respiratory features from breathing rate and oxygen saturation data.
Implements domain-specific preprocessing and feature extraction for respiratory health.
"""

from datetime import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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

        except Exception as e:
            self.logger.exception("Error processing respiratory data: %s", e)
            # Return default values on error
            return [
                16.0,
                14.0,
                2.0,
                98.0,
                95.0,
                1.0,
                0.5,
                0.8,
            ]  # Typical healthy values

    def _preprocess_respiratory_rate(
        self, timestamps: list[datetime], values: list[float]
    ) -> pd.Series:
        """Clean and normalize respiratory rate time series."""
        if not timestamps or not values:
            return pd.Series(dtype=float)

        # Create pandas Series for resampling
        ts = pd.Series(values, index=pd.to_datetime(timestamps))

        # Resample to 5-minute frequency (RR is typically less frequent than HR)
        rr_resampled = ts.resample("5T").mean()

        # Remove outliers (RR outside 5-60 breaths/min range)
        rr_resampled = rr_resampled.mask(
            (rr_resampled <= 5) | (rr_resampled > 60), np.nan
        )

        # Fill short gaps by interpolation (up to 3 periods = 15 minutes)
        rr_interpolated = rr_resampled.interpolate(limit=3, limit_direction="forward")

        # Apply light smoothing (3-period moving average)
        rr_smoothed = rr_interpolated.rolling(
            window=3, min_periods=1, center=True
        ).mean()

        # Fill remaining NaNs
        return rr_smoothed.fillna(method="ffill").fillna(method="bfill")

    def _preprocess_spo2(
        self, timestamps: list[datetime], values: list[float]
    ) -> pd.Series:
        """Clean and normalize SpO2 time series."""
        if not timestamps or not values:
            return pd.Series(dtype=float)

        # Create pandas Series
        ts = pd.Series(values, index=pd.to_datetime(timestamps))

        # Resample to 10-minute frequency (SpO2 is often periodic)
        spo2_resampled = ts.resample("10T").mean()

        # Remove outliers (SpO2 outside 80-100% range)
        spo2_resampled = spo2_resampled.mask(
            (spo2_resampled <= 80) | (spo2_resampled > 100), np.nan
        )

        # Interpolate short gaps
        spo2_interpolated = spo2_resampled.interpolate(limit=2)

        # Fill remaining NaNs
        return spo2_interpolated.fillna(method="ffill").fillna(method="bfill")

    def _extract_features(
        self, rr_series: pd.Series | None, spo2_series: pd.Series | None
    ) -> RespirationFeatures:
        """Extract respiratory features from cleaned time series."""
        # Respiratory rate statistics
        if rr_series is not None and len(rr_series) > 0:
            avg_respiratory_rate = float(np.nanmean(rr_series))
            resting_respiratory_rate = float(
                np.nanpercentile(rr_series, 25)
            )  # 25th percentile
            respiratory_variability = float(np.nanstd(rr_series))
        else:
            avg_respiratory_rate = 16.0  # Default healthy RR
            resting_respiratory_rate = 14.0
            respiratory_variability = 2.0

        # SpO2 statistics
        if spo2_series is not None and len(spo2_series) > 0:
            avg_spo2 = float(np.nanmean(spo2_series))
            min_spo2 = float(np.nanmin(spo2_series))
            spo2_variability = float(np.nanstd(spo2_series))
        else:
            avg_spo2 = 98.0  # Default healthy SpO2
            min_spo2 = 95.0
            spo2_variability = 1.0

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

    def _calculate_stability_score(self, rr_series: pd.Series | None) -> float:
        """Calculate respiratory stability score (0-1, higher is better)."""
        if rr_series is None or len(rr_series) < 12:  # Need at least 1 hour of data
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

        except Exception:
            return 0.5

    def _calculate_oxygenation_score(self, spo2_series: pd.Series | None) -> float:
        """Calculate oxygenation efficiency score (0-1, higher is better)."""
        if spo2_series is None or len(spo2_series) < 6:  # Need at least 1 hour of data
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

        except Exception:
            return 0.8
