"""CardioProcessor - Heart Rate and HRV Analysis.

Extracts cardiovascular features from heart rate and heart rate variability data.
Implements domain-specific preprocessing and feature extraction for cardiac health.
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

# Constants for cardiovascular analysis
MIN_HEART_RATE = 30  # BPM - minimum physiological HR
MAX_HEART_RATE = 220  # BPM - maximum physiological HR
MIN_HRV = 5  # ms - minimum HRV for analysis
MAX_HRV = 200  # ms - maximum HRV for analysis

# Time constants
MIN_DATA_HOURS = 24  # hours - minimum data for reliable analysis
PERCENTILE_10 = 10  # 10th percentile for resting HR
PERCENTILE_25 = 25  # 25th percentile for recovery analysis
PERCENTILE_75 = 75  # 75th percentile for elevated HR

# Processing parameters
HR_RESAMPLE_INTERVAL = "1T"  # 1-minute intervals
HRV_RESAMPLE_INTERVAL = "5T"  # 5-minute intervals
HR_INTERPOLATION_LIMIT = 5  # max gap to interpolate
HRV_INTERPOLATION_LIMIT = 2  # max gap to interpolate
SMOOTHING_WINDOW = 3  # rolling window size

# Night and day hours for circadian analysis
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5]
DAY_HOURS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Circadian analysis constants
MIN_HOURLY_COVERAGE = 12  # minimum hours of coverage needed
EXPECTED_DAY_NIGHT_DIFF = 20.0  # BPM - normalization factor


class CardioFeatures(BaseModel):
    """Cardiovascular features extracted from HR/HRV data."""

    avg_hr: float = Field(description="Average heart rate (BPM)")
    max_hr: float = Field(description="Maximum heart rate (BPM)")
    resting_hr: float = Field(description="Estimated resting heart rate (BPM)")
    hr_variability: float = Field(description="Heart rate standard deviation")
    avg_hrv: float = Field(description="Average HRV SDNN (ms)")
    hrv_variability: float = Field(description="HRV standard deviation")
    hr_recovery_score: float = Field(description="Heart rate recovery indicator")
    circadian_rhythm_score: float = Field(description="Day/night HR pattern score")


class CardioProcessor:
    """Extract cardiovascular features from heart rate and HRV time series.

    Processes heart rate and heart rate variability data to extract meaningful
    cardiovascular health indicators including fitness, stress, and recovery metrics.
    """

    def __init__(self) -> None:
        """Initialize CardioProcessor."""
        self.logger = logging.getLogger(__name__)

    def process(
        self,
        hr_timestamps: list[datetime],
        hr_values: list[float],
        hrv_timestamps: list[datetime] | None = None,
        hrv_values: list[float] | None = None,
    ) -> list[float]:
        """Process heart rate and HRV data to extract cardiovascular features.

        Args:
            hr_timestamps: List of timestamps for HR samples
            hr_values: List of heart rate values (BPM)
            hrv_timestamps: Optional list of timestamps for HRV samples
            hrv_values: Optional list of HRV SDNN values (ms)

        Returns:
            List of 8 cardiovascular features
        """
        try:
            self.logger.info(
                "Processing cardiovascular data: %d HR samples", len(hr_values)
            )

            # Preprocess heart rate data
            hr_clean = self._preprocess_heart_rate(hr_timestamps, hr_values)

            # Preprocess HRV data if available
            hrv_clean = None
            if hrv_timestamps and hrv_values:
                hrv_clean = self._preprocess_hrv(hrv_timestamps, hrv_values)

            # Extract features
            features = self._extract_features(hr_clean, hrv_clean)

        except Exception:
            self.logger.exception("Error processing cardiovascular data")
            # Return zero vector on error
            return [0.0] * 8
        else:
            self.logger.info(
                "Extracted cardiovascular features: avg_hr=%.1f, resting_hr=%.1f",
                features.avg_hr,
                features.resting_hr,
            )

            # Return as list for fusion layer
            return [
                features.avg_hr,
                features.max_hr,
                features.resting_hr,
                features.hr_variability,
                features.avg_hrv,
                features.hrv_variability,
                features.hr_recovery_score,
                features.circadian_rhythm_score,
            ]

    @staticmethod
    def _preprocess_heart_rate(
        timestamps: list[datetime], values: list[float]
    ) -> pd.Series:
        """Clean and normalize heart rate time series."""
        if not timestamps or not values:
            return pd.Series(dtype=float)

        # Create pandas Series for resampling
        ts = pd.Series(values, index=pd.to_datetime(timestamps))

        # Resample to 1-minute frequency
        hr_per_min = ts.resample(HR_RESAMPLE_INTERVAL).mean()

        # Remove outliers (HR outside physiological range)
        hr_per_min = hr_per_min.mask(
            (hr_per_min <= MIN_HEART_RATE) | (hr_per_min > MAX_HEART_RATE), np.nan
        )

        # Fill short gaps by interpolation
        hr_interpolated = hr_per_min.interpolate(
            limit=HR_INTERPOLATION_LIMIT, limit_direction="forward"
        )

        # Apply smoothing
        hr_smoothed = hr_interpolated.rolling(
            window=SMOOTHING_WINDOW, min_periods=1, center=True
        ).mean()

        # Fill remaining NaNs with forward fill
        return hr_smoothed.ffill().bfill()

    @staticmethod
    def _preprocess_hrv(timestamps: list[datetime], values: list[float]) -> pd.Series:
        """Clean and normalize HRV time series."""
        if not timestamps or not values:
            return pd.Series(dtype=float)

        # Create pandas Series
        ts = pd.Series(values, index=pd.to_datetime(timestamps))

        # Resample to 5-minute frequency (HRV is typically less frequent)
        hrv_resampled = ts.resample(HRV_RESAMPLE_INTERVAL).mean()

        # Remove outliers (HRV outside physiological range)
        hrv_resampled = hrv_resampled.mask(
            (hrv_resampled <= MIN_HRV) | (hrv_resampled > MAX_HRV), np.nan
        )

        # Interpolate short gaps
        hrv_interpolated = hrv_resampled.interpolate(limit=HRV_INTERPOLATION_LIMIT)

        # Fill remaining NaNs
        return hrv_interpolated.ffill().bfill()

    def _extract_features(
        self,
        hr_series: pd.Series,
        hrv_series: pd.Series | None,
    ) -> CardioFeatures:
        """Extract cardiovascular features from cleaned time series."""
        # Basic HR statistics
        if len(hr_series) > 0:
            avg_hr = float(np.nanmean(hr_series))
            max_hr = float(np.nanmax(hr_series))
            resting_hr = float(
                np.nanpercentile(hr_series, PERCENTILE_10)
            )  # 10th percentile as resting
            hr_variability = float(np.nanstd(hr_series))
        else:
            avg_hr = max_hr = resting_hr = hr_variability = 0.0

        # HRV statistics
        if hrv_series is not None and len(hrv_series) > 0:
            avg_hrv = float(np.nanmean(hrv_series))
            hrv_variability = float(np.nanstd(hrv_series))
        else:
            avg_hrv = hrv_variability = 0.0

        # Advanced features
        hr_recovery_score = self._calculate_recovery_score(hr_series)
        circadian_rhythm_score = self._calculate_circadian_score(hr_series)

        return CardioFeatures(
            avg_hr=avg_hr,
            max_hr=max_hr,
            resting_hr=resting_hr,
            hr_variability=hr_variability,
            avg_hrv=avg_hrv,
            hrv_variability=hrv_variability,
            hr_recovery_score=hr_recovery_score,
            circadian_rhythm_score=circadian_rhythm_score,
        )

    @staticmethod
    def _calculate_recovery_score(hr_series: pd.Series) -> float:
        """Calculate heart rate recovery score (0-1, higher is better)."""
        if len(hr_series) < MIN_DATA_HOURS:  # Need at least 24 hours of data
            return 0.5  # Neutral score

        try:
            # Calculate ratio of resting periods to elevated periods
            resting_threshold = np.nanpercentile(hr_series, PERCENTILE_25)
            elevated_threshold = np.nanpercentile(hr_series, PERCENTILE_75)

            resting_periods = (hr_series <= resting_threshold).sum()
            elevated_periods = (hr_series >= elevated_threshold).sum()

            if elevated_periods == 0:
                return 1.0

            recovery_ratio = resting_periods / (resting_periods + elevated_periods)
            return float(np.clip(recovery_ratio, 0.0, 1.0))

        except (ValueError, ZeroDivisionError, TypeError):
            return 0.5

    @staticmethod
    def _calculate_circadian_score(hr_series: pd.Series) -> float:
        """Calculate circadian rhythm regularity score (0-1, higher is better)."""
        if len(hr_series) < MIN_DATA_HOURS:  # Need at least 24 hours
            return 0.5

        try:
            if not isinstance(hr_series.index, pd.DatetimeIndex):
                logger.warning(
                    "hr_series.index is not DatetimeIndex in _calculate_circadian_score. "
                    "Cannot calculate hourly means."
                )
                return 0.5  # Return default score if index type is not as expected

            # Group by hour of day and calculate consistency
            hourly_means = hr_series.groupby(hr_series.index.hour).mean()

            if len(hourly_means) < MIN_HOURLY_COVERAGE:  # Need reasonable coverage
                return 0.5

            # Calculate day/night difference (expect lower HR at night)
            night_hr = hourly_means[hourly_means.index.isin(NIGHT_HOURS)].mean()
            day_hr = hourly_means[hourly_means.index.isin(DAY_HOURS)].mean()

            if pd.isna(night_hr) or pd.isna(day_hr):
                return 0.5

            # Healthy circadian pattern: day HR > night HR
            day_night_diff = day_hr - night_hr

            # Normalize to 0-1 scale (expect 5-20 BPM difference)
            circadian_score = np.clip(
                day_night_diff / EXPECTED_DAY_NIGHT_DIFF, 0.0, 1.0
            )

            return float(circadian_score)

        except (ValueError, ZeroDivisionError, TypeError, KeyError):
            return 0.5
