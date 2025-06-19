"""Apple Watch Data Processor for Clarity Digital Twin.

Specialized processing and transformation of Apple Watch data
for optimal integration with PAT (Pretrained Actigraphy Transformer) models.
"""

# removed - breaks FastAPI

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import logging
import operator
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from scipy.interpolate import interp1d
import scipy.signal

from clarity.core.exceptions import ProcessingError
from clarity.integrations.healthkit import HealthDataBatch, HealthDataPoint

if TYPE_CHECKING:
    pass  # Only for type stubs now

logger = logging.getLogger(__name__)


class ActivityLevel(StrEnum):
    """Apple Watch activity levels."""

    SEDENTARY = "sedentary"
    LIGHT = "light"
    MODERATE = "moderate"
    VIGOROUS = "vigorous"


class SleepStage(StrEnum):
    """Sleep stages from Apple Watch."""

    AWAKE = "awake"
    REM = "rem"
    CORE = "core"
    DEEP = "deep"


@dataclass
class ProcessedHealthData:
    """Processed health data ready for ML models."""

    # Time series data (minute-level, 10080 points for a week)
    heart_rate_series: npt.NDArray[np.floating[Any]] | None = None
    hrv_series: npt.NDArray[np.floating[Any]] | None = None
    respiratory_rate_series: npt.NDArray[np.floating[Any]] | None = None
    movement_proxy_vector: npt.NDArray[np.floating[Any]] | None = None  # For PAT model

    # Summary statistics
    resting_hr: float | None = None
    max_hr: float | None = None
    avg_hr: float | None = None
    hrv_median: float | None = None
    avg_respiratory_rate: float | None = None
    avg_spo2: float | None = None
    min_spo2: float | None = None

    # Blood pressure (if available)
    systolic_bp: float | None = None
    diastolic_bp: float | None = None

    # Temperature deviation
    avg_temp_deviation: float | None = None

    # VO2 Max
    vo2_max: float | None = None
    vo2_max_trend: float | None = None  # Change from previous

    # Workout summary
    workout_count: int = 0
    total_active_minutes: int = 0
    avg_workout_intensity: float | None = None

    # ECG
    afib_detected: bool = False
    ecg_classification: str | None = None

    # Metadata
    start_time: datetime | None = None
    end_time: datetime | None = None
    data_completeness: float = 0.0  # Percentage of expected data points


class AppleWatchDataProcessor:
    """Processes raw Apple Watch health data into ML-ready formats.

    Implements state-of-the-art preprocessing techniques including:
    - Outlier removal using physiological bounds
    - Time series resampling and alignment
    - Signal filtering (Butterworth for HR, median for RR)
    - Z-score normalization
    - Missing data interpolation
    """

    # NHANES population statistics for normalization
    NHANES_STEP_MEAN = 7.49  # sqrt(steps) mean
    NHANES_STEP_STD = 7.71  # sqrt(steps) std

    # Physiological bounds for validation
    HR_MIN = 30
    HR_MAX = 220
    HRV_MIN = 0
    HRV_MAX = 300  # ms
    RR_MIN = 5
    RR_MAX = 60
    SPO2_MIN = 80
    SPO2_MAX = 100
    BP_SYSTOLIC_MIN = 70
    BP_SYSTOLIC_MAX = 200
    BP_DIASTOLIC_MIN = 40
    BP_DIASTOLIC_MAX = 130
    TEMP_DEVIATION_MAX = 5  # °C
    VO2_MAX_MIN = 10
    VO2_MAX_MAX = 90

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    async def process_health_batch(
        self, batch: HealthDataBatch, target_duration_days: int = 7
    ) -> ProcessedHealthData:
        """Process a batch of health data into ML-ready format.

        Args:
            batch: Raw health data batch from HealthKit
            target_duration_days: Target duration for time series (default 7 days)

        Returns:
            ProcessedHealthData with all modalities processed and aligned
        """
        try:
            # Determine time range
            end_time = batch.end_time or datetime.now(UTC)
            start_time = end_time - timedelta(days=target_duration_days)

            # Initialize result
            result = ProcessedHealthData(start_time=start_time, end_time=end_time)

            # Process each modality
            if batch.heart_rate_samples:
                await self._process_heart_rate(
                    batch.heart_rate_samples, result, start_time, end_time
                )

            if batch.hrv_samples:
                await self._process_hrv(batch.hrv_samples, result, start_time, end_time)

            if batch.respiratory_rate_samples:
                await self._process_respiratory_rate(
                    batch.respiratory_rate_samples, result, start_time, end_time
                )

            if batch.step_count_samples:
                await self._process_steps(
                    batch.step_count_samples, result, start_time, end_time
                )

            if batch.blood_oxygen_samples:
                await self._process_spo2(batch.blood_oxygen_samples, result)

            if batch.blood_pressure_samples:
                await self._process_blood_pressure(batch.blood_pressure_samples, result)

            if batch.body_temperature_samples:
                await self._process_temperature(batch.body_temperature_samples, result)

            if batch.vo2_max_samples:
                await self._process_vo2_max(batch.vo2_max_samples, result)

            if batch.workout_samples:
                await self._process_workouts(batch.workout_samples, result)

            if batch.electrocardiogram_samples:
                await self._process_ecg(batch.electrocardiogram_samples, result)

        except Exception as e:
            self.logger.exception("Error processing health batch")
            msg = f"Failed to process health data: {e!s}"
            raise ProcessingError(msg) from e
        else:
            # Calculate data completeness
            result.data_completeness = self._calculate_completeness(result)
            return result

    async def _process_heart_rate(
        self,
        samples: list[HealthDataPoint],
        result: ProcessedHealthData,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Process heart rate data with advanced filtering."""
        # Extract time series
        times = np.array([s.timestamp.timestamp() for s in samples])
        values = np.array([s.value for s in samples])

        # Remove outliers using physiological bounds
        mask = (values >= self.HR_MIN) & (values <= self.HR_MAX)
        times = times[mask]
        values = values[mask]

        if len(values) == 0:
            return

        # Resample to minute-level (1440 points per day)
        minute_timestamps = np.arange(
            start_time.timestamp(),
            end_time.timestamp(),
            60,  # 60 seconds
        )

        # Interpolate to regular grid
        if len(values) > 1:
            # Use linear interpolation for gaps < 5 minutes
            f = interp1d(
                times, values, kind="linear", bounds_error=False, fill_value=np.nan
            )
            minute_values = f(minute_timestamps)

            # Forward fill for small gaps (< 5 minutes)
            minute_values = pd.Series(minute_values).ffill(limit=5).to_numpy()

            # Apply Butterworth low-pass filter to remove motion artifacts
            # Cutoff at 0.5 Hz (30 bpm variation)
            if not np.all(np.isnan(minute_values)):
                b, a = scipy.signal.butter(4, 0.5 / (0.5 * 60), btype="low")
                valid_mask = ~np.isnan(minute_values)
                min_filter_points = 12
                if (
                    np.sum(valid_mask) > min_filter_points
                ):  # Need at least 12 points for filter
                    minute_values[valid_mask] = scipy.signal.filtfilt(
                        b, a, minute_values[valid_mask]
                    )
        else:
            minute_values = np.full(len(minute_timestamps), values[0])

        # Calculate summary statistics
        valid_values = minute_values[~np.isnan(minute_values)]
        if len(valid_values) > 0:
            result.avg_hr = float(np.mean(valid_values))
            result.max_hr = float(np.max(valid_values))
            # Resting HR as 5th percentile
            result.resting_hr = float(np.percentile(valid_values, 5))

        # Z-score normalization
        if len(valid_values) > 1:
            hr_mean = 70  # Population average
            hr_std = 12  # Population std
            minute_values = (minute_values - hr_mean) / hr_std

        result.heart_rate_series = minute_values

    async def _process_hrv(
        self,
        samples: list[HealthDataPoint],
        result: ProcessedHealthData,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Process HRV data with outlier removal and normalization."""
        # Extract time series
        times = np.array([s.timestamp.timestamp() for s in samples])
        values = np.array([s.value for s in samples])  # SDNN in ms

        # Remove outliers (5x median filter)
        min_outlier_points = 3
        if len(values) > min_outlier_points:
            median_val = np.median(values)
            mask = values < 5 * median_val
            times = times[mask]
            values = values[mask]

        if len(values) == 0:
            return

        # Resample to minute-level
        minute_timestamps = np.arange(start_time.timestamp(), end_time.timestamp(), 60)

        # Interpolate (HRV changes slowly, so interpolation is reasonable)
        if len(values) > 1:
            f = interp1d(
                times, values, kind="linear", bounds_error=False, fill_value=np.nan
            )
            minute_values = f(minute_timestamps)

            # Fill small gaps only
            minute_values = pd.Series(minute_values).ffill(limit=10).to_numpy()
        else:
            minute_values = np.full(len(minute_timestamps), np.nan)

        # Calculate summary statistics
        valid_values = minute_values[~np.isnan(minute_values)]
        if len(valid_values) > 0:
            result.hrv_median = float(np.median(valid_values))

        # Log transform for normalization (HRV is often log-normal)
        if len(valid_values) > 0:
            minute_values = np.log1p(minute_values)  # log(1 + x) to handle zeros

        result.hrv_series = minute_values

    async def _process_respiratory_rate(
        self,
        samples: list[HealthDataPoint],
        result: ProcessedHealthData,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Process respiratory rate with median filtering."""
        # Extract time series
        times = np.array([s.timestamp.timestamp() for s in samples])
        values = np.array([s.value for s in samples])

        # Remove physiologically implausible values
        mask = (values >= self.RR_MIN) & (values <= self.RR_MAX)
        times = times[mask]
        values = values[mask]

        if len(values) == 0:
            return

        # Resample to 5-minute intervals (respiratory rate changes slowly)
        five_min_timestamps = np.arange(
            start_time.timestamp(),
            end_time.timestamp(),
            300,  # 5 minutes
        )

        if len(values) > 1:
            # Interpolate
            f = interp1d(
                times, values, kind="linear", bounds_error=False, fill_value=np.nan
            )
            five_min_values = f(five_min_timestamps)

            # Apply 3-point median filter to remove transient spikes
            min_median_points = 3
            if len(five_min_values) > min_median_points:
                five_min_values = scipy.signal.medfilt(five_min_values, kernel_size=3)
        else:
            five_min_values = np.full(len(five_min_timestamps), values[0])

        # Calculate average
        valid_values = five_min_values[~np.isnan(five_min_values)]
        if len(valid_values) > 0:
            result.avg_respiratory_rate = float(np.mean(valid_values))

        # Z-score normalization
        rr_mean = 15  # Population average
        rr_std = 3  # Population std
        five_min_values = (five_min_values - rr_mean) / rr_std

        # Upsample to minute-level for consistency
        minute_timestamps = np.arange(start_time.timestamp(), end_time.timestamp(), 60)
        if len(five_min_values) > 1:
            f = interp1d(
                five_min_timestamps,
                five_min_values,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            minute_values = f(minute_timestamps)
        else:
            minute_values = np.full(len(minute_timestamps), np.nan)

        result.respiratory_rate_series = minute_values

    async def _process_steps(
        self,
        samples: list[HealthDataPoint],
        result: ProcessedHealthData,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Process steps into PAT-compatible movement proxy vector."""
        # Create minute-level step array
        minutes_in_week = 7 * 24 * 60
        minute_steps = np.zeros(minutes_in_week)

        # Aggregate steps by minute
        for sample in samples:
            if start_time <= sample.timestamp <= end_time:
                minute_idx = int((sample.timestamp - start_time).total_seconds() / 60)
                if 0 <= minute_idx < minutes_in_week:
                    minute_steps[minute_idx] += sample.value

        # Apply sqrt transformation (variance stabilization)
        movement_vector = np.sqrt(minute_steps)

        # Z-score using NHANES population statistics
        movement_vector = (
            movement_vector - self.NHANES_STEP_MEAN
        ) / self.NHANES_STEP_STD

        result.movement_proxy_vector = movement_vector

        # Log weekly step summary
        total_steps = np.sum(minute_steps)
        # Calculate days from time range
        days = (
            (result.end_time - result.start_time).days
            if result.start_time and result.end_time
            else 7
        )
        self.logger.info(
            "Processed steps over period",
            extra={"total_steps": int(total_steps), "days": days},
        )

    async def _process_spo2(
        self, samples: list[HealthDataPoint], result: ProcessedHealthData
    ) -> None:
        """Process SpO2 data (sparse samples)."""
        values = [
            float(s.value)
            for s in samples
            if isinstance(s.value, (int, float))
            and self.SPO2_MIN <= s.value <= self.SPO2_MAX
        ]

        if values:
            result.avg_spo2 = float(np.mean(values))
            result.min_spo2 = float(np.min(values))

    async def _process_blood_pressure(
        self, samples: list[HealthDataPoint], result: ProcessedHealthData
    ) -> None:
        """Process blood pressure (episodic data)."""
        systolic_values = []
        diastolic_values = []

        for sample in samples:
            # Check if sample has required blood pressure attributes
            if not (hasattr(sample, "systolic") and hasattr(sample, "diastolic")):
                continue

            # Check if values are not None and within valid ranges
            if (
                sample.systolic is not None
                and sample.diastolic is not None
                and self.BP_SYSTOLIC_MIN <= sample.systolic <= self.BP_SYSTOLIC_MAX
                and self.BP_DIASTOLIC_MIN <= sample.diastolic <= self.BP_DIASTOLIC_MAX
            ):
                systolic_values.append(sample.systolic)
                diastolic_values.append(sample.diastolic)

        if systolic_values:
            # Use most recent or average
            last_systolic = systolic_values[-1]
            last_diastolic = diastolic_values[-1]
            if last_systolic is not None and last_diastolic is not None:
                result.systolic_bp = float(last_systolic)  # Most recent
                result.diastolic_bp = float(last_diastolic)

    async def _process_temperature(
        self, samples: list[HealthDataPoint], result: ProcessedHealthData
    ) -> None:
        """Process temperature deviation data."""
        deviations = [
            float(s.value)
            for s in samples
            if isinstance(s.value, (int, float))
            and abs(s.value) <= self.TEMP_DEVIATION_MAX
        ]

        if deviations:
            result.avg_temp_deviation = float(np.mean(deviations))

    async def _process_vo2_max(
        self, samples: list[HealthDataPoint], result: ProcessedHealthData
    ) -> None:
        """Process VO2 max data."""
        values = [
            (s.timestamp, float(s.value))
            for s in samples
            if isinstance(s.value, (int, float))
            and self.VO2_MAX_MIN <= s.value <= self.VO2_MAX_MAX
        ]

        if values:
            # Sort by time
            values.sort(key=operator.itemgetter(0))
            result.vo2_max = float(values[-1][1])  # Most recent

            # Calculate trend if we have history
            if len(values) > 1:
                # Change from second-to-last to last
                result.vo2_max_trend = float(values[-1][1] - values[-2][1])

    async def _process_workouts(
        self, samples: list[Any], result: ProcessedHealthData
    ) -> None:
        """Process workout data."""
        result.workout_count = len(samples)

        total_minutes = 0
        total_intensity = 0

        for workout in samples:
            if hasattr(workout, "duration_minutes"):
                total_minutes += workout.duration_minutes
            if hasattr(workout, "avg_heart_rate") and result.resting_hr:
                # Calculate intensity as % of HR reserve
                hr_reserve = (workout.avg_heart_rate - result.resting_hr) / (
                    220 - workout.age - result.resting_hr
                )
                total_intensity += hr_reserve

        result.total_active_minutes = total_minutes
        if result.workout_count > 0:
            result.avg_workout_intensity = total_intensity / result.workout_count

    async def _process_ecg(
        self, samples: list[Any], result: ProcessedHealthData
    ) -> None:
        """Process ECG classifications."""
        for ecg in samples:
            if hasattr(ecg, "classification"):
                result.ecg_classification = ecg.classification
                if ecg.classification == "Atrial Fibrillation":
                    result.afib_detected = True
                    break

    @staticmethod
    def _calculate_completeness(result: ProcessedHealthData) -> float:
        """Calculate percentage of data completeness."""
        expected_fields = [
            result.heart_rate_series,
            result.movement_proxy_vector,
            result.hrv_series,
            result.respiratory_rate_series,
        ]

        completeness_scores = []

        for field in expected_fields:
            if field is not None and isinstance(field, np.ndarray):
                # Calculate percentage of non-NaN values
                valid_ratio = float(np.sum(~np.isnan(field)) / len(field))
                completeness_scores.append(valid_ratio)
            else:
                completeness_scores.append(0.0)

        return float(np.mean(completeness_scores) * 100)  # As percentage
