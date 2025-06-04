"""Sleep Data Processor - Clinical-Grade Sleep Analysis.

This processor extracts comprehensive sleep metrics from Apple HealthKit sleep data,
providing research-grade features for sleep quality assessment and circadian analysis.

Features extracted align with clinical sleep medicine standards:
- Sleep efficiency (time asleep / time in bed)
- Sleep latency (time to fall asleep)
- WASO (Wake After Sleep Onset)
- Sleep stage architecture (REM%, Deep%)
- Sleep schedule consistency
- Overall sleep quality score

References sleep research standards from AASM, NSRR, and MESA studies.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from clarity.models.health_data import HealthMetric, SleepData, SleepStage

logger = logging.getLogger(__name__)

# Clinical sleep constants
IDEAL_SLEEP_HOURS = 8.0
MIN_SLEEP_EFFICIENCY = 0.85  # 85% considered good
MAX_HEALTHY_LATENCY = 20     # Minutes to fall asleep
MAX_HEALTHY_WASO = 30        # Minutes awake after sleep onset
CONSISTENCY_STD_THRESHOLD = 60  # Minutes std dev for good consistency
MIN_VALUES_FOR_CONSISTENCY = 2


class SleepFeatures(BaseModel):
    """Clinical-grade sleep features extracted from sleep stage data."""

    total_sleep_minutes: int = Field(
        ...,
        ge=0,
        le=720,  # Max 12 hours
        description="Total sleep duration in minutes"
    )

    sleep_efficiency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Sleep efficiency ratio (time asleep / time in bed)"
    )

    sleep_latency: float = Field(
        ...,
        ge=0.0,
        le=180.0,
        description="Sleep onset latency in minutes"
    )

    waso_minutes: float = Field(
        ...,
        ge=0.0,
        le=480.0,
        description="Wake After Sleep Onset in minutes"
    )

    awakenings_count: int = Field(
        ...,
        ge=0,
        le=50,
        description="Number of awakenings after sleep onset"
    )

    rem_percentage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="REM sleep as percentage of total sleep"
    )

    deep_percentage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Deep sleep as percentage of total sleep"
    )

    consistency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Sleep schedule consistency score (0=irregular, 1=regular)"
    )


class SleepProcessor:
    """Processor for Apple HealthKit sleep analysis data.
    
    Extracts robust sleep features following clinical sleep medicine standards
    and research best practices from AASM guidelines and MESA/NSRR datasets.
    
    Features:
    - Clinical sleep metrics (efficiency, latency, WASO)
    - Sleep architecture analysis (REM%, Deep%)
    - Circadian consistency scoring
    - Multi-night aggregation and averaging
    """

    def __init__(self) -> None:
        """Initialize the sleep processor."""
        self.processor_name = "SleepProcessor"
        self.version = "1.0.0"
        logger.info("✅ %s v%s initialized - Clinical sleep analysis ready",
                   self.processor_name, self.version)

    def process(self, metrics: list[HealthMetric]) -> SleepFeatures:
        """Process raw sleep metrics to compute comprehensive sleep features.
        
        Args:
            metrics: List of HealthMetric objects with SLEEP_ANALYSIS type
            
        Returns:
            SleepFeatures: Comprehensive sleep analysis results
        """
        try:
            logger.info("😴 Processing %d sleep metrics for analysis", len(metrics))

            # Extract sleep data from metrics
            sleep_data_list = self._extract_sleep_data(metrics)

            if not sleep_data_list:
                logger.warning("No valid sleep data found in metrics")
                return self._create_empty_features()

            # Calculate comprehensive sleep features
            features = self._calculate_sleep_features(sleep_data_list)

            logger.info("✅ Extracted sleep features for %d nights: "
                       "avg_sleep=%.1fh, efficiency=%.1f%%, quality_score=%.2f",
                       len(sleep_data_list),
                       features.total_sleep_minutes / 60,
                       features.sleep_efficiency * 100,
                       self._calculate_quality_score(features))

            return features

        except Exception as e:
            logger.exception("Failed to process sleep data")
            return self._create_empty_features()

    @staticmethod
    def _extract_sleep_data(metrics: list[HealthMetric]) -> list[SleepData]:
        """Extract valid sleep data from health metrics.
        
        Args:
            metrics: List of health metrics
            
        Returns:
            List of SleepData objects
        """
        sleep_data = []

        for metric in metrics:
            if metric.sleep_data is not None:
                sleep_data.append(metric.sleep_data)

        logger.debug("Extracted %d valid sleep data records", len(sleep_data))
        return sleep_data

    def _calculate_sleep_features(self, sleep_data_list: list[SleepData]) -> SleepFeatures:
        """Calculate comprehensive sleep features from sleep data.
        
        Args:
            sleep_data_list: List of sleep data records
            
        Returns:
            SleepFeatures: Computed sleep metrics
        """
        if not sleep_data_list:
            return self._create_empty_features()

        # Aggregate metrics across nights
        total_sleep_values = []
        efficiency_values = []
        latency_values = []
        waso_values = []
        awakening_values = []
        rem_percentage_values = []
        deep_percentage_values = []
        sleep_start_times = []

        for sleep_data in sleep_data_list:
            # Basic sleep metrics
            total_sleep_values.append(sleep_data.total_sleep_minutes)
            efficiency_values.append(sleep_data.sleep_efficiency)

            # Sleep latency
            latency = sleep_data.time_to_sleep_minutes if sleep_data.time_to_sleep_minutes is not None else 0.0
            latency_values.append(float(latency))

            # Awakenings
            awakenings = sleep_data.wake_count if sleep_data.wake_count is not None else 0
            awakening_values.append(int(awakenings))

            # WASO calculation
            waso = self._calculate_waso(sleep_data)
            waso_values.append(waso)

            # Sleep stage percentages
            rem_pct, deep_pct = self._calculate_stage_percentages(sleep_data)
            rem_percentage_values.append(rem_pct)
            deep_percentage_values.append(deep_pct)

            # Sleep timing for consistency
            sleep_start_times.append(sleep_data.sleep_start)

        # Calculate averages
        avg_total_sleep = int(round(np.mean(total_sleep_values)))
        avg_efficiency = float(np.mean(efficiency_values))
        avg_latency = float(np.mean(latency_values))
        avg_waso = float(np.mean(waso_values))
        avg_awakenings = int(round(np.mean(awakening_values)))
        avg_rem_percentage = float(np.mean(rem_percentage_values))
        avg_deep_percentage = float(np.mean(deep_percentage_values))

        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(sleep_start_times)

        return SleepFeatures(
            total_sleep_minutes=avg_total_sleep,
            sleep_efficiency=round(avg_efficiency, 3),
            sleep_latency=round(avg_latency, 1),
            waso_minutes=round(avg_waso, 1),
            awakenings_count=avg_awakenings,
            rem_percentage=round(avg_rem_percentage, 3),
            deep_percentage=round(avg_deep_percentage, 3),
            consistency_score=round(consistency_score, 3)
        )

    @staticmethod
    def _calculate_waso(sleep_data: SleepData) -> float:
        """Calculate Wake After Sleep Onset (WASO).
        
        Uses sleep stages if available, otherwise derives from timing.
        
        Args:
            sleep_data: Sleep data record
            
        Returns:
            WASO in minutes
        """
        if sleep_data.sleep_stages and SleepStage.AWAKE in sleep_data.sleep_stages:
            # Use awake minutes from sleep stages
            return float(sleep_data.sleep_stages[SleepStage.AWAKE])

        # Fallback: calculate from timing
        time_in_bed = (sleep_data.sleep_end - sleep_data.sleep_start).total_seconds() / 60.0
        latency = sleep_data.time_to_sleep_minutes if sleep_data.time_to_sleep_minutes is not None else 0.0
        waso = max(0.0, time_in_bed - sleep_data.total_sleep_minutes - latency)

        return waso

    @staticmethod
    def _calculate_stage_percentages(sleep_data: SleepData) -> tuple[float, float]:
        """Calculate REM and Deep sleep percentages.
        
        Args:
            sleep_data: Sleep data record
            
        Returns:
            Tuple of (rem_percentage, deep_percentage)
        """
        if not sleep_data.sleep_stages or sleep_data.total_sleep_minutes == 0:
            return 0.0, 0.0

        rem_minutes = sleep_data.sleep_stages.get(SleepStage.REM, 0)
        deep_minutes = sleep_data.sleep_stages.get(SleepStage.DEEP, 0)

        rem_percentage = rem_minutes / sleep_data.total_sleep_minutes
        deep_percentage = deep_minutes / sleep_data.total_sleep_minutes

        return rem_percentage, deep_percentage

    @staticmethod
    def _calculate_consistency_score(sleep_start_times: list[datetime]) -> float:
        """Calculate sleep schedule consistency score.
        
        Based on standard deviation of sleep start times across nights.
        Follows Sleep Regularity Index principles from circadian research.
        
        Args:
            sleep_start_times: List of sleep start timestamps
            
        Returns:
            Consistency score (0=irregular, 1=very regular)
        """
        if len(sleep_start_times) < MIN_VALUES_FOR_CONSISTENCY:
            return 0.5  # Neutral score for insufficient data

        # Convert to minutes of day (0-1439)
        start_minutes = []
        for dt in sleep_start_times:
            minutes_of_day = dt.hour * 60 + dt.minute
            start_minutes.append(minutes_of_day)

        # Calculate standard deviation
        std_minutes = float(np.std(start_minutes))

        # Map to 0-1 score: <15min = 1.0, >120min = 0.0, linear between
        if std_minutes <= 15:
            return 1.0
        if std_minutes >= 120:
            return 0.0
        return max(0.0, 1.0 - (std_minutes - 15) / (120 - 15))

    @staticmethod
    def _create_empty_features() -> SleepFeatures:
        """Create empty/zero sleep features for error cases.
        
        Returns:
            SleepFeatures with all zero values
        """
        return SleepFeatures(
            total_sleep_minutes=0,
            sleep_efficiency=0.0,
            sleep_latency=0.0,
            waso_minutes=0.0,
            awakenings_count=0,
            rem_percentage=0.0,
            deep_percentage=0.0,
            consistency_score=0.0
        )

    def get_summary_stats(self, features: SleepFeatures) -> dict[str, Any]:
        """Generate summary statistics for sleep features.
        
        Args:
            features: Computed sleep features
            
        Returns:
            Dictionary of summary statistics
        """
        quality_score = self._calculate_quality_score(features)

        return {
            "sleep_quality_score": round(quality_score, 3),
            "total_nights_analyzed": 1,  # Based on single feature set
            "avg_sleep_duration_hours": round(features.total_sleep_minutes / 60, 1),
            "sleep_efficiency_rating": self._rate_sleep_efficiency(features.sleep_efficiency),
            "sleep_latency_rating": self._rate_sleep_latency(features.sleep_latency),
            "waso_rating": self._rate_waso(features.waso_minutes),
            "rem_sleep_rating": self._rate_rem_percentage(features.rem_percentage),
            "deep_sleep_rating": self._rate_deep_percentage(features.deep_percentage),
            "consistency_rating": self._rate_consistency(features.consistency_score),
            "processor_version": self.version
        }

    @staticmethod
    def _calculate_quality_score(features: SleepFeatures) -> float:
        """Calculate overall sleep quality score (0-1).
        
        Combines multiple sleep metrics using clinical guidelines.
        
        Args:
            features: Sleep features
            
        Returns:
            Overall quality score
        """
        scores = []

        # Sleep efficiency (weight: 25%)
        eff_score = min(1.0, features.sleep_efficiency / MIN_SLEEP_EFFICIENCY)
        scores.append(eff_score * 0.25)

        # Sleep duration (weight: 20%) - optimal around 7-9 hours
        duration_hours = features.total_sleep_minutes / 60
        if 7 <= duration_hours <= 9:
            duration_score = 1.0
        elif duration_hours < 7:
            duration_score = max(0.0, duration_hours / 7)
        else:  # > 9 hours
            duration_score = max(0.0, 1.0 - (duration_hours - 9) / 3)
        scores.append(duration_score * 0.20)

        # Sleep latency (weight: 15%)
        latency_score = max(0.0, 1.0 - features.sleep_latency / MAX_HEALTHY_LATENCY)
        scores.append(latency_score * 0.15)

        # WASO (weight: 15%)
        waso_score = max(0.0, 1.0 - features.waso_minutes / MAX_HEALTHY_WASO)
        scores.append(waso_score * 0.15)

        # REM percentage (weight: 10%) - optimal 20-25%
        rem_optimal = 0.225  # 22.5%
        rem_score = max(0.0, 1.0 - abs(features.rem_percentage - rem_optimal) / rem_optimal)
        scores.append(rem_score * 0.10)

        # Deep percentage (weight: 10%) - optimal 15-20%
        deep_optimal = 0.175  # 17.5%
        deep_score = max(0.0, 1.0 - abs(features.deep_percentage - deep_optimal) / deep_optimal)
        scores.append(deep_score * 0.10)

        # Consistency (weight: 5%)
        scores.append(features.consistency_score * 0.05)

        return sum(scores)

    @staticmethod
    def _rate_sleep_efficiency(efficiency: float) -> str:
        """Rate sleep efficiency."""
        if efficiency >= 0.90:
            return "excellent"
        if efficiency >= 0.85:
            return "good"
        if efficiency >= 0.75:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_sleep_latency(latency: float) -> str:
        """Rate sleep latency."""
        if latency <= 10:
            return "excellent"
        if latency <= 20:
            return "good"
        if latency <= 30:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_waso(waso: float) -> str:
        """Rate WASO."""
        if waso <= 20:
            return "excellent"
        if waso <= 30:
            return "good"
        if waso <= 45:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_rem_percentage(rem_pct: float) -> str:
        """Rate REM percentage."""
        rem_percent = rem_pct * 100
        if 20 <= rem_percent <= 25:
            return "excellent"
        if 15 <= rem_percent <= 30:
            return "good"
        if 10 <= rem_percent <= 35:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_deep_percentage(deep_pct: float) -> str:
        """Rate deep sleep percentage."""
        deep_percent = deep_pct * 100
        if 15 <= deep_percent <= 20:
            return "excellent"
        if 10 <= deep_percent <= 25:
            return "good"
        if 5 <= deep_percent <= 30:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_consistency(consistency: float) -> str:
        """Rate sleep consistency."""
        if consistency >= 0.8:
            return "excellent"
        if consistency >= 0.6:
            return "good"
        if consistency >= 0.4:
            return "fair"
        return "poor"
