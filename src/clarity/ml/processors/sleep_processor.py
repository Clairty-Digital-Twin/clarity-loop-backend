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

from datetime import datetime
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from clarity.models.health_data import HealthMetric, SleepData, SleepStage

logger = logging.getLogger(__name__)

# Clinical constants based on sleep medicine research
MIN_VALUES_FOR_CONSISTENCY = 2
CONSISTENCY_EXCELLENT_THRESHOLD = 15  # minutes
CONSISTENCY_POOR_THRESHOLD = 120  # minutes

# Sleep duration thresholds (hours)
OPTIMAL_SLEEP_MIN = 7
OPTIMAL_SLEEP_MAX = 9

# Sleep efficiency thresholds
EFFICIENCY_EXCELLENT = 0.90
EFFICIENCY_GOOD = 0.85
EFFICIENCY_FAIR = 0.75

# Sleep latency thresholds (minutes)
LATENCY_EXCELLENT = 10
LATENCY_GOOD = 20
LATENCY_FAIR = 30

# WASO thresholds (minutes)
WASO_EXCELLENT = 20
WASO_GOOD = 30
WASO_FAIR = 45

# REM sleep percentage thresholds
REM_OPTIMAL_MIN = 20
REM_OPTIMAL_MAX = 25
REM_GOOD_MIN = 15
REM_GOOD_MAX = 30
REM_FAIR_MIN = 10
REM_FAIR_MAX = 35

# Deep sleep percentage thresholds
DEEP_OPTIMAL_MIN = 15
DEEP_OPTIMAL_MAX = 20
DEEP_GOOD_MIN = 10
DEEP_GOOD_MAX = 25
DEEP_FAIR_MIN = 5
DEEP_FAIR_MAX = 30

# Consistency score thresholds
CONSISTENCY_EXCELLENT_SCORE = 0.8
CONSISTENCY_GOOD_SCORE = 0.6
CONSISTENCY_FAIR_SCORE = 0.4


class SleepFeatures(BaseModel):
    """Sleep analysis features extracted from sleep data."""

    total_sleep_minutes: int = Field(description="Total sleep time in minutes")
    sleep_efficiency: float = Field(description="Sleep efficiency (0-1)")
    sleep_latency: float = Field(description="Time to fall asleep in minutes")
    waso_minutes: float = Field(description="Wake After Sleep Onset in minutes")
    awakenings_count: int = Field(description="Number of nighttime awakenings")
    rem_percentage: float = Field(description="REM sleep percentage (0-1)")
    deep_percentage: float = Field(description="Deep sleep percentage (0-1)")
    consistency_score: float = Field(description="Sleep schedule consistency (0-1)")
    quality_score: float = Field(description="Overall sleep quality score (0-1)")


class SleepProcessor:
    """Processor for extracting clinical-grade sleep features from HealthKit data."""

    def __init__(self) -> None:
        """Initialize the sleep processor."""
        self.logger = logging.getLogger(__name__)

    def process(self, metrics: list[HealthMetric]) -> SleepFeatures:
        """Process sleep metrics and extract comprehensive features.

        Args:
            metrics: List of health metrics containing sleep data

        Returns:
            SleepFeatures: Comprehensive sleep analysis results
        """
        self.logger.info("Processing %d sleep metrics", len(metrics))

        try:
            # Extract and validate sleep data
            sleep_data_list = self._extract_sleep_data(metrics)

            if not sleep_data_list:
                self.logger.warning("No valid sleep data found")
                return self._create_empty_features()

            # Calculate comprehensive features
            features = self._calculate_sleep_features(sleep_data_list)

            # Add quality score
            features.quality_score = self._calculate_quality_score(features)

            self.logger.info("Successfully processed sleep data")

        except Exception:
            logger.exception("Failed to process sleep data")
            return self._create_empty_features()
        else:
            return features

    @staticmethod
    def _extract_sleep_data(metrics: list[HealthMetric]) -> list[SleepData]:
        """Extract valid sleep data from health metrics.

        Args:
            metrics: List of health metrics

        Returns:
            List of SleepData objects
        """
        sleep_data = [
            metric.sleep_data
            for metric in metrics
            if metric.sleep_data is not None
        ]

        logger.debug("Extracted %d valid sleep data records", len(sleep_data))
        return sleep_data

    def _calculate_sleep_features(self, sleep_data_list: list[SleepData]) -> SleepFeatures:
        """Calculate comprehensive sleep features from sleep data.

        Args:
            sleep_data_list: List of sleep data records

        Returns:
            SleepFeatures: Computed sleep metrics
        """
        # Initialize feature collections
        feature_sets = self._initialize_feature_collections()

        # Process each sleep record
        for sleep_data in sleep_data_list:
            self._process_sleep_record(sleep_data, feature_sets)

        # Calculate aggregated features
        return self._aggregate_features(feature_sets)

    @staticmethod
    def _initialize_feature_collections() -> dict[str, list[Any]]:
        """Initialize collections for feature aggregation."""
        return {
            "total_sleep": [],
            "efficiency": [],
            "latency": [],
            "waso": [],
            "awakenings": [],
            "rem_percentage": [],
            "deep_percentage": [],
            "start_times": []
        }

    def _process_sleep_record(
        self,
        sleep_data: SleepData,
        feature_sets: dict[str, list[Any]]
    ) -> None:
        """Process a single sleep record and add to feature collections."""
        feature_sets["total_sleep"].append(float(sleep_data.total_sleep_minutes))
        feature_sets["efficiency"].append(sleep_data.sleep_efficiency)
        feature_sets["latency"].append(float(sleep_data.time_to_sleep_minutes or 0))
        feature_sets["waso"].append(self._calculate_waso(sleep_data))
        feature_sets["awakenings"].append(float(sleep_data.wake_count or 0))

        # Calculate sleep stage percentages
        rem_pct, deep_pct = self._calculate_stage_percentages(sleep_data)
        feature_sets["rem_percentage"].append(rem_pct)
        feature_sets["deep_percentage"].append(deep_pct)
        feature_sets["start_times"].append(sleep_data.sleep_start)

    def _aggregate_features(self, feature_sets: dict[str, list[Any]]) -> SleepFeatures:
        """Aggregate individual features into final sleep features."""
        # Calculate averages for main metrics
        avg_total_sleep = round(np.mean(feature_sets["total_sleep"]))
        avg_efficiency = float(np.mean(feature_sets["efficiency"]))
        avg_latency = float(np.mean(feature_sets["latency"]))
        avg_waso = float(np.mean(feature_sets["waso"]))
        avg_awakenings = round(np.mean(feature_sets["awakenings"]))
        avg_rem_percentage = float(np.mean(feature_sets["rem_percentage"]))
        avg_deep_percentage = float(np.mean(feature_sets["deep_percentage"]))

        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(feature_sets["start_times"])

        return SleepFeatures(
            total_sleep_minutes=avg_total_sleep,
            sleep_efficiency=avg_efficiency,
            sleep_latency=avg_latency,
            waso_minutes=avg_waso,
            awakenings_count=avg_awakenings,
            rem_percentage=avg_rem_percentage,
            deep_percentage=avg_deep_percentage,
            consistency_score=consistency_score,
            quality_score=0.0  # Will be calculated separately
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
            return float(sleep_data.sleep_stages[SleepStage.AWAKE])

        # Derive from timing: time_in_bed - total_sleep - latency
        time_in_bed = (sleep_data.sleep_end - sleep_data.sleep_start).total_seconds() / 60
        latency = float(sleep_data.time_to_sleep_minutes or 0)

        return max(0.0, time_in_bed - sleep_data.total_sleep_minutes - latency)

    @staticmethod
    def _calculate_stage_percentages(sleep_data: SleepData) -> tuple[float, float]:
        """Calculate REM and Deep sleep percentages.

        Args:
            sleep_data: Sleep data record

        Returns:
            Tuple of (rem_percentage, deep_percentage)
        """
        if not sleep_data.sleep_stages:
            return 0.0, 0.0

        stages = sleep_data.sleep_stages
        total_sleep = sleep_data.total_sleep_minutes

        if total_sleep == 0:
            return 0.0, 0.0

        rem_minutes = stages.get(SleepStage.REM, 0)
        deep_minutes = stages.get(SleepStage.DEEP, 0)

        rem_percentage = float(rem_minutes) / float(total_sleep)
        deep_percentage = float(deep_minutes) / float(total_sleep)

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
            return 0.0

        # Convert to minutes since midnight for consistency calculation
        minutes_since_midnight = [
            (dt.hour * 60 + dt.minute) for dt in sleep_start_times
        ]

        # Calculate standard deviation
        std_minutes = float(np.std(minutes_since_midnight))

        # Map to 0-1 score: <15min = 1.0, >120min = 0.0, linear between
        if std_minutes <= CONSISTENCY_EXCELLENT_THRESHOLD:
            return 1.0
        if std_minutes >= CONSISTENCY_POOR_THRESHOLD:
            return 0.0
        return max(0.0, 1.0 - (std_minutes - CONSISTENCY_EXCELLENT_THRESHOLD) /
                   (CONSISTENCY_POOR_THRESHOLD - CONSISTENCY_EXCELLENT_THRESHOLD))

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
            consistency_score=0.0,
            quality_score=0.0
        )

    def get_summary_stats(self, features: SleepFeatures) -> dict[str, Any]:
        """Generate summary statistics for sleep features.

        Args:
            features: Computed sleep features

        Returns:
            Dictionary of summary statistics
        """
        return {
            "sleep_duration_hours": round(features.total_sleep_minutes / 60, 1),
            "sleep_efficiency_rating": self._rate_sleep_efficiency(features.sleep_efficiency),
            "sleep_latency_rating": self._rate_sleep_latency(features.sleep_latency),
            "waso_rating": self._rate_waso(features.waso_minutes),
            "rem_sleep_rating": self._rate_rem_percentage(features.rem_percentage),
            "deep_sleep_rating": self._rate_deep_percentage(features.deep_percentage),
            "consistency_rating": self._rate_consistency(features.consistency_score),
            "overall_quality_rating": self._rate_quality_score(features.quality_score)
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

        # Sleep duration (weight: 20%) - optimal around 7-9 hours
        duration_hours = features.total_sleep_minutes / 60
        if OPTIMAL_SLEEP_MIN <= duration_hours <= OPTIMAL_SLEEP_MAX:
            duration_score = 1.0
        elif duration_hours < OPTIMAL_SLEEP_MIN:
            duration_score = max(0.0, duration_hours / OPTIMAL_SLEEP_MIN)
        else:  # > 9 hours
            duration_score = max(0.0, 1.0 - (duration_hours - OPTIMAL_SLEEP_MAX) / 4)
        scores.append(duration_score * 0.20)

        # Sleep efficiency (weight: 25%)
        scores.append(features.sleep_efficiency * 0.25)

        # Sleep latency (weight: 15%) - inverse relationship
        latency_score = max(0.0, 1.0 - features.sleep_latency / 60)  # Normalize by 1 hour
        scores.append(latency_score * 0.15)

        # WASO (weight: 15%) - inverse relationship
        waso_score = max(0.0, 1.0 - features.waso_minutes / 120)  # Normalize by 2 hours
        scores.append(waso_score * 0.15)

        # REM sleep (weight: 10%) - optimal around 22.5%
        rem_optimal = 0.225
        rem_score = max(0.0, 1.0 - abs(features.rem_percentage - rem_optimal) / rem_optimal)

        # Deep sleep (weight: 10%) - optimal around 17.5%
        deep_optimal = 0.175
        deep_score = max(0.0, 1.0 - abs(features.deep_percentage - deep_optimal) / deep_optimal)

        scores.extend([
            rem_score * 0.10,
            deep_score * 0.10,
            features.consistency_score * 0.05
        ])

        return sum(scores)

    @staticmethod
    def _rate_sleep_efficiency(efficiency: float) -> str:
        """Rate sleep efficiency."""
        if efficiency >= EFFICIENCY_EXCELLENT:
            return "excellent"
        if efficiency >= EFFICIENCY_GOOD:
            return "good"
        if efficiency >= EFFICIENCY_FAIR:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_sleep_latency(latency: float) -> str:
        """Rate sleep latency."""
        if latency <= LATENCY_EXCELLENT:
            return "excellent"
        if latency <= LATENCY_GOOD:
            return "good"
        if latency <= LATENCY_FAIR:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_waso(waso: float) -> str:
        """Rate WASO."""
        if waso <= WASO_EXCELLENT:
            return "excellent"
        if waso <= WASO_GOOD:
            return "good"
        if waso <= WASO_FAIR:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_rem_percentage(rem_pct: float) -> str:
        """Rate REM percentage."""
        rem_percent = rem_pct * 100
        if REM_OPTIMAL_MIN <= rem_percent <= REM_OPTIMAL_MAX:
            return "excellent"
        if REM_GOOD_MIN <= rem_percent <= REM_GOOD_MAX:
            return "good"
        if REM_FAIR_MIN <= rem_percent <= REM_FAIR_MAX:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_deep_percentage(deep_pct: float) -> str:
        """Rate deep sleep percentage."""
        deep_percent = deep_pct * 100
        if DEEP_OPTIMAL_MIN <= deep_percent <= DEEP_OPTIMAL_MAX:
            return "excellent"
        if DEEP_GOOD_MIN <= deep_percent <= DEEP_GOOD_MAX:
            return "good"
        if DEEP_FAIR_MIN <= deep_percent <= DEEP_FAIR_MAX:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_consistency(consistency: float) -> str:
        """Rate sleep consistency."""
        if consistency >= CONSISTENCY_EXCELLENT_SCORE:
            return "excellent"
        if consistency >= CONSISTENCY_GOOD_SCORE:
            return "good"
        if consistency >= CONSISTENCY_FAIR_SCORE:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_quality_score(quality: float) -> str:
        """Rate overall quality score."""
        if quality >= 0.8:
            return "excellent"
        if quality >= 0.6:
            return "good"
        if quality >= 0.4:
            return "fair"
        return "poor"
