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

# removed - breaks FastAPI

import logging

import numpy as np
from pydantic import BaseModel, Field

from clarity.models.health_data import HealthMetric, SleepData, SleepStage

logger = logging.getLogger(__name__)

# Clinical thresholds based on sleep medicine research
MIN_VALUES_FOR_CONSISTENCY = 3  # Need 3+ nights for consistency analysis
CONSISTENCY_STD_THRESHOLD = 30.0  # Minutes - 30+ min std = inconsistent
NOON_HOUR_THRESHOLD = 12  # Hours - for sleep time normalization

# Sleep efficiency rating thresholds
EFFICIENCY_EXCELLENT = 0.9
EFFICIENCY_GOOD = 0.8
EFFICIENCY_FAIR = 0.7

# Sleep latency rating thresholds (minutes)
LATENCY_EXCELLENT = 15
LATENCY_GOOD = 30
LATENCY_FAIR = 45

# WASO rating thresholds (minutes)
WASO_EXCELLENT = 20
WASO_GOOD = 40
WASO_FAIR = 60

# REM percentage rating thresholds
REM_EXCELLENT_MIN = 0.18
REM_EXCELLENT_MAX = 0.25
REM_GOOD_MIN = 0.15
REM_GOOD_MAX = 0.30
REM_FAIR_MIN = 0.10
REM_FAIR_MAX = 0.35

# Deep sleep percentage rating thresholds
DEEP_EXCELLENT_MIN = 0.13
DEEP_EXCELLENT_MAX = 0.20
DEEP_GOOD_MIN = 0.10
DEEP_GOOD_MAX = 0.25
DEEP_FAIR_MIN = 0.05
DEEP_FAIR_MAX = 0.30

# Consistency rating thresholds
CONSISTENCY_EXCELLENT = 0.8
CONSISTENCY_GOOD = 0.6
CONSISTENCY_FAIR = 0.4

# Consistency calculation thresholds (minutes)
CONSISTENCY_PERFECT_THRESHOLD = 15  # Within 15 minutes = perfect

# Overall quality rating thresholds
QUALITY_EXCELLENT = 4.0
QUALITY_GOOD = 3.0
QUALITY_FAIR = 2.0


class SleepFeatures(BaseModel):
    """Comprehensive sleep features extracted from sleep data.

    Attributes align with clinical sleep research standards and are suitable
    for machine learning feature vectors or clinical assessments.
    """

    # Basic sleep metrics
    total_sleep_minutes: float = Field(default=0.0, description="Total sleep duration")
    sleep_efficiency: float = Field(default=0.0, description="Sleep efficiency ratio")
    sleep_latency: float = Field(
        default=0.0, description="Time to fall asleep (minutes)"
    )
    awakenings_count: float = Field(default=0.0, description="Number of awakenings")

    # Sleep architecture
    rem_percentage: float = Field(default=0.0, description="REM sleep percentage")
    deep_percentage: float = Field(default=0.0, description="Deep sleep percentage")
    light_percentage: float = Field(default=0.0, description="Light sleep percentage")
    waso_minutes: float = Field(default=0.0, description="Wake after sleep onset")

    # Sleep consistency and quality
    consistency_score: float = Field(
        default=0.0, description="Sleep schedule consistency"
    )
    overall_quality_score: float = Field(
        default=0.0, description="Overall sleep quality"
    )


class SleepProcessor:
    """Sleep data processor for comprehensive sleep analysis.

    Processes Apple HealthKit sleep data to extract clinically-relevant features
    for sleep quality assessment and trend analysis.
    """

    def __init__(self) -> None:
        """Initialize the sleep processor."""
        self.logger = logging.getLogger(__name__)

    def process(self, sleep_metrics: list[HealthMetric]) -> SleepFeatures:
        """Process sleep metrics to extract comprehensive features.

        Args:
            sleep_metrics: List of sleep-related health metrics

        Returns:
            SleepFeatures object with extracted features
        """
        # Filter to sleep-related metrics only
        sleep_data_list = [
            metric.sleep_data
            for metric in sleep_metrics
            if metric.sleep_data is not None
        ]

        if not sleep_data_list:
            self.logger.warning("No sleep data found in metrics")
            return self._create_empty_features()

        self.logger.info("Processing %d sleep records", len(sleep_data_list))

        # Initialize feature collections
        feature_sets: dict[str, list[float]] = {
            "total_sleep": [],
            "efficiency": [],
            "latency": [],
            "awakenings": [],
            "rem_pct": [],
            "deep_pct": [],
            "light_pct": [],
            "waso": [],
            "start_times": [],
        }

        # Process each sleep data record
        for sleep_data in sleep_data_list:
            self._extract_basic_features(sleep_data, feature_sets)
            self._extract_sleep_stages(sleep_data, feature_sets)
            self._extract_timing_features(sleep_data, feature_sets)

        # Calculate aggregated features
        features = SleepFeatures(
            total_sleep_minutes=float(np.mean(feature_sets["total_sleep"])),
            sleep_efficiency=float(np.mean(feature_sets["efficiency"])),
            sleep_latency=float(np.mean(feature_sets["latency"])),
            awakenings_count=float(np.mean(feature_sets["awakenings"])),
            rem_percentage=float(np.mean(feature_sets["rem_pct"])),
            deep_percentage=float(np.mean(feature_sets["deep_pct"])),
            light_percentage=float(np.mean(feature_sets["light_pct"])),
            waso_minutes=float(np.mean(feature_sets["waso"])),
            consistency_score=self._calculate_consistency_score(
                feature_sets["start_times"]
            ),
        )

        # Calculate overall quality score
        features.overall_quality_score = self._calculate_overall_quality_score(features)

        return features

    @staticmethod
    def _extract_basic_features(
        sleep_data: SleepData, feature_sets: dict[str, list[float]]
    ) -> None:
        """Extract basic sleep metrics from sleep data."""
        feature_sets["total_sleep"].append(float(sleep_data.total_sleep_minutes))
        feature_sets["efficiency"].append(float(sleep_data.sleep_efficiency))
        feature_sets["latency"].append(float(sleep_data.time_to_sleep_minutes or 0))
        feature_sets["awakenings"].append(float(sleep_data.wake_count or 0))

    def _extract_sleep_stages(
        self, sleep_data: SleepData, feature_sets: dict[str, list[float]]
    ) -> None:
        """Extract sleep stage percentages and WASO."""
        rem_pct, deep_pct, light_pct = self._extract_stage_percentages(sleep_data)
        feature_sets["rem_pct"].append(rem_pct)
        feature_sets["deep_pct"].append(deep_pct)
        feature_sets["light_pct"] = feature_sets.get("light_pct", [])
        feature_sets["light_pct"].append(light_pct)
        feature_sets["waso"].append(self._calculate_waso(sleep_data))

    @staticmethod
    def _extract_timing_features(
        sleep_data: SleepData, feature_sets: dict[str, list[float]]
    ) -> None:
        """Extract sleep timing features for consistency analysis."""
        start_hour = sleep_data.sleep_start.hour + sleep_data.sleep_start.minute / 60.0
        # Convert to consistent time scale (e.g., 23:30 = 23.5, 00:30 = 24.5)
        if start_hour < NOON_HOUR_THRESHOLD:  # Early morning hours (past midnight)
            start_hour += 24
        feature_sets["start_times"].append(start_hour)

    @staticmethod
    def _calculate_waso(sleep_data: SleepData) -> float:
        """Calculate Wake After Sleep Onset from sleep data."""
        if sleep_data.sleep_stages and SleepStage.AWAKE in sleep_data.sleep_stages:
            # WASO from sleep stages (minutes awake after initial sleep)
            return float(sleep_data.sleep_stages[SleepStage.AWAKE])

        # Fallback: estimate from efficiency and latency
        time_in_bed = (
            sleep_data.sleep_end - sleep_data.sleep_start
        ).total_seconds() / 60
        latency = sleep_data.time_to_sleep_minutes or 0
        waso = time_in_bed - sleep_data.total_sleep_minutes - latency
        return max(0.0, waso)  # Ensure non-negative

    @staticmethod
    def _extract_stage_percentages(sleep_data: SleepData) -> tuple[float, float, float]:
        """Extract REM, deep, and light sleep percentages."""
        if not sleep_data.sleep_stages:
            return 0.0, 0.0, 0.0

        total_sleep = sleep_data.total_sleep_minutes
        if total_sleep <= 0:
            return 0.0, 0.0, 0.0

        rem_minutes = sleep_data.sleep_stages.get(SleepStage.REM, 0)
        deep_minutes = sleep_data.sleep_stages.get(SleepStage.DEEP, 0)
        light_minutes = sleep_data.sleep_stages.get(SleepStage.LIGHT, 0)

        rem_percentage = rem_minutes / total_sleep
        deep_percentage = deep_minutes / total_sleep
        light_percentage = light_minutes / total_sleep

        return float(rem_percentage), float(deep_percentage), float(light_percentage)

    def get_summary_stats(
        self, sleep_metrics: list[HealthMetric]
    ) -> dict[str, str | float]:
        """Get summary statistics and ratings for sleep data."""
        features = self.process(sleep_metrics)

        return {
            "overall_quality_rating": self._rate_overall_quality(
                features.overall_quality_score
            ),
            "sleep_duration_hours": features.total_sleep_minutes / 60,
            "sleep_efficiency_rating": self._rate_efficiency(features.sleep_efficiency),
            "sleep_latency_rating": self._rate_latency(features.sleep_latency),
            "waso_rating": self._rate_waso(features.waso_minutes),
            "rem_sleep_rating": self._rate_rem_percentage(features.rem_percentage),
            "deep_sleep_rating": self._rate_deep_percentage(features.deep_percentage),
            "consistency_rating": self._rate_consistency(features.consistency_score),
        }

    @staticmethod
    def _rate_overall_quality(quality_score: float) -> str:
        """Rate overall sleep quality."""
        if quality_score >= QUALITY_EXCELLENT:
            return "excellent"
        if quality_score >= QUALITY_GOOD:
            return "good"
        if quality_score >= QUALITY_FAIR:
            return "fair"
        return "poor"

    @staticmethod
    def _calculate_consistency_score(start_times: list[float]) -> float:
        """Calculate sleep schedule consistency score."""
        if len(start_times) < MIN_VALUES_FOR_CONSISTENCY:
            return 0.0  # Need multiple nights for consistency

        # Calculate standard deviation of sleep start times
        std_hours = float(np.std(start_times))
        std_minutes = std_hours * 60

        # Convert to consistency score (1.0 = perfect, 0.0 = very inconsistent)
        if std_minutes <= CONSISTENCY_PERFECT_THRESHOLD:  # Within 15 minutes
            return 1.0
        if std_minutes >= CONSISTENCY_STD_THRESHOLD:  # 30+ minutes variation
            return 0.0

        return 1.0 - (
            (std_minutes - CONSISTENCY_PERFECT_THRESHOLD)
            / (CONSISTENCY_STD_THRESHOLD - CONSISTENCY_PERFECT_THRESHOLD)
        )

    @staticmethod
    def _calculate_overall_quality_score(features: SleepFeatures) -> float:
        """Calculate comprehensive sleep quality score."""
        scores = []

        # Sleep efficiency component (0-1)
        scores.append(min(1.0, features.sleep_efficiency))

        # Sleep latency component (0-1, inverted - lower is better)
        latency_score = max(
            0.0, 1.0 - (features.sleep_latency / 60)
        )  # 60 min = 0 score
        scores.append(latency_score)

        # WASO component (0-1, inverted - lower is better)
        waso_score = max(0.0, 1.0 - (features.waso_minutes / 120))  # 120 min = 0 score
        scores.append(waso_score)

        # REM percentage component (0-1)
        rem_score = 0.0
        if REM_EXCELLENT_MIN <= features.rem_percentage <= REM_EXCELLENT_MAX:
            rem_score = 1.0
        elif REM_GOOD_MIN <= features.rem_percentage <= REM_GOOD_MAX:
            rem_score = 0.75
        elif REM_FAIR_MIN <= features.rem_percentage <= REM_FAIR_MAX:
            rem_score = 0.5
        scores.append(rem_score)

        # Deep sleep percentage component (0-1)
        deep_score = 0.0
        if DEEP_EXCELLENT_MIN <= features.deep_percentage <= DEEP_EXCELLENT_MAX:
            deep_score = 1.0
        elif DEEP_GOOD_MIN <= features.deep_percentage <= DEEP_GOOD_MAX:
            deep_score = 0.75
        elif DEEP_FAIR_MIN <= features.deep_percentage <= DEEP_FAIR_MAX:
            deep_score = 0.5
        scores.extend([deep_score, features.consistency_score])

        # Return sum of components (max 6.0)
        return float(np.sum(scores))

    @staticmethod
    def _create_empty_features() -> SleepFeatures:
        """Create empty features for when no sleep data is available."""
        return SleepFeatures(
            total_sleep_minutes=0.0,
            sleep_efficiency=0.0,
            sleep_latency=0.0,
            awakenings_count=0.0,
            rem_percentage=0.0,
            deep_percentage=0.0,
            light_percentage=0.0,
            waso_minutes=0.0,
            consistency_score=0.0,
            overall_quality_score=0.0,
        )

    @staticmethod
    def _rate_efficiency(efficiency: float) -> str:
        """Rate sleep efficiency."""
        if efficiency >= EFFICIENCY_EXCELLENT:
            return "excellent"
        if efficiency >= EFFICIENCY_GOOD:
            return "good"
        if efficiency >= EFFICIENCY_FAIR:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_latency(latency: float) -> str:
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
        """Rate wake after sleep onset."""
        if waso <= WASO_EXCELLENT:
            return "excellent"
        if waso <= WASO_GOOD:
            return "good"
        if waso <= WASO_FAIR:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_rem_percentage(rem_pct: float) -> str:
        """Rate REM sleep percentage."""
        if REM_EXCELLENT_MIN <= rem_pct <= REM_EXCELLENT_MAX:  # 18-25%
            return "excellent"
        if REM_GOOD_MIN <= rem_pct <= REM_GOOD_MAX:  # 15-30%
            return "good"
        if REM_FAIR_MIN <= rem_pct <= REM_FAIR_MAX:  # 10-35%
            return "fair"
        return "poor"

    @staticmethod
    def _rate_deep_percentage(deep_pct: float) -> str:
        """Rate deep sleep percentage."""
        if DEEP_EXCELLENT_MIN <= deep_pct <= DEEP_EXCELLENT_MAX:  # 13-20%
            return "excellent"
        if DEEP_GOOD_MIN <= deep_pct <= DEEP_GOOD_MAX:  # 10-25%
            return "good"
        if DEEP_FAIR_MIN <= deep_pct <= DEEP_FAIR_MAX:  # 5-30%
            return "fair"
        return "poor"

    @staticmethod
    def _rate_consistency(consistency: float) -> str:
        """Rate sleep consistency."""
        if consistency >= CONSISTENCY_EXCELLENT:
            return "excellent"
        if consistency >= CONSISTENCY_GOOD:
            return "good"
        if consistency >= CONSISTENCY_FAIR:
            return "fair"
        return "poor"
