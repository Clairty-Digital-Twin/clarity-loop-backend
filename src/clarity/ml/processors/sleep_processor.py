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

# Clinical sleep thresholds (based on AASM guidelines)
OPTIMAL_SLEEP_MIN = 7.0  # hours
OPTIMAL_SLEEP_MAX = 9.0  # hours
NORMAL_SLEEP_EFFICIENCY = 0.85  # 85%
NORMAL_SLEEP_LATENCY = 30  # minutes
NORMAL_WASO = 20  # minutes
NORMAL_REM_PERCENTAGE = 0.20  # 20%
NORMAL_DEEP_PERCENTAGE = 0.15  # 15%

# Consistency analysis
MIN_VALUES_FOR_CONSISTENCY = 2
CONSISTENCY_STD_THRESHOLD = 30  # minutes

# Quality rating thresholds
EXCELLENT_QUALITY = 0.8
GOOD_QUALITY = 0.6
FAIR_QUALITY = 0.4


class SleepFeatures(BaseModel):
    """Comprehensive sleep features extracted from sleep data.
    
    Attributes align with clinical sleep research standards and are suitable
    for machine learning feature vectors or clinical assessments.
    """

    # Duration metrics (minutes)
    total_sleep_minutes: float = Field(description="Total time spent asleep")
    sleep_efficiency: float = Field(description="Sleep efficiency ratio (0-1)")
    sleep_latency: float = Field(description="Time to fall asleep (minutes)")
    waso_minutes: float = Field(description="Wake After Sleep Onset (minutes)")

    # Fragmentation metrics
    awakenings_count: float = Field(description="Number of awakenings")

    # Sleep architecture (percentages 0-1)
    rem_percentage: float = Field(description="REM sleep percentage")
    deep_percentage: float = Field(description="Deep sleep percentage")
    light_percentage: float = Field(description="Light sleep percentage")

    # Schedule consistency (0-1 score)
    consistency_score: float = Field(description="Sleep schedule consistency")

    # Overall quality score (0-1)
    overall_quality_score: float = Field(description="Overall sleep quality score")


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
        if not sleep_metrics:
            self.logger.warning("No sleep metrics provided")
            return self._create_empty_features()

        # Filter for sleep data only
        sleep_data_list = [
            metric.sleep_data for metric in sleep_metrics
            if metric.sleep_data is not None
        ]

        if not sleep_data_list:
            self.logger.warning("No valid sleep data found in metrics")
            return self._create_empty_features()

        self.logger.info(f"Processing {len(sleep_data_list)} sleep records")

        # Initialize feature collections
        feature_sets = self._initialize_feature_collections()

        # Process each sleep record
        for sleep_data in sleep_data_list:
            self._process_sleep_record(sleep_data, feature_sets)

        # Aggregate features
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
        feature_sets["awakenings"].append(float(sleep_data.wake_count))

        # Extract sleep stage percentages
        rem_pct, deep_pct = self._extract_stage_percentages(sleep_data)
        feature_sets["rem_percentage"].append(rem_pct)
        feature_sets["deep_percentage"].append(deep_pct)

        # Track start times for consistency analysis
        start_hour = sleep_data.sleep_start.hour + sleep_data.sleep_start.minute / 60.0
        # Convert to consistent time scale (e.g., 23:30 = 23.5, 00:30 = 24.5)
        if start_hour < 12:  # Early morning hours (past midnight)
            start_hour += 24
        feature_sets["start_times"].append(start_hour)

    def _calculate_waso(self, sleep_data: SleepData) -> float:
        """Calculate Wake After Sleep Onset from sleep data."""
        if sleep_data.sleep_stages and SleepStage.AWAKE in sleep_data.sleep_stages:
            return float(sleep_data.sleep_stages[SleepStage.AWAKE])

        # Derive from timing: time_in_bed - total_sleep - latency
        time_in_bed = (sleep_data.sleep_end - sleep_data.sleep_start).total_seconds() / 60
        latency = float(sleep_data.time_to_sleep_minutes or 0)
        waso = time_in_bed - sleep_data.total_sleep_minutes - latency
        return max(0.0, waso)  # Ensure non-negative

    def _extract_stage_percentages(self, sleep_data: SleepData) -> tuple[float, float]:
        """Extract REM and deep sleep percentages."""
        if not sleep_data.sleep_stages:
            return 0.0, 0.0

        total_sleep = sleep_data.total_sleep_minutes
        if total_sleep <= 0:
            return 0.0, 0.0

        rem_minutes = sleep_data.sleep_stages.get(SleepStage.REM, 0)
        deep_minutes = sleep_data.sleep_stages.get(SleepStage.DEEP, 0)

        rem_percentage = float(rem_minutes) / total_sleep
        deep_percentage = float(deep_minutes) / total_sleep

        return rem_percentage, deep_percentage

    def _aggregate_features(self, feature_sets: dict[str, list[Any]]) -> SleepFeatures:
        """Aggregate feature collections into final SleepFeatures."""
        if not feature_sets["total_sleep"]:
            return self._create_empty_features()

        # Calculate averages for most metrics
        avg_total_sleep = float(np.mean(feature_sets["total_sleep"]))
        avg_efficiency = float(np.mean(feature_sets["efficiency"]))
        avg_latency = float(np.mean(feature_sets["latency"]))
        avg_waso = float(np.mean(feature_sets["waso"]))
        avg_awakenings = float(np.mean(feature_sets["awakenings"]))
        avg_rem_pct = float(np.mean(feature_sets["rem_percentage"]))
        avg_deep_pct = float(np.mean(feature_sets["deep_percentage"]))

        # Calculate light sleep percentage (remainder)
        light_pct = max(0.0, 1.0 - avg_rem_pct - avg_deep_pct)

        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(feature_sets["start_times"])

        # Calculate overall quality score
        features = SleepFeatures(
            total_sleep_minutes=avg_total_sleep,
            sleep_efficiency=avg_efficiency,
            sleep_latency=avg_latency,
            waso_minutes=avg_waso,
            awakenings_count=avg_awakenings,
            rem_percentage=avg_rem_pct,
            deep_percentage=avg_deep_pct,
            light_percentage=light_pct,
            consistency_score=consistency_score,
            overall_quality_score=0.0  # Will be calculated below
        )

        # Calculate overall quality score based on all features
        features.overall_quality_score = self._calculate_overall_quality_score(features)

        return features

    def _calculate_consistency_score(self, start_times: list[float]) -> float:
        """Calculate sleep schedule consistency score."""
        if len(start_times) < MIN_VALUES_FOR_CONSISTENCY:
            return 0.0  # Need multiple nights for consistency

        # Calculate standard deviation in minutes
        std_hours = float(np.std(start_times))
        std_minutes = std_hours * 60

        # Convert to consistency score (1.0 = perfect, 0.0 = very inconsistent)
        if std_minutes <= 15:  # Within 15 minutes
            return 1.0
        if std_minutes >= CONSISTENCY_STD_THRESHOLD:  # 30+ minutes variation
            return 0.0
        # Linear scale between 15 and 30 minutes
        return 1.0 - ((std_minutes - 15) / (CONSISTENCY_STD_THRESHOLD - 15))

    def _calculate_overall_quality_score(self, features: SleepFeatures) -> float:
        """Calculate comprehensive sleep quality score."""
        scores = []

        # Sleep duration (weight: 20%)
        duration_hours = features.total_sleep_minutes / 60.0
        if OPTIMAL_SLEEP_MIN <= duration_hours <= OPTIMAL_SLEEP_MAX:
            duration_score = 1.0
        elif duration_hours < OPTIMAL_SLEEP_MIN:  # Too little
            duration_score = max(0.0, duration_hours / OPTIMAL_SLEEP_MIN)
        else:  # > 9 hours
            duration_score = max(0.0, 1.0 - (duration_hours - OPTIMAL_SLEEP_MAX) / 4)

        # Sleep efficiency (weight: 25%)
        efficiency_score = features.sleep_efficiency

        # Sleep latency (weight: 15%) - inverse relationship
        latency_score = max(0.0, 1.0 - (features.sleep_latency / 60.0))  # 60 min = 0 score

        # WASO (weight: 15%) - inverse relationship
        waso_score = max(0.0, 1.0 - (features.waso_minutes / 120.0))  # 120 min = 0 score

        # REM sleep (weight: 10%)
        rem_score = 1.0 - abs(features.rem_percentage - NORMAL_REM_PERCENTAGE) / 0.2
        rem_score = max(0.0, min(1.0, rem_score))

        # Deep sleep (weight: 10%)
        deep_score = 1.0 - abs(features.deep_percentage - NORMAL_DEEP_PERCENTAGE) / 0.15
        deep_score = max(0.0, min(1.0, deep_score))

        # Consistency (weight: 5%)
        consistency_score = features.consistency_score

        # Calculate weighted average
        scores.extend([
            duration_score * 0.20,
            efficiency_score * 0.25,
            latency_score * 0.15,
            waso_score * 0.15,
            rem_score * 0.10,
            deep_score * 0.10,
            consistency_score * 0.05
        ])

        return float(np.sum(scores))

    def _create_empty_features(self) -> SleepFeatures:
        """Create empty features for when no sleep data is available."""
        return SleepFeatures(
            total_sleep_minutes=0.0,
            sleep_efficiency=0.0,
            sleep_latency=0.0,
            waso_minutes=0.0,
            awakenings_count=0.0,
            rem_percentage=0.0,
            deep_percentage=0.0,
            light_percentage=0.0,
            consistency_score=0.0,
            overall_quality_score=0.0
        )

    def get_summary_stats(self, sleep_metrics: list[HealthMetric]) -> dict[str, Any]:
        """Generate human-readable sleep summary statistics."""
        features = self.process(sleep_metrics)

        return {
            "sleep_duration_hours": round(features.total_sleep_minutes / 60.0, 2),
            "sleep_efficiency_rating": self._rate_efficiency(features.sleep_efficiency),
            "sleep_latency_rating": self._rate_latency(features.sleep_latency),
            "waso_rating": self._rate_waso(features.waso_minutes),
            "rem_sleep_rating": self._rate_rem_percentage(features.rem_percentage),
            "deep_sleep_rating": self._rate_deep_percentage(features.deep_percentage),
            "consistency_rating": self._rate_consistency(features.consistency_score),
            "overall_quality_rating": self._rate_quality_score(features.overall_quality_score)
        }

    @staticmethod
    def _rate_efficiency(efficiency: float) -> str:
        """Rate sleep efficiency."""
        if efficiency >= 0.9:
            return "excellent"
        if efficiency >= 0.8:
            return "good"
        if efficiency >= 0.7:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_latency(latency: float) -> str:
        """Rate sleep latency."""
        if latency <= 15:
            return "excellent"
        if latency <= 30:
            return "good"
        if latency <= 45:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_waso(waso: float) -> str:
        """Rate wake after sleep onset."""
        if waso <= 20:
            return "excellent"
        if waso <= 40:
            return "good"
        if waso <= 60:
            return "fair"
        return "poor"

    @staticmethod
    def _rate_rem_percentage(rem_pct: float) -> str:
        """Rate REM sleep percentage."""
        if 0.18 <= rem_pct <= 0.25:  # 18-25%
            return "excellent"
        if 0.15 <= rem_pct <= 0.30:  # 15-30%
            return "good"
        if 0.10 <= rem_pct <= 0.35:  # 10-35%
            return "fair"
        return "poor"

    @staticmethod
    def _rate_deep_percentage(deep_pct: float) -> str:
        """Rate deep sleep percentage."""
        if 0.13 <= deep_pct <= 0.20:  # 13-20%
            return "excellent"
        if 0.10 <= deep_pct <= 0.25:  # 10-25%
            return "good"
        if 0.05 <= deep_pct <= 0.30:  # 5-30%
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

    @staticmethod
    def _rate_quality_score(quality: float) -> str:
        """Rate overall quality score."""
        if quality >= EXCELLENT_QUALITY:
            return "excellent"
        if quality >= GOOD_QUALITY:
            return "good"
        if quality >= FAIR_QUALITY:
            return "fair"
        return "poor"
