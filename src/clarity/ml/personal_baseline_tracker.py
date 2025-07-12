"""Personal Baseline Tracker for Individualized Bipolar Monitoring.

Based on Lipschitz et al. (2025) showing that personalized models
outperform population-based approaches (86% AUC for depression, 85% for mania).

This module maintains individual baselines for all health metrics,
enabling detection of personal deviations rather than population norms.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthMetric,
    SleepData,
)


@dataclass
class PersonalBaseline:
    """Individual baseline statistics for a user."""

    user_id: str

    # Sleep baselines
    sleep_duration_median: float = 0.0
    sleep_duration_p25: float = 0.0
    sleep_duration_p75: float = 0.0
    sleep_midpoint_median: float = 0.0  # Hours from midnight
    sleep_efficiency_median: float = 0.0
    sleep_latency_median: float = 0.0

    # Activity baselines
    daily_steps_median: float = 0.0
    daily_steps_p25: float = 0.0
    daily_steps_p75: float = 0.0
    active_energy_median: float = 0.0
    exercise_minutes_median: float = 0.0

    # Physiological baselines
    resting_hr_median: float = 0.0
    resting_hr_p25: float = 0.0
    resting_hr_p75: float = 0.0
    hrv_median: float = 0.0

    # Variability baselines
    sleep_variability_baseline: float = 0.0
    activity_variability_baseline: float = 0.0
    hr_variability_baseline: float = 0.0

    # Circadian profile
    typical_bedtime: float = 0.0  # Hours from midnight
    typical_waketime: float = 0.0
    circadian_consistency: float = 0.0

    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    data_days: int = 0
    confidence_score: float = 0.0


class PersonalBaselineTracker:
    """Tracks and maintains personal baselines for bipolar monitoring.

    Key features:
    - Adaptive baselines that update with new data
    - Robust to outliers using median/percentile approach
    - Handles sparse data gracefully
    - Efficient updates without full recalculation
    """

    def __init__(self, min_days_for_baseline: int = 14) -> None:
        """Initialize the tracker.

        Args:
            min_days_for_baseline: Minimum days of data for reliable baseline
        """
        self.logger = logging.getLogger(__name__)
        self.min_days = min_days_for_baseline

        # In production, this would use DynamoDB
        self._baselines: dict[str, PersonalBaseline] = {}

    def update_baseline(
        self,
        user_id: str,
        health_metrics: list[HealthMetric],
        force_full_recalc: bool = False,
    ) -> PersonalBaseline:
        """Update personal baseline with new health data.

        Args:
            user_id: User identifier
            health_metrics: Recent health metrics (ideally 28+ days)
            force_full_recalc: Force full recalculation vs incremental

        Returns:
            Updated PersonalBaseline
        """
        self.logger.info(
            "Updating baseline for user",
            extra={
                "user_id": self._sanitize_user_id(user_id),
                "metrics_count": len(health_metrics),
                "force_recalc": force_full_recalc,
            },
        )

        # Get or create baseline
        if user_id not in self._baselines:
            self._baselines[user_id] = PersonalBaseline(user_id=user_id)

        baseline = self._baselines[user_id]

        # Extract data by type
        sleep_data = self._extract_sleep_data(health_metrics)
        activity_data = self._extract_activity_data(health_metrics)
        bio_data = self._extract_biometric_data(health_metrics)

        # Update each component
        self._update_sleep_baseline(baseline, sleep_data)
        self._update_activity_baseline(baseline, activity_data)
        self._update_physiological_baseline(baseline, bio_data)
        self._update_variability_baseline(baseline, sleep_data, activity_data)
        self._update_circadian_profile(baseline, sleep_data)

        # Update metadata
        baseline.last_updated = datetime.now(UTC)
        baseline.data_days = self._count_unique_days(health_metrics)
        baseline.confidence_score = self._calculate_confidence(baseline)

        self.logger.info(
            "Baseline updated successfully",
            extra={
                "user_id": self._sanitize_user_id(user_id),
                "data_days": baseline.data_days,
                "confidence": round(baseline.confidence_score, 3),
            },
        )

        return baseline

    def get_baseline(self, user_id: str) -> PersonalBaseline | None:
        """Retrieve user's baseline if available."""
        return self._baselines.get(user_id)

    def calculate_deviation_scores(
        self, user_id: str, current_metrics: dict[str, float]
    ) -> dict[str, float]:
        """Calculate how current metrics deviate from personal baseline.

        Args:
            user_id: User identifier
            current_metrics: Current metric values

        Returns:
            Dictionary of deviation z-scores
        """
        baseline = self.get_baseline(user_id)
        if not baseline:
            return {}

        deviations = {}

        # Sleep duration deviation
        if "sleep_hours" in current_metrics and baseline.sleep_duration_median > 0:
            z_score = self._calculate_z_score(
                current_metrics["sleep_hours"],
                baseline.sleep_duration_median,
                baseline.sleep_duration_p25,
                baseline.sleep_duration_p75,
            )
            deviations["sleep_duration_z"] = z_score

        # Activity deviation
        if "daily_steps" in current_metrics and baseline.daily_steps_median > 0:
            z_score = self._calculate_z_score(
                current_metrics["daily_steps"],
                baseline.daily_steps_median,
                baseline.daily_steps_p25,
                baseline.daily_steps_p75,
            )
            deviations["activity_z"] = z_score

        # Heart rate deviation
        if "resting_hr" in current_metrics and baseline.resting_hr_median > 0:
            z_score = self._calculate_z_score(
                current_metrics["resting_hr"],
                baseline.resting_hr_median,
                baseline.resting_hr_p25,
                baseline.resting_hr_p75,
            )
            deviations["hr_z"] = z_score

        # Circadian deviation
        if "sleep_midpoint" in current_metrics and baseline.sleep_midpoint_median > 0:
            # Handle wraparound
            diff = current_metrics["sleep_midpoint"] - baseline.sleep_midpoint_median
            if diff > 12:
                diff -= 24
            elif diff < -12:
                diff += 24
            deviations["circadian_shift_hours"] = diff

        return deviations

    def _extract_sleep_data(
        self, metrics: list[HealthMetric]
    ) -> list[dict[str, Any]]:
        """Extract sleep data points from metrics."""
        sleep_points = []

        for metric in metrics:
            if metric.sleep_data:
                sleep = metric.sleep_data
                point = {
                    "date": metric.created_at.date(),
                    "duration_hours": (
                        sleep.total_sleep_minutes / 60.0
                        if sleep.total_sleep_minutes
                        else 0
                    ),
                    "efficiency": sleep.sleep_efficiency or 0,
                    "latency": sleep.time_to_sleep_minutes or 0,
                }

                # Calculate midpoint if times available
                if sleep.sleep_start and sleep.sleep_end:
                    midpoint = self._calculate_sleep_midpoint(
                        sleep.sleep_start, sleep.sleep_end
                    )
                    point["midpoint"] = midpoint

                sleep_points.append(point)

        return sleep_points

    def _extract_activity_data(
        self, metrics: list[HealthMetric]
    ) -> list[dict[str, Any]]:
        """Extract activity data points from metrics."""
        activity_by_date = defaultdict(list)

        for metric in metrics:
            if metric.activity_data:
                activity = metric.activity_data
                date = metric.created_at.date()
                activity_by_date[date].append(
                    {
                        "steps": activity.steps or 0,
                        "active_energy": activity.active_energy or 0,
                        "exercise_minutes": activity.exercise_minutes or 0,
                    }
                )

        # Aggregate by day
        daily_activity = []
        for date, activities in activity_by_date.items():
            daily_activity.append(
                {
                    "date": date,
                    "steps": sum(a["steps"] for a in activities),
                    "active_energy": sum(a["active_energy"] for a in activities),
                    "exercise_minutes": sum(a["exercise_minutes"] for a in activities),
                }
            )

        return daily_activity

    def _extract_biometric_data(
        self, metrics: list[HealthMetric]
    ) -> list[dict[str, Any]]:
        """Extract biometric data points from metrics."""
        bio_points = []

        for metric in metrics:
            if metric.biometric_data:
                bio = metric.biometric_data
                if bio.heart_rate or bio.heart_rate_variability:
                    bio_points.append(
                        {
                            "date": metric.created_at.date(),
                            "hr": bio.heart_rate or 0,
                            "hrv": bio.heart_rate_variability or 0,
                        }
                    )

        return bio_points

    def _update_sleep_baseline(
        self, baseline: PersonalBaseline, sleep_data: list[dict[str, Any]]
    ) -> None:
        """Update sleep-related baselines."""
        if not sleep_data:
            return

        # Extract values
        durations = [s["duration_hours"] for s in sleep_data if s["duration_hours"] > 0]
        efficiencies = [s["efficiency"] for s in sleep_data if s["efficiency"] > 0]
        latencies = [s["latency"] for s in sleep_data if s["latency"] >= 0]
        midpoints = [s["midpoint"] for s in sleep_data if "midpoint" in s]

        # Update baselines using robust statistics
        if durations:
            baseline.sleep_duration_median = float(np.median(durations))
            baseline.sleep_duration_p25 = float(np.percentile(durations, 25))
            baseline.sleep_duration_p75 = float(np.percentile(durations, 75))

        if efficiencies:
            baseline.sleep_efficiency_median = float(np.median(efficiencies))

        if latencies:
            baseline.sleep_latency_median = float(np.median(latencies))

        if midpoints:
            baseline.sleep_midpoint_median = float(np.median(midpoints))

    def _update_activity_baseline(
        self, baseline: PersonalBaseline, activity_data: list[dict[str, Any]]
    ) -> None:
        """Update activity-related baselines."""
        if not activity_data:
            return

        # Extract values
        steps = [a["steps"] for a in activity_data if a["steps"] > 0]
        energy = [a["active_energy"] for a in activity_data if a["active_energy"] > 0]
        exercise = [
            a["exercise_minutes"] for a in activity_data if a["exercise_minutes"] >= 0
        ]

        # Update baselines
        if steps:
            baseline.daily_steps_median = float(np.median(steps))
            baseline.daily_steps_p25 = float(np.percentile(steps, 25))
            baseline.daily_steps_p75 = float(np.percentile(steps, 75))

        if energy:
            baseline.active_energy_median = float(np.median(energy))

        if exercise:
            baseline.exercise_minutes_median = float(np.median(exercise))

    def _update_physiological_baseline(
        self, baseline: PersonalBaseline, bio_data: list[dict[str, Any]]
    ) -> None:
        """Update physiological baselines."""
        if not bio_data:
            return

        # Extract values
        hrs = [b["hr"] for b in bio_data if b["hr"] > 0]
        hrvs = [b["hrv"] for b in bio_data if b["hrv"] > 0]

        # Update baselines
        if hrs:
            baseline.resting_hr_median = float(np.median(hrs))
            baseline.resting_hr_p25 = float(np.percentile(hrs, 25))
            baseline.resting_hr_p75 = float(np.percentile(hrs, 75))

        if hrvs:
            baseline.hrv_median = float(np.median(hrvs))

    def _update_variability_baseline(
        self,
        baseline: PersonalBaseline,
        sleep_data: list[dict[str, Any]],
        activity_data: list[dict[str, Any]],
    ) -> None:
        """Update variability baselines."""
        # Sleep variability
        if len(sleep_data) >= 7:
            durations = [
                s["duration_hours"] for s in sleep_data if s["duration_hours"] > 0
            ]
            if durations:
                baseline.sleep_variability_baseline = float(np.std(durations) / np.mean(
                    durations
                ))

        # Activity variability
        if len(activity_data) >= 7:
            steps = [a["steps"] for a in activity_data if a["steps"] > 0]
            if steps:
                baseline.activity_variability_baseline = float(np.std(steps) / np.mean(steps))

    def _update_circadian_profile(
        self, baseline: PersonalBaseline, sleep_data: list[dict[str, Any]]
    ) -> None:
        """Update circadian rhythm profile."""
        if not sleep_data:
            return

        midpoints = [s["midpoint"] for s in sleep_data if "midpoint" in s]

        if len(midpoints) >= 7:
            # Typical sleep midpoint
            baseline.sleep_midpoint_median = float(np.median(midpoints))

            # Estimate bed/wake times (rough approximation)
            avg_duration = baseline.sleep_duration_median
            baseline.typical_bedtime = (
                baseline.sleep_midpoint_median - avg_duration / 2
            ) % 24
            baseline.typical_waketime = (
                baseline.sleep_midpoint_median + avg_duration / 2
            ) % 24

            # Circadian consistency (lower is better)
            baseline.circadian_consistency = float(np.std(midpoints))

    def _calculate_sleep_midpoint(self, start: datetime, end: datetime) -> float:
        """Calculate sleep midpoint as hours from midnight."""
        if end < start:
            end += timedelta(days=1)

        duration = (end - start).total_seconds() / 3600
        start_hour = start.hour + start.minute / 60
        return (start_hour + duration / 2) % 24

    def _calculate_z_score(
        self, value: float, median: float, p25: float, p75: float
    ) -> float:
        """Calculate robust z-score using IQR method."""
        iqr = p75 - p25
        if iqr == 0:
            return 0.0

        # Use median and IQR for robust z-score
        mad = iqr / 1.35  # Approximate MAD from IQR
        return (value - median) / mad if mad > 0 else 0.0

    def _count_unique_days(self, metrics: list[HealthMetric]) -> int:
        """Count unique days in metrics."""
        unique_days: set[Any] = set()
        unique_days.update(metric.created_at.date() for metric in metrics)
        return len(unique_days)

    def _calculate_confidence(self, baseline: PersonalBaseline) -> float:
        """Calculate confidence in baseline accuracy."""
        # Base confidence on data availability
        data_confidence = min(baseline.data_days / self.min_days, 1.0)

        # Adjust for data recency
        days_old = (datetime.now(UTC) - baseline.last_updated).days
        recency_factor = max(0, 1 - days_old / 30)  # Decay over 30 days

        return data_confidence * (0.7 + 0.3 * recency_factor)

    def _sanitize_user_id(self, user_id: str) -> str:
        """Sanitize user ID for logging."""
        if len(user_id) > 8:
            return f"{user_id[:4]}...{user_id[-4:]}"
        return user_id

    def export_baseline(self, user_id: str) -> dict[str, Any] | None:
        """Export baseline as JSON-serializable dictionary."""
        baseline = self.get_baseline(user_id)
        if not baseline:
            return None

        return {
            "user_id": baseline.user_id,
            "sleep": {
                "duration_median_hours": baseline.sleep_duration_median,
                "duration_iqr": [
                    baseline.sleep_duration_p25,
                    baseline.sleep_duration_p75,
                ],
                "midpoint_median": baseline.sleep_midpoint_median,
                "efficiency_median": baseline.sleep_efficiency_median,
            },
            "activity": {
                "daily_steps_median": baseline.daily_steps_median,
                "daily_steps_iqr": [baseline.daily_steps_p25, baseline.daily_steps_p75],
                "active_energy_median": baseline.active_energy_median,
            },
            "physiology": {
                "resting_hr_median": baseline.resting_hr_median,
                "resting_hr_iqr": [baseline.resting_hr_p25, baseline.resting_hr_p75],
                "hrv_median": baseline.hrv_median,
            },
            "variability": {
                "sleep_cv": baseline.sleep_variability_baseline,
                "activity_cv": baseline.activity_variability_baseline,
            },
            "circadian": {
                "typical_bedtime": baseline.typical_bedtime,
                "typical_waketime": baseline.typical_waketime,
                "consistency": baseline.circadian_consistency,
            },
            "metadata": {
                "last_updated": baseline.last_updated.isoformat(),
                "data_days": baseline.data_days,
                "confidence": baseline.confidence_score,
            },
        }
