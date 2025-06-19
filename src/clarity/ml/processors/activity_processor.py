"""Activity and Fitness Data Processor.

This processor extracts basic activity metrics from Apple HealthKit data,
complementing the PAT model's complex analysis with straightforward statistics
that users can easily understand and chat about.

Covers metrics identified as missing in FINAL_APPLE_HEALTH.md:
- Step count totals and averages
- Distance walked/ran
- Active energy (calories) burned
- Exercise time and active minutes
- Flights climbed
- Basic activity patterns

This fills the gap where the PAT model provides deep insights but users
need direct answers to questions like "How many steps did I walk this week?"
"""

# removed - breaks FastAPI

from collections.abc import Sequence
from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from clarity.models.health_data import ActivityData, HealthMetric

if TYPE_CHECKING:
    pass  # Only for type stubs now

logger = logging.getLogger(__name__)

# Constants
MIN_VALUES_FOR_CONSISTENCY = 2


class ActivityProcessor:
    """Process activity and fitness metrics for clear, actionable insights.

    This processor generates 12 comprehensive activity features:
    1. Total steps in period
    2. Average daily steps
    3. Peak daily steps
    4. Total distance (km)
    5. Average daily distance
    6. Total active energy (kcal)
    7. Average daily active energy
    8. Total exercise minutes
    9. Average daily exercise minutes
    10. Total flights climbed
    11. Total active minutes
    12. Activity consistency score (0-1)

    These metrics are designed to be easily understood and provide direct
    answers to common user questions about their activity levels.
    """

    def __init__(self) -> None:
        """Initialize the activity processor."""
        self.processor_name = "ActivityProcessor"
        self.version = "1.0.0"
        logger.info("✅ %s v%s initialized", self.processor_name, self.version)

    def process(self, metrics: list[HealthMetric]) -> list[dict[str, Any]]:
        """Process activity metrics and extract key features.

        Args:
            metrics: List of health metrics containing activity data

        Returns:
            List of activity feature dictionaries
        """
        try:
            logger.info("🏃 Processing %d metrics for activity analysis", len(metrics))

            # Extract activity data from metrics
            activity_data = self._extract_activity_data(metrics)

            if not activity_data:
                logger.warning("No activity data found in metrics")
                return [{"warning": "No activity data available"}]

            # Calculate features
            features = self._calculate_activity_features(activity_data)

            logger.info("✅ Extracted %d activity features", len(features))
        except Exception as e:
            logger.exception("Failed to process activity data")
            return [{"error": f"ActivityProcessor failed: {e!s}"}]
        else:
            return features

    @staticmethod
    def _extract_activity_data(metrics: list[HealthMetric]) -> list[ActivityData]:
        """Extract activity data from health metrics.

        Args:
            metrics: List of health metrics

        Returns:
            List of ActivityData objects
        """
        logger = __import__("logging").getLogger(__name__)

        activity_data = [
            metric.activity_data for metric in metrics if metric.activity_data
        ]

        logger.debug("Extracted %d activity data points", len(activity_data))
        return activity_data

    def _calculate_activity_features(
        self, activity_data: list[ActivityData]
    ) -> list[dict[str, Any]]:
        """Calculate comprehensive activity features.

        Args:
            activity_data: List of activity data points

        Returns:
            List of feature dictionaries
        """
        if not activity_data:
            return []

        # Aggregate metrics
        steps = [data.steps for data in activity_data if data.steps is not None]
        distances = [
            data.distance for data in activity_data if data.distance is not None
        ]
        active_energy = [
            data.active_energy
            for data in activity_data
            if data.active_energy is not None
        ]
        exercise_minutes = [
            data.exercise_minutes
            for data in activity_data
            if data.exercise_minutes is not None
        ]
        flights_climbed = [
            data.flights_climbed
            for data in activity_data
            if data.flights_climbed is not None
        ]
        active_minutes = [
            data.active_minutes
            for data in activity_data
            if data.active_minutes is not None
        ]
        vo2_max_values = [
            data.vo2_max for data in activity_data if data.vo2_max is not None
        ]
        resting_hr_values = [
            data.resting_heart_rate
            for data in activity_data
            if data.resting_heart_rate is not None
        ]

        # Calculate features
        features = []

        # 1. Step Count Features
        if steps:
            step_features = self._calculate_step_features(steps)
            features.extend(step_features)

        # 2. Distance Features
        if distances:
            distance_features = self._calculate_distance_features(distances)
            features.extend(distance_features)

        # 3. Energy Features
        if active_energy:
            energy_features = self._calculate_energy_features(active_energy)
            features.extend(energy_features)

        # 4. Exercise Features
        if exercise_minutes:
            exercise_features = self._calculate_exercise_features(exercise_minutes)
            features.extend(exercise_features)

        # 5. Other Activity Features
        if flights_climbed:
            features.append(
                {
                    "feature_name": "total_flights_climbed",
                    "value": sum(flights_climbed),
                    "unit": "flights",
                    "description": "Total flights of stairs climbed",
                }
            )

        if active_minutes:
            features.extend(
                (
                    {
                        "feature_name": "total_active_minutes",
                        "value": sum(active_minutes),
                        "unit": "minutes",
                        "description": "Total minutes of active movement",
                    },
                    {
                        "feature_name": "average_daily_active_minutes",
                        "value": np.mean(active_minutes),
                        "unit": "minutes/day",
                        "description": "Average active minutes per day",
                    },
                )
            )

        # 6. Fitness Level Features
        if vo2_max_values:
            latest_vo2_max = vo2_max_values[-1]  # Most recent value
            features.append(
                {
                    "feature_name": "latest_vo2_max",
                    "value": latest_vo2_max,
                    "unit": "mL/kg/min",
                    "description": "Most recent VO₂ max measurement (cardio fitness)",
                }
            )

        if resting_hr_values:
            features.append(
                {
                    "feature_name": "average_resting_heart_rate",
                    "value": np.mean(resting_hr_values),
                    "unit": "bpm",
                    "description": "Average resting heart rate",
                }
            )

        # 7. Activity Consistency Score
        if steps and len(steps) > 1:
            consistency_score = self._calculate_consistency_score(steps)
            features.append(
                {
                    "feature_name": "activity_consistency_score",
                    "value": consistency_score,
                    "unit": "score",
                    "description": "Activity consistency (0=inconsistent, 1=very consistent)",
                }
            )

        return features

    @staticmethod
    def _calculate_step_features(steps: list[int]) -> list[dict[str, Any]]:
        """Calculate step-related features.

        Args:
            steps: List of daily step counts

        Returns:
            List of step feature dictionaries
        """
        total_steps = sum(steps)
        avg_daily_steps = np.mean(steps)
        peak_daily_steps = max(steps)

        return [
            {
                "feature_name": "total_steps",
                "value": total_steps,
                "unit": "steps",
                "description": f"Total steps over {len(steps)} days",
            },
            {
                "feature_name": "average_daily_steps",
                "value": avg_daily_steps,
                "unit": "steps/day",
                "description": "Average steps per day",
            },
            {
                "feature_name": "peak_daily_steps",
                "value": peak_daily_steps,
                "unit": "steps",
                "description": "Highest single-day step count",
            },
        ]

    @staticmethod
    def _calculate_distance_features(distances: list[float]) -> list[dict[str, Any]]:
        """Calculate distance-related features.

        Args:
            distances: List of daily distances in kilometers

        Returns:
            List of distance feature dictionaries
        """
        total_distance = sum(distances)
        avg_daily_distance = np.mean(distances)

        return [
            {
                "feature_name": "total_distance",
                "value": total_distance,
                "unit": "km",
                "description": f"Total distance over {len(distances)} days",
            },
            {
                "feature_name": "average_daily_distance",
                "value": avg_daily_distance,
                "unit": "km/day",
                "description": "Average distance per day",
            },
        ]

    @staticmethod
    def _calculate_energy_features(active_energy: list[float]) -> list[dict[str, Any]]:
        """Calculate active energy features.

        Args:
            active_energy: List of daily active energy in kcal

        Returns:
            List of energy feature dictionaries
        """
        total_energy = sum(active_energy)
        avg_daily_energy = np.mean(active_energy)

        return [
            {
                "feature_name": "total_active_energy",
                "value": total_energy,
                "unit": "kcal",
                "description": f"Total active calories over {len(active_energy)} days",
            },
            {
                "feature_name": "average_daily_active_energy",
                "value": avg_daily_energy,
                "unit": "kcal/day",
                "description": "Average active calories per day",
            },
        ]

    @staticmethod
    def _calculate_exercise_features(
        exercise_minutes: list[int],
    ) -> list[dict[str, Any]]:
        """Calculate exercise-related features.

        Args:
            exercise_minutes: List of daily exercise minutes

        Returns:
            List of exercise feature dictionaries
        """
        total_exercise = sum(exercise_minutes)
        avg_daily_exercise = np.mean(exercise_minutes)

        return [
            {
                "feature_name": "total_exercise_minutes",
                "value": total_exercise,
                "unit": "minutes",
                "description": f"Total exercise time over {len(exercise_minutes)} days",
            },
            {
                "feature_name": "average_daily_exercise",
                "value": avg_daily_exercise,
                "unit": "minutes/day",
                "description": "Average exercise time per day",
            },
        ]

    @staticmethod
    def _calculate_consistency_score(values: Sequence[int | float]) -> float:
        """Calculate activity consistency score.

        Args:
            values: List of daily values (e.g., steps)

        Returns:
            Consistency score between 0 and 1
        """
        if len(values) < MIN_VALUES_FOR_CONSISTENCY:
            return 1.0

        # Calculate coefficient of variation (CV)
        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val == 0:
            return 0.0

        cv = std_val / mean_val

        # Convert CV to consistency score (lower CV = higher consistency)
        # CV of 0 = score of 1, CV of 1 = score of 0
        return max(0.0, min(1.0, 1.0 - float(cv)))

    def get_summary_stats(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Get summary statistics for activity features.

        Args:
            features: List of calculated features

        Returns:
            Dictionary with summary statistics
        """
        if not features:
            return {"summary": "No activity features calculated"}

        # Extract key metrics
        summary = {
            "total_features": len(features),
            "feature_categories": self._categorize_features(features),
            "processed_at": datetime.now(UTC).isoformat(),
            "processor": self.processor_name,
            "version": self.version,
        }

        # Add key values if available
        for feature in features:
            if feature["feature_name"] == "total_steps":
                summary["total_steps"] = feature["value"]
            elif feature["feature_name"] == "average_daily_steps":
                summary["avg_daily_steps"] = round(feature["value"])
            elif feature["feature_name"] == "total_distance":
                summary["total_distance_km"] = round(feature["value"], 1)
            elif feature["feature_name"] == "total_active_energy":
                summary["total_calories"] = round(feature["value"])

        return summary

    @staticmethod
    def _categorize_features(features: list[dict[str, Any]]) -> dict[str, int]:
        """Categorize features by type.

        Args:
            features: List of features

        Returns:
            Dictionary with category counts
        """
        categories = {"steps": 0, "distance": 0, "energy": 0, "exercise": 0, "other": 0}

        for feature in features:
            name = feature["feature_name"]
            if "step" in name:
                categories["steps"] += 1
            elif "distance" in name:
                categories["distance"] += 1
            elif "energy" in name or "calor" in name:
                categories["energy"] += 1
            elif "exercise" in name or "active_minutes" in name:
                categories["exercise"] += 1
            else:
                categories["other"] += 1

        return categories
