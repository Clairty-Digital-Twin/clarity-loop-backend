"""Depression Risk Analyzer for CLARITY Digital Twin Platform.

Implements state-of-the-art depression prediction based on 2024-2025 research:
- Circadian phase delays predict depression (Lim et al., 2024)
- Activity variability gives 7-day advance warning (Ortiz et al., 2025)
- Personalized baselines improve accuracy (Lipschitz et al., 2025)
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from clarity.ml.circadian_phase_detector import (
    CircadianPhaseDetector,
    CircadianPhaseResult,
)
from clarity.ml.mania_risk_analyzer import ManiaRiskConfig, ManiaRiskResult
from clarity.ml.personal_baseline_tracker import (
    PersonalBaseline,
    PersonalBaselineTracker,
)
from clarity.ml.processors.sleep_processor import SleepFeatures
from clarity.ml.variability_analyzer import VariabilityAnalyzer, VariabilityResult
from clarity.models.health_data import HealthMetric


@dataclass
class DepressionRiskConfig(ManiaRiskConfig):
    """Configuration for depression risk analysis."""

    def __post_init__(self):
        # Depression-specific weights based on research
        self.weights = {
            # Primary predictors
            "circadian_phase_delay": 0.35,  # Lim: AUC 0.80 for depression
            "activity_variability_spike": 0.35,  # Ortiz: 7-day warning
            "social_withdrawal": 0.30,  # Reduced activity patterns
            # Secondary predictors
            "hypersomnia": 0.25,  # Excessive sleep
            "activity_reduction": 0.25,  # Overall decrease
            "sleep_fragmentation": 0.20,  # Poor quality
            # Supporting indicators
            "prolonged_sleep_latency": 0.15,  # Difficulty falling asleep
            "early_morning_awakening": 0.15,  # Classic symptom
            "decreased_hr_variability": 0.10,  # Autonomic dysfunction
            "circadian_flattening": 0.10,  # Amplitude reduction
            # Personal deviation scores
            "sleep_deviation": 0.20,
            "activity_deviation": 0.25,
            "circadian_deviation": 0.20,
        }

        # Depression-specific thresholds
        self.hypersomnia_threshold = 9.0  # Hours of sleep
        self.activity_reduction_ratio = 0.7  # 30% reduction
        self.social_withdrawal_threshold = 0.5  # Interaction metric


class DepressionRiskAnalyzer:
    """Analyzes health data patterns to detect depression risk.

    Mirror structure of ManiaRiskAnalyzer but optimized for depression detection.
    """

    def __init__(self, config_path: Path | None = None, user_id: str | None = None):
        """Initialize depression risk analyzer."""
        self.logger = logging.getLogger(__name__)
        self.user_id = user_id
        self.config = DepressionRiskConfig()

        # Initialize analysis modules
        self.phase_detector = CircadianPhaseDetector()
        self.variability_analyzer = VariabilityAnalyzer()
        self.baseline_tracker = PersonalBaselineTracker()

        # Risk thresholds
        self.moderate_threshold = 0.35
        self.high_threshold = 0.65

        self.logger.info(
            "DepressionRiskAnalyzer initialized",
            extra={
                "user_id": self._sanitize_user_id(user_id),
                "research_basis": "Lim2024, Ortiz2025, Lipschitz2025",
            },
        )

    def analyze(
        self,
        sleep_features: SleepFeatures | None = None,
        pat_metrics: dict[str, float] | None = None,
        activity_stats: dict[str, Any] | None = None,
        cardio_stats: dict[str, float] | None = None,
        historical_baseline: dict[str, float] | None = None,
        user_id: str | None = None,
        recent_health_metrics: list[HealthMetric] | None = None,
        baseline_health_metrics: list[HealthMetric] | None = None,
    ) -> ManiaRiskResult:
        """Analyze multi-modal health data for depression risk indicators.

        Returns ManiaRiskResult for compatibility, but optimized for depression.
        """
        # Initialize scoring
        score = 0.0
        factors = []
        confidence = 1.0

        uid = user_id or self.user_id

        # 1. Circadian Phase Analysis (Lim et al., 2024)
        if recent_health_metrics:
            phase_result = self._analyze_circadian_phase(
                recent_health_metrics, baseline_health_metrics
            )
            if phase_result:
                phase_score, phase_factors = self._score_phase_shift(phase_result)
                score += phase_score
                factors.extend(phase_factors)
                confidence *= phase_result.confidence

        # 2. Activity Variability Analysis (Ortiz et al., 2025)
        if recent_health_metrics:
            personal_baseline = None
            if baseline_health_metrics and uid:
                personal_baseline = self.baseline_tracker.update_baseline(
                    uid, baseline_health_metrics
                )

            variability_result = self._analyze_variability(
                recent_health_metrics, personal_baseline
            )
            if variability_result:
                var_score, var_factors = self._score_variability(variability_result)
                score += var_score
                factors.extend(var_factors)
                confidence *= variability_result.confidence

        # 3. Sleep Pattern Analysis
        sleep_score, sleep_factors = self._analyze_sleep_patterns(
            sleep_features, pat_metrics, historical_baseline
        )
        score += sleep_score
        factors.extend(sleep_factors)

        # 4. Activity Reduction Analysis
        activity_score, activity_factors = self._analyze_activity_reduction(
            activity_stats, historical_baseline, pat_metrics
        )
        score += activity_score
        factors.extend(activity_factors)

        # 5. Physiological Analysis
        physio_score, physio_factors = self._analyze_physiology(cardio_stats)
        score += physio_score
        factors.extend(physio_factors)

        # Determine risk level
        score = max(0.0, min(1.0, score))
        alert_level = self._determine_alert_level(score, confidence)

        # Generate time prediction
        days_until_episode = self._predict_time_to_episode(
            phase_result if "phase_result" in locals() else None,
            variability_result if "variability_result" in locals() else None,
            score,
        )

        # Generate insights and recommendations
        clinical_insight = self._generate_clinical_insight(
            alert_level, factors, score, days_until_episode
        )
        recommendations = self._generate_recommendations(alert_level, factors)

        self.logger.info(
            "Depression risk analysis completed",
            extra={
                "user_id": self._sanitize_user_id(uid),
                "risk_score": round(score, 3),
                "alert_level": alert_level,
                "days_until_episode": days_until_episode,
                "top_factors": factors[:3] if factors else [],
            },
        )

        return ManiaRiskResult(
            risk_score=score,
            alert_level=alert_level,
            contributing_factors=factors,
            confidence=confidence,
            clinical_insight=clinical_insight,
            recommendations=recommendations,
        )

    def _analyze_circadian_phase(
        self,
        recent_metrics: list[HealthMetric],
        baseline_metrics: list[HealthMetric] | None,
    ) -> CircadianPhaseResult | None:
        """Analyze circadian phase shifts for depression."""
        try:
            recent_sleep = [m for m in recent_metrics if m.sleep_data]
            baseline_sleep = (
                [m for m in baseline_metrics if m.sleep_data]
                if baseline_metrics
                else None
            )

            if len(recent_sleep) < 2:
                return None

            return self.phase_detector.detect_phase_shift(recent_sleep, baseline_sleep)
        except Exception as e:
            self.logger.warning(f"Circadian phase analysis failed: {e}")
            return None

    def _analyze_variability(
        self,
        recent_metrics: list[HealthMetric],
        personal_baseline: PersonalBaseline | None,
    ) -> VariabilityResult | None:
        """Analyze activity variability for depression prediction."""
        try:
            activity_metrics = [m for m in recent_metrics if m.activity_data]
            sleep_metrics = [m for m in recent_metrics if m.sleep_data]

            if not activity_metrics and not sleep_metrics:
                return None

            baseline_stats = None
            if personal_baseline:
                baseline_stats = {
                    "activity_variability_baseline": personal_baseline.activity_variability_baseline,
                    "sleep_variability_baseline": personal_baseline.sleep_variability_baseline,
                }

            return self.variability_analyzer.analyze_variability(
                activity_metrics, sleep_metrics, baseline_stats
            )
        except Exception as e:
            self.logger.warning(f"Variability analysis failed: {e}")
            return None

    def _score_phase_shift(
        self, phase_result: CircadianPhaseResult
    ) -> tuple[float, list[str]]:
        """Score circadian phase shift for depression."""
        score = 0.0
        factors = []

        if phase_result.clinical_significance in ["high", "moderate"]:
            if phase_result.phase_shift_direction == "delay":
                # Phase delay predicts depression (AUC 0.80)
                score = self.config.weights["circadian_phase_delay"]
                factors.append(
                    f"Circadian phase delay: {abs(phase_result.phase_shift_hours):.1f}h later"
                )
            elif phase_result.phase_shift_direction == "advance":
                # Phase advance is protective against depression
                score = -self.config.weights["circadian_phase_delay"] * 0.3
                factors.append(
                    f"Circadian phase advance: {abs(phase_result.phase_shift_hours):.1f}h earlier (protective)"
                )

        return score, factors

    def _score_variability(
        self, var_result: VariabilityResult
    ) -> tuple[float, list[str]]:
        """Score variability patterns for depression."""
        score = 0.0
        factors = []

        if var_result.spike_detected:
            if var_result.risk_type == "depression":
                score = self.config.weights["activity_variability_spike"]
                factors.append(
                    f"Activity variability spike detected ({var_result.days_until_risk} days warning)"
                )
            elif var_result.risk_type == "hypomania":
                # Sleep variability suggests mania, not depression
                score *= -0.2
                factors.append(
                    "Sleep variability suggests hypomania risk (not depression)"
                )

        if var_result.variability_trend == "increasing":
            score += self.config.weights["activity_variability_spike"] * 0.3
            factors.append("Increasing behavioral variability")

        return score, factors

    def _analyze_sleep_patterns(
        self,
        sleep: SleepFeatures | None,
        pat_metrics: dict[str, float] | None,
        baseline: dict[str, float] | None,
    ) -> tuple[float, list[str]]:
        """Analyze sleep patterns for depression indicators."""
        score = 0.0
        factors = []

        # Get sleep duration
        hours = None
        if sleep:
            hours = sleep.total_sleep_minutes / 60
        elif pat_metrics and "total_sleep_time" in pat_metrics:
            hours = pat_metrics["total_sleep_time"]

        if hours:
            # Check for hypersomnia
            if hours >= self.config.hypersomnia_threshold:
                score += self.config.weights["hypersomnia"]
                factors.append(f"Excessive sleep: {hours:.1f}h")

            # Check against baseline
            if baseline and "avg_sleep_hours" in baseline:
                baseline_hours = baseline["avg_sleep_hours"]
                if baseline_hours > 0:
                    increase = (hours - baseline_hours) / baseline_hours
                    if increase > 0.3:  # 30% increase
                        score += self.config.weights["hypersomnia"] * 0.5
                        factors.append(
                            f"Sleep increased {int(increase * 100)}% from baseline"
                        )

        # Check sleep quality indicators
        if sleep:
            # Prolonged sleep latency
            if hasattr(sleep, "sleep_latency") and sleep.sleep_latency > 30:
                score += self.config.weights["prolonged_sleep_latency"]
                factors.append(f"Prolonged sleep onset: {sleep.sleep_latency} min")

            # Poor sleep efficiency
            if hasattr(sleep, "sleep_efficiency") and sleep.sleep_efficiency < 0.7:
                score += self.config.weights["sleep_fragmentation"]
                factors.append("Poor sleep quality")

        return score, factors

    def _analyze_activity_reduction(
        self,
        activity: dict[str, Any] | None,
        baseline: dict[str, float] | None,
        pat_metrics: dict[str, float] | None,
    ) -> tuple[float, list[str]]:
        """Analyze activity reduction patterns."""
        score = 0.0
        factors = []

        # Check for activity reduction vs baseline
        if activity and baseline:
            current_steps = activity.get("avg_daily_steps", 0)
            baseline_steps = baseline.get("avg_steps", 0)

            if baseline_steps > 0:
                ratio = current_steps / baseline_steps
                if ratio <= self.config.activity_reduction_ratio:
                    score += self.config.weights["activity_reduction"]
                    reduction_pct = int((1 - ratio) * 100)
                    factors.append(f"Activity reduced {reduction_pct}% from baseline")

        # Check PAT metrics for social withdrawal
        if pat_metrics:
            if "social_interaction_score" in pat_metrics:
                if (
                    pat_metrics["social_interaction_score"]
                    < self.config.social_withdrawal_threshold
                ):
                    score += self.config.weights["social_withdrawal"]
                    factors.append("Reduced social activity patterns")

            # Activity fragmentation can indicate depression
            if "activity_fragmentation" in pat_metrics:
                if (
                    pat_metrics["activity_fragmentation"] < 0.3
                ):  # Very low fragmentation
                    score += self.config.weights["activity_reduction"] * 0.5
                    factors.append("Minimal activity variation")

        return score, factors

    def _analyze_physiology(
        self,
        cardio: dict[str, float] | None,
    ) -> tuple[float, list[str]]:
        """Analyze physiological markers for depression."""
        score = 0.0
        factors = []

        if not cardio:
            return score, factors

        # Decreased HRV is associated with depression
        if "avg_hrv" in cardio:
            hrv = cardio["avg_hrv"]
            if hrv < 30:  # Low HRV
                score += self.config.weights["decreased_hr_variability"]
                factors.append(f"Reduced HRV: {int(hrv)} ms")

        # Circadian amplitude reduction
        if "circadian_rhythm_score" in cardio:
            if cardio["circadian_rhythm_score"] < 0.4:
                score += self.config.weights["circadian_flattening"]
                factors.append("Flattened circadian rhythm")

        return score, factors

    def _determine_alert_level(self, score: float, confidence: float) -> str:
        """Determine depression risk alert level."""
        if confidence < 0.7 and score >= self.high_threshold:
            return "moderate"

        if score >= self.high_threshold:
            return "high"
        if score >= self.moderate_threshold:
            return "moderate"
        if score > 0.1:
            return "low"
        return "none"

    def _predict_time_to_episode(
        self,
        phase_result: CircadianPhaseResult | None,
        var_result: VariabilityResult | None,
        risk_score: float,
    ) -> int | None:
        """Predict days until potential depressive episode."""
        predictions = []

        # Phase delays give gradual onset (Lim)
        if phase_result and phase_result.clinical_significance == "high":
            if phase_result.phase_shift_direction == "delay":
                predictions.append(3)  # 3-5 days typical

        # Activity variability gives 7-day warning (Ortiz)
        if var_result and var_result.days_until_risk:
            predictions.append(var_result.days_until_risk)

        # High risk score suggests sooner onset
        if risk_score > 0.65:
            predictions.append(4)
        elif risk_score > 0.4:
            predictions.append(7)

        return min(predictions) if predictions else None

    def _generate_clinical_insight(
        self, level: str, factors: list[str], score: float, days_until: int | None
    ) -> str:
        """Generate clinical insight for depression risk."""
        if level == "high":
            timing = f" Risk window: {days_until} days." if days_until else ""
            return (
                f"High depression risk detected (score: {score:.2f}).{timing} "
                f"Primary indicators: {', '.join(factors[:2])}. "
                "Consider contacting mental health provider for assessment."
            )
        if level == "moderate":
            timing = f" Monitor for {days_until} days." if days_until else ""
            return (
                f"Moderate depression warning signs (score: {score:.2f}).{timing} "
                "Early intervention strategies recommended."
            )
        if level == "low":
            return (
                f"Mild mood changes detected (score: {score:.2f}). "
                "Continue monitoring and maintain healthy routines."
            )
        return "No significant depression risk indicators detected."

    def _generate_recommendations(
        self,
        level: str,
        factors: list[str],
    ) -> list[str]:
        """Generate recommendations for depression risk."""
        recommendations = []

        if level in ["high", "moderate"]:
            # Immediate recommendations
            if level == "high":
                recommendations.append(
                    "Schedule appointment with mental health provider"
                )
                recommendations.append("Reach out to support network")

            # Phase delay recommendations
            if any("phase delay" in f.lower() for f in factors):
                recommendations.append("Get morning sunlight exposure (30+ minutes)")
                recommendations.append("Avoid screens 2-3 hours before bed")
                recommendations.append("Consider light therapy in morning")

            # Activity recommendations
            if any("activity" in f.lower() for f in factors):
                recommendations.append("Set small daily activity goals")
                recommendations.append("Schedule social activities")
                recommendations.append("Try gentle exercise (walking, yoga)")

            # Sleep recommendations
            if any("sleep" in f.lower() for f in factors):
                recommendations.append("Maintain consistent sleep schedule")
                recommendations.append("Limit daytime naps to 20 minutes")
                recommendations.append("Practice sleep hygiene")

            # General depression prevention
            recommendations.append("Practice mindfulness or meditation")
            if level == "high":
                recommendations.append("Consider crisis resources if needed")

        return recommendations

    def _sanitize_user_id(self, user_id: str | None) -> str:
        """Sanitize user ID for logging."""
        if not user_id:
            return "anonymous"

        if len(user_id) > 8:
            return f"{user_id[:4]}...{user_id[-4:]}"
        return user_id
