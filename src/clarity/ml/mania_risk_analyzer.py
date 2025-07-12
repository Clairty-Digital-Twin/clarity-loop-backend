"""Mania Risk Analyzer for CLARITY Digital Twin Platform.

This module implements RULE-BASED mania/hypomania risk screening using
clinically-established indicators from DSM-5 criteria. This is NOT a diagnostic
tool and has NOT been validated for clinical accuracy.

The rules are based on:
- DSM-5 criteria for manic/hypomanic episodes
- Published research on sleep patterns in bipolar disorder
- Standard clinical observations of activity and physiological changes

IMPORTANT: These are heuristic screening scores, not ML predictions.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import yaml

from clarity.ml.processors.sleep_processor import SleepFeatures


@dataclass
class ManiaRiskConfig:
    """Configuration for mania risk thresholds and weights."""

    # Sleep thresholds
    min_sleep_hours: float = 5.0
    critical_sleep_hours: float = 3.0
    sleep_loss_percent: float = 0.4  # 40% reduction

    # Circadian thresholds
    circadian_disruption_threshold: float = 0.5
    phase_advance_hours: float = 1.0

    # Activity thresholds
    activity_surge_ratio: float = 1.5
    activity_fragmentation_threshold: float = 0.8

    # Physiological thresholds
    elevated_resting_hr: float = 90.0
    low_hrv_threshold: float = 20.0

    # Weights for scoring
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.weights:
            # Weights based on clinical literature - see weight_rationale.md
            self.weights = {
                # Sleep: Harvey 2005 - 40-50% of mania preceded by <3hr sleep
                "severe_sleep_loss": 0.45,
                # Sleep: Wehr 1987 - 25% hypomania risk with 5-7hr sleep
                "acute_sleep_loss": 0.25,
                # Clinical observation - weak standalone indicator
                "rapid_sleep_onset": 0.10,
                # Circadian: Murray 2010 - 80% BD have rhythm instability
                "circadian_disruption": 0.20,
                # Sleep: Bauer 2006 - >2hr variability predicts instability
                "sleep_inconsistency": 0.15,
                # Activity: Merikangas 2019 - prodromal fragmentation
                "activity_fragmentation": 0.20,
                # Activity: DSM-5 criterion B1 - increased goal-directed
                "activity_surge": 0.15,
                # Physiology: weak indicators of sympathetic activation
                "elevated_hr": 0.05,
                "low_hrv": 0.05,
                # Circadian: Wehr 2018 - phase advance precedes 65% of mania
                "circadian_phase_advance": 0.40,
            }


class ManiaRiskResult(BaseModel):
    """Result of mania risk analysis."""

    risk_score: float = Field(ge=0.0, le=1.0, description="Overall mania risk score")
    alert_level: str = Field(description="Risk level: none/low/moderate/high")
    contributing_factors: list[str] = Field(description="Factors contributing to risk")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in assessment")
    clinical_insight: str = Field(description="Clinical interpretation")
    recommendations: list[str] = Field(description="Actionable recommendations")


class ManiaRiskAnalyzer:
    """Analyzes health data patterns to detect mania/hypomania risk."""

    def __init__(self, config_path: Path | None = None, user_id: str | None = None) -> None:
        """Initialize with configuration.

        Args:
            config_path: Path to YAML configuration file
            user_id: User ID for alert tracking (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.audit_logger = logging.getLogger("clarity.audit.mania_risk")
        self.user_id = user_id
        # Use OrderedDict with max size for memory safety
        self._last_high_alert_cache: OrderedDict[str, datetime] = OrderedDict()
        self._max_cache_size = 1000  # Maximum cache entries

        if config_path and config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
                # Handle nested weights if present
                weights = config_dict.pop("weights") if "weights" in config_dict else {}
                self.config = ManiaRiskConfig(**config_dict, weights=weights)
        else:
            self.config = ManiaRiskConfig()

        self.moderate_threshold = 0.4
        self.high_threshold = 0.7

        # Constants for magic values
        self.MIN_DATA_DAYS = 3
        self.HOURS_IN_DAY = 24
        self.RAPID_SLEEP_ONSET_THRESHOLD = 5  # minutes
        self.SLEEP_CONSISTENCY_THRESHOLD = 0.4
        self.DATA_ESTIMATION_CONFIDENCE_FACTOR = 0.8
        self.INVALID_DATA_CONFIDENCE = 0.3
        self.LIMITED_DATA_CONFIDENCE = 0.5
        self.LOW_CONFIDENCE_THRESHOLD = 0.7
        self.LOW_RISK_THRESHOLD = 0.1
        self.PHASE_SHIFT_FACTOR = 2
        self.SLEEP_DEFICIT_SCALING_FACTOR = 0.6
        self.MIN_PATTERN_DAYS = 8
        self.USER_ID_TRUNCATE_LENGTH = 8
        self.PRIMARY_FACTORS_COUNT = 2

        self.logger.info(
            "ManiaRiskAnalyzer initialized",
            extra={
                "user_id": self._sanitize_user_id(user_id),
                "config_path": str(config_path) if config_path else "default",
                "thresholds": {
                    "moderate": self.moderate_threshold,
                    "high": self.high_threshold,
                },
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
    ) -> ManiaRiskResult:
        """Analyze multi-modal health data for mania risk indicators.

        Args:
            sleep_features: Processed sleep metrics from SleepProcessor
            pat_metrics: Metrics from PAT model analysis
            activity_stats: Activity statistics from ActivityProcessor
            cardio_stats: Cardiovascular metrics from CardioProcessor
            historical_baseline: User's personal baseline (28-day averages)
            user_id: User ID for alert tracking

        Returns:
            ManiaRiskResult with score, level, and clinical insights
        """
        analysis_start = datetime.now(UTC)
        score = 0.0
        factors = []
        confidence = 1.0
        component_scores = {}

        # Use provided user_id or instance user_id
        uid = user_id or self.user_id

        # Log analysis start with available data sources
        self.logger.info(
            "Starting mania risk analysis",
            extra={
                "user_id": self._sanitize_user_id(uid),
                "data_sources": {
                    "sleep_features": sleep_features is not None,
                    "pat_metrics": pat_metrics is not None,
                    "activity_stats": activity_stats is not None,
                    "cardio_stats": cardio_stats is not None,
                    "historical_baseline": historical_baseline is not None,
                },
            },
        )

        # 1. Sleep Analysis
        sleep_score, sleep_factors, sleep_conf = self._analyze_sleep(
            sleep_features, pat_metrics, historical_baseline
        )
        score += sleep_score
        factors.extend(sleep_factors)
        confidence *= sleep_conf
        component_scores["sleep"] = sleep_score

        # Log sleep analysis results
        if sleep_score > 0:
            self.logger.info(
                "Sleep analysis completed",
                extra={
                    "user_id": self._sanitize_user_id(uid),
                    "component": "sleep",
                    "score": round(sleep_score, 3),
                    "factors": sleep_factors,
                    "confidence": round(sleep_conf, 3),
                },
            )

        # 2. Circadian Rhythm Analysis
        circadian_score, circadian_factors = self._analyze_circadian(
            sleep_features, pat_metrics, cardio_stats
        )
        score += circadian_score
        factors.extend(circadian_factors)
        component_scores["circadian"] = circadian_score

        # Log circadian analysis results
        if circadian_score > 0:
            self.logger.info(
                "Circadian analysis completed",
                extra={
                    "user_id": self._sanitize_user_id(uid),
                    "component": "circadian",
                    "score": round(circadian_score, 3),
                    "factors": circadian_factors,
                },
            )

        # 3. Activity Analysis
        activity_score, activity_factors = self._analyze_activity(
            pat_metrics, activity_stats, historical_baseline
        )
        score += activity_score
        factors.extend(activity_factors)
        component_scores["activity"] = activity_score

        # Log activity analysis results
        if activity_score > 0:
            self.logger.info(
                "Activity analysis completed",
                extra={
                    "user_id": self._sanitize_user_id(uid),
                    "component": "activity",
                    "score": round(activity_score, 3),
                    "factors": activity_factors,
                },
            )

        # 4. Physiological Analysis
        physio_score, physio_factors = self._analyze_physiology(cardio_stats)
        score += physio_score
        factors.extend(physio_factors)
        component_scores["physiology"] = physio_score

        # Log physiology analysis results
        if physio_score > 0:
            self.logger.info(
                "Physiology analysis completed",
                extra={
                    "user_id": self._sanitize_user_id(uid),
                    "component": "physiology",
                    "score": round(physio_score, 3),
                    "factors": physio_factors,
                },
            )

        # 5. Temporal Pattern Analysis
        temporal_score, temporal_factors = self._analyze_temporal_patterns(
            sleep_features, activity_stats, historical_baseline
        )
        score += temporal_score
        factors.extend(temporal_factors)

        # Handle case with insufficient data
        if len(factors) == 1 and "Insufficient sleep data" in factors[0]:
            return ManiaRiskResult(
                risk_score=0.0,
                alert_level="none",
                contributing_factors=["Insufficient sleep data"],
                confidence=0.5,
                clinical_insight="Insufficient data for mania risk assessment",
                recommendations=[],
            )

        # Clamp score and determine alert level
        score = max(0.0, min(1.0, score))
        alert_level = self._determine_alert_level(score, confidence)

        # Check for rate limiting on high alerts
        if uid and alert_level == "high":
            is_duplicate = self._check_duplicate_high_alert(uid)
            if is_duplicate:
                self.logger.warning(
                    "High alert rate limited",
                    extra={
                        "user_id": self._sanitize_user_id(uid),
                        "original_level": "high",
                        "adjusted_level": "moderate",
                        "reason": "duplicate_high_alert_within_24h",
                    },
                )
                alert_level = "moderate"  # Downgrade to prevent alert fatigue

        # Generate clinical insight and recommendations
        clinical_insight = self._generate_clinical_insight(alert_level, factors, score)
        recommendations = self._generate_recommendations(alert_level, factors)

        # Calculate analysis duration
        analysis_duration = (datetime.now(UTC) - analysis_start).total_seconds()

        # Log final analysis results
        self.logger.info(
            "Mania risk analysis completed",
            extra={
                "user_id": self._sanitize_user_id(uid),
                "risk_score": round(score, 3),
                "alert_level": alert_level,
                "confidence": round(confidence, 3),
                "component_scores": {
                    k: round(v, 3) for k, v in component_scores.items()
                },
                "num_factors": len(factors),
                "top_factors": factors[:3] if factors else [],
                "analysis_duration_seconds": round(analysis_duration, 3),
                "has_recommendations": len(recommendations) > 0,
            },
        )

        # Log high-risk alerts separately for monitoring
        if alert_level in {"moderate", "high"}:
            self.logger.warning(
                "Elevated mania risk detected: %s", alert_level,
                extra={
                    "user_id": self._sanitize_user_id(uid),
                    "alert_type": "mania_risk",
                    "severity": alert_level,
                    "risk_score": round(score, 3),
                    "primary_factors": factors[:2] if factors else [],
                    "clinical_action_required": alert_level == "high",
                },
            )

            # HIPAA-compliant audit logging for high alerts
            if alert_level == "high":
                self.audit_logger.info(
                    "HIGH_RISK_DETECTION",
                    extra={
                        "timestamp": datetime.now(UTC).isoformat(),
                        "user_id_hash": self._sanitize_user_id(uid),
                        "risk_score": round(score, 3),
                        "event_type": "mania_high_risk_alert",
                        "data_sources": list(component_scores.keys()),
                        "confidence": round(confidence, 3),
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

    def _analyze_sleep(
        self,
        sleep: SleepFeatures | None,
        pat_metrics: dict[str, float] | None,
        baseline: dict[str, float] | None,
    ) -> tuple[float, list[str], float]:
        """Analyze sleep patterns for mania indicators."""
        score = 0.0
        factors = []
        confidence = 1.0
        data_completeness = 1.0

        # Get sleep duration from available sources
        hours = None
        data_source = None

        if sleep is not None:
            hours = sleep.total_sleep_minutes / 60
            data_source = "HealthKit"

            # Check data density - if we have sleep object, check for completeness
            if hasattr(sleep, "data_coverage_days") and sleep.data_coverage_days < self.MIN_DATA_DAYS:
                # If less than 3 days of data in the past week, lower confidence
                data_completeness = self.LIMITED_DATA_CONFIDENCE
                factors.append(
                    f"Limited data: only {sleep.data_coverage_days} days"
                )

        elif pat_metrics and "total_sleep_time" in pat_metrics:
            hours = pat_metrics["total_sleep_time"]
            data_source = "PAT estimation"
            confidence *= self.DATA_ESTIMATION_CONFIDENCE_FACTOR  # Lower confidence for estimated data
            factors.append("PAT estimation")

        if hours is None:
            return 0.0, ["Insufficient sleep data"], 0.5

        # Unit conversion guardrails - bounds checking
        if not (0 < hours <= self.HOURS_IN_DAY):
            self.logger.warning(
                "Invalid sleep hours detected",
                extra={
                    "user_id": self._sanitize_user_id(self.user_id),
                    "sleep_hours": hours,
                    "data_source": data_source,
                },
            )
            return 0.0, [f"Invalid sleep data: {hours} hours recorded"], self.INVALID_DATA_CONFIDENCE

        # Apply data completeness factor to confidence
        confidence *= data_completeness

        # Check against absolute thresholds
        if hours <= self.config.critical_sleep_hours:
            score += self.config.weights["severe_sleep_loss"]
            factors.append(f"Critically low sleep: {hours:.1f}h ({data_source})")
        elif hours < self.config.min_sleep_hours:
            # Scale the score based on how low the sleep is
            # 4.5 hours should give more score than 4.9 hours
            sleep_deficit_ratio = (
                self.config.min_sleep_hours - hours
            ) / self.config.min_sleep_hours
            score += self.config.weights["acute_sleep_loss"] * (
                1 + sleep_deficit_ratio * self.SLEEP_DEFICIT_SCALING_FACTOR
            )
            factors.append(f"Low sleep duration: {hours:.1f}h ({data_source})")

        # Check against personal baseline if available
        if baseline and "avg_sleep_hours" in baseline:
            baseline_hours = baseline["avg_sleep_hours"]
            if baseline_hours > 0:
                reduction = 1 - (hours / baseline_hours)
                if reduction >= self.config.sleep_loss_percent:
                    score += self.config.weights["acute_sleep_loss"]
                    factors.append(
                        f"Sleep reduced {int(reduction * 100)}% from baseline"
                    )

        # Check sleep latency (rapid sleep onset during mania)
        # Ensure sleep_latency is in minutes
        if sleep and hasattr(sleep, "sleep_latency") and sleep.sleep_latency < self.RAPID_SLEEP_ONSET_THRESHOLD:
            score += self.config.weights["rapid_sleep_onset"]
            factors.append("Very rapid sleep onset (<5 min)")

        # Check sleep consistency
        if (
            sleep
            and hasattr(sleep, "consistency_score")
            and sleep.consistency_score < self.SLEEP_CONSISTENCY_THRESHOLD
        ):
            score += self.config.weights["sleep_inconsistency"]
            factors.append("Irregular sleep schedule")

        return score, factors, confidence

    def _analyze_circadian(
        self,
        sleep: SleepFeatures | None,
        pat_metrics: dict[str, float] | None,
        cardio: dict[str, float] | None,
    ) -> tuple[float, list[str]]:
        """Analyze circadian rhythm disruption."""
        score = 0.0
        factors = []

        # Check circadian rhythm score
        circadian_score = None
        if pat_metrics and "circadian_rhythm_score" in pat_metrics:
            circadian_score = pat_metrics["circadian_rhythm_score"]
        elif cardio and "circadian_rhythm_score" in cardio:
            circadian_score = cardio["circadian_rhythm_score"]

        if circadian_score is not None and circadian_score < self.config.circadian_disruption_threshold:
            score += self.config.weights["circadian_disruption"]
            factors.append(
                f"Disrupted circadian rhythm (score: {circadian_score:.2f})"
            )

        # Sleep consistency is already checked in _analyze_sleep
        # Skip it here to avoid double-counting

        return score, factors

    def _analyze_activity(
        self,
        pat_metrics: dict[str, float] | None,
        activity: dict[str, Any] | None,
        baseline: dict[str, float] | None,
    ) -> tuple[float, list[str]]:
        """Analyze activity patterns for mania indicators."""
        score = 0.0
        factors = []

        # Check activity fragmentation from PAT
        if pat_metrics and "activity_fragmentation" in pat_metrics:
            frag = pat_metrics["activity_fragmentation"]
            if frag > self.config.activity_fragmentation_threshold:
                score += self.config.weights["activity_fragmentation"]
                factors.append(f"High activity fragmentation: {frag:.2f}")

        # Check for activity surges
        if activity and baseline:
            # Compare current vs baseline steps
            current_steps = activity.get("avg_daily_steps", 0)
            baseline_steps = baseline.get("avg_steps", 0)
            if baseline_steps > 0:
                ratio = current_steps / baseline_steps
                if ratio >= self.config.activity_surge_ratio:
                    score += self.config.weights["activity_surge"]
                    factors.append(f"Activity surge: {ratio:.1f}x baseline")

        return score, factors

    def _analyze_physiology(
        self,
        cardio: dict[str, float] | None,
    ) -> tuple[float, list[str]]:
        """Analyze physiological markers."""
        score = 0.0
        factors: list[str] = []

        if not cardio:
            return score, factors

        # Check elevated heart rate
        if "resting_hr" in cardio:
            hr = cardio["resting_hr"]
            if hr > self.config.elevated_resting_hr:
                score += self.config.weights["elevated_hr"]
                factors.append(f"Elevated resting HR: {int(hr)} bpm")

        # Check reduced HRV
        if "avg_hrv" in cardio:
            hrv = cardio["avg_hrv"]
            if hrv < self.config.low_hrv_threshold:
                score += self.config.weights["low_hrv"]
                factors.append(f"Low HRV: {int(hrv)} ms")

        return score, factors

    def _analyze_temporal_patterns(
        self,
        sleep: SleepFeatures | None,
        activity: dict[str, Any] | None,
        baseline: dict[str, float] | None,
    ) -> tuple[float, list[str]]:
        """Analyze temporal patterns and trends."""
        score = 0.0
        factors: list[str] = []

        # This is where we would implement:
        # - Trend analysis (sleep getting progressively worse)
        # - Intra-day variability analysis
        # - Phase advance detection
        # For now, placeholder for future enhancement

        return score, factors

    def _determine_alert_level(self, score: float, confidence: float = 1.0) -> str:
        """Determine categorical alert level from risk score.

        Args:
            score: The risk score (0-1)
            confidence: Confidence in the assessment (0-1)

        Returns:
            Alert level string
        """
        # Apply guardrail: if confidence is low, cap at moderate
        if confidence < self.LOW_CONFIDENCE_THRESHOLD and score >= self.high_threshold:
            return "moderate"

        if score >= self.high_threshold:
            return "high"
        if score >= self.moderate_threshold:
            return "moderate"
        if score > self.LOW_RISK_THRESHOLD:
            return "low"
        return "none"

    def _generate_clinical_insight(
        self,
        level: str,
        factors: list[str],
        score: float,
    ) -> str:
        """Generate clinical insight message."""
        if level == "high":
            primary_factors = factors[:self.PRIMARY_FACTORS_COUNT] if len(factors) > self.PRIMARY_FACTORS_COUNT else factors
            factors_str = " and ".join(primary_factors).lower()
            return (
                f"Elevated mania risk detected (score: {score:.2f}) - "
                f"{factors_str}. Consider contacting your healthcare provider."
            )
        if level == "moderate":
            return (
                f"Moderate mania warning signs observed (score: {score:.2f}). "
                "Monitor symptoms closely and maintain sleep routine."
            )
        if level == "low":
            return (
                f"Mild irregularities detected (score: {score:.2f}). "
                "Continue healthy sleep and activity habits."
            )
        return "No significant mania risk indicators detected."

    def _generate_recommendations(
        self,
        level: str,
        factors: list[str],
    ) -> list[str]:
        """Generate actionable recommendations based on risk factors."""
        recommendations = []

        if level in {"high", "moderate"}:
            # General high-risk recommendations first
            if level == "high":
                recommendations.append(
                    "Contact your healthcare provider within 24 hours"
                )

            # Sleep-related recommendations
            if any("sleep" in f.lower() for f in factors):
                recommendations.append(
                    "Prioritize sleep: aim for 7-8 hours at consistent times"
                )
                if level == "high":
                    recommendations.append("Avoid screens 2 hours before bedtime")

            # Activity-related recommendations
            if any("activity" in f.lower() for f in factors):
                recommendations.append("Monitor activity levels - avoid overcommitment")
                if level == "high":
                    recommendations.append(
                        "Schedule regular rest periods throughout the day"
                    )

            # Circadian-related recommendations
            if any("circadian" in f.lower() for f in factors):
                recommendations.append("Maintain consistent wake/sleep times")
                if level == "high":
                    recommendations.append("Get morning sunlight exposure")

            # Additional high-risk recommendations
            if level == "high":
                recommendations.append("Avoid major decisions or commitments")

        return recommendations

    def _sanitize_user_id(self, user_id: str | None) -> str:
        """Sanitize user ID for logging to protect PHI.

        Args:
            user_id: Raw user ID

        Returns:
            Sanitized user ID safe for logging
        """
        if not user_id:
            return "anonymous"

        # For production, you might hash the user_id or use first/last few chars
        # For now, we'll use a simple truncation
        if len(user_id) > self.USER_ID_TRUNCATE_LENGTH:
            return f"{user_id[:4]}...{user_id[-4:]}"
        return user_id

    def _check_duplicate_high_alert(self, user_id: str) -> bool:
        """Check if user had a high alert in the last 24 hours.

        Args:
            user_id: User identifier

        Returns:
            True if duplicate high alert within 24 hours
        """
        now = datetime.now(UTC)
        cutoff_time = now - timedelta(hours=24)

        # Check in-memory cache (for production, use DynamoDB or Redis)
        if user_id in self._last_high_alert_cache:
            last_alert_time = self._last_high_alert_cache[user_id]
            if last_alert_time > cutoff_time:
                return True

        # Update cache with new alert time (with memory safety)
        self._last_high_alert_cache[user_id] = now

        # Enforce max cache size for memory safety
        if len(self._last_high_alert_cache) > self._max_cache_size:
            # Remove oldest entries (FIFO)
            while len(self._last_high_alert_cache) > self._max_cache_size * 0.8:
                self._last_high_alert_cache.popitem(last=False)

        # Clean expired entries periodically
        if len(self._last_high_alert_cache) > self._max_cache_size * 0.5:
            self._cleanup_alert_cache(cutoff_time)

        return False

    def _cleanup_alert_cache(self, cutoff_time: datetime) -> None:
        """Remove old entries from alert cache.

        Args:
            cutoff_time: Remove entries older than this time
        """
        self._last_high_alert_cache = OrderedDict(
            (uid, alert_time)
            for uid, alert_time in self._last_high_alert_cache.items()
            if alert_time > cutoff_time
        )
