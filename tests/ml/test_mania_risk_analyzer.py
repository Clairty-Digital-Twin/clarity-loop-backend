"""Tests for Mania Risk Analyzer - Test-Driven Development.

Following TDD principles, we write tests first to define expected behavior
for our mania risk detection module based on 2024-2025 research.
"""

from datetime import UTC, datetime
from pathlib import Path
import tempfile

import pytest
import yaml

from clarity.ml.processors.sleep_processor import SleepFeatures


class TestManiaRiskAnalyzer:
    """Test suite for mania risk detection following clinical research."""

    def test_analyzer_initialization(self):
        """Test that analyzer initializes with default configuration."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()
        assert analyzer is not None
        assert analyzer.config is not None
        assert analyzer.config.min_sleep_hours == 5.0
        assert analyzer.config.critical_sleep_hours == 3.0
        assert analyzer.moderate_threshold == 0.4
        assert analyzer.high_threshold == 0.7

    def test_no_risk_with_healthy_sleep(self):
        """Test that healthy sleep patterns produce no risk."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        # Healthy sleep: 7.5 hours, good efficiency
        healthy_sleep = SleepFeatures(
            total_sleep_minutes=450,  # 7.5 hours
            sleep_efficiency=0.85,
            sleep_latency=15.0,
            awakenings_count=2.0,
            consistency_score=0.8,
            overall_quality_score=0.85,
            data_coverage_days=7,  # Full week of data
        )

        result = analyzer.analyze(
            sleep_features=healthy_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.85,
                "activity_fragmentation": 0.3,
            },
        )

        assert result.risk_score < 0.1
        assert result.alert_level == "none"
        assert len(result.contributing_factors) == 0
        assert "No significant mania risk" in result.clinical_insight

    def test_high_risk_severe_sleep_loss(self):
        """Test that severe sleep loss (3h) triggers high risk alert."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        # Manic pattern: 3 hours sleep, falls asleep instantly
        manic_sleep = SleepFeatures(
            total_sleep_minutes=180,  # 3 hours - critical threshold
            sleep_efficiency=0.95,  # Paradoxically high
            sleep_latency=2.0,  # Falls asleep instantly
            awakenings_count=0.0,
            consistency_score=0.2,  # Very irregular schedule
            overall_quality_score=0.4,
            data_coverage_days=7,  # Full week of data
        )

        result = analyzer.analyze(
            sleep_features=manic_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.3,  # Disrupted
                "activity_fragmentation": 0.9,  # High fragmentation
            },
        )

        assert result.risk_score >= 0.7
        assert result.alert_level == "high"
        assert any(
            "critically low sleep" in f.lower() for f in result.contributing_factors
        )
        assert any("circadian" in f.lower() for f in result.contributing_factors)
        assert "healthcare provider" in result.clinical_insight.lower()
        assert len(result.recommendations) >= 3
        assert any(
            "Contact your healthcare provider" in r for r in result.recommendations
        )

    def test_moderate_risk_below_threshold(self):
        """Test that 4.5 hours sleep triggers moderate risk."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        # Borderline pattern: 4.5 hours sleep
        borderline_sleep = SleepFeatures(
            total_sleep_minutes=270,  # 4.5 hours
            sleep_efficiency=0.80,
            sleep_latency=10.0,
            awakenings_count=3.0,
            consistency_score=0.6,
            overall_quality_score=0.65,
            data_coverage_days=7,  # Full week of data
        )

        result = analyzer.analyze(
            sleep_features=borderline_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.45,  # Slightly disrupted
                "activity_fragmentation": 0.75,  # Moderate fragmentation
            },
        )

        assert 0.4 <= result.risk_score < 0.7
        assert result.alert_level == "moderate"
        assert any("low sleep" in f.lower() for f in result.contributing_factors)
        assert "Monitor symptoms closely" in result.clinical_insight

    def test_activity_surge_contributes_to_risk(self):
        """Test that activity surge relative to baseline increases risk."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        result = analyzer.analyze(
            pat_metrics={
                "total_sleep_time": 6.0,  # Slightly reduced but not critical
                "circadian_rhythm_score": 0.7,
                "activity_fragmentation": 0.5,
            },
            activity_stats={
                "avg_daily_steps": 15000,
            },
            historical_baseline={
                "avg_steps": 8000,  # Nearly 2x baseline
            },
        )

        assert result.risk_score >= 0.1  # Some risk due to activity surge
        assert any("activity surge" in f.lower() for f in result.contributing_factors)
        assert "1.9x baseline" in str(result.contributing_factors)

    def test_circadian_disruption_detection(self):
        """Test that circadian rhythm disruption is properly detected."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        result = analyzer.analyze(
            pat_metrics={
                "total_sleep_time": 6.5,  # Normal-ish sleep duration
                "circadian_rhythm_score": 0.3,  # Very disrupted
                "activity_fragmentation": 0.4,
            }
        )

        assert result.risk_score >= 0.20  # Moderate contribution from circadian
        assert any("circadian rhythm" in f.lower() for f in result.contributing_factors)
        assert "0.30" in str(result.contributing_factors) or "0.3" in str(
            result.contributing_factors
        )

    def test_physiological_markers_support(self):
        """Test that elevated HR and low HRV add to risk score."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        result = analyzer.analyze(
            pat_metrics={
                "total_sleep_time": 5.5,  # Slightly low
                "circadian_rhythm_score": 0.7,
                "activity_fragmentation": 0.5,
            },
            cardio_stats={
                "resting_hr": 95,  # Elevated
                "avg_hrv": 15,  # Low
            },
        )

        # Should have base risk from sleep plus physio markers
        assert any(
            "elevated resting hr" in f.lower() for f in result.contributing_factors
        )
        assert any("low hrv" in f.lower() for f in result.contributing_factors)
        assert "95 bpm" in str(result.contributing_factors)
        assert "15 ms" in str(result.contributing_factors)

    def test_sleep_reduction_from_baseline(self):
        """Test detection of sleep reduction relative to personal baseline."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        result = analyzer.analyze(
            sleep_features=SleepFeatures(
                total_sleep_minutes=240,  # 4 hours
                sleep_efficiency=0.85,
                sleep_latency=10.0,
                awakenings_count=1.0,
                consistency_score=0.7,
                overall_quality_score=0.75,
                data_coverage_days=7,  # Full week of data
            ),
            historical_baseline={
                "avg_sleep_hours": 8.0,  # Normal is 8 hours
            },
        )

        # 50% reduction from baseline should trigger
        assert any("50% from baseline" in f for f in result.contributing_factors)
        assert result.risk_score >= 0.4  # At least moderate

    def test_rapid_sleep_onset_pattern(self):
        """Test that very rapid sleep onset (<5 min) in context of low sleep adds risk."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        result = analyzer.analyze(
            sleep_features=SleepFeatures(
                total_sleep_minutes=240,  # 4 hours
                sleep_efficiency=0.90,
                sleep_latency=3.0,  # Very rapid
                awakenings_count=0.0,
                consistency_score=0.5,
                overall_quality_score=0.6,
                data_coverage_days=7,  # Full week of data
            )
        )

        assert any(
            "rapid sleep onset" in f.lower() for f in result.contributing_factors
        )
        assert "<5 min" in str(result.contributing_factors)

    def test_confidence_reduced_with_estimated_data(self):
        """Test that confidence is lower when using PAT estimates vs actual sleep data."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        # Using only PAT metrics (no direct sleep data)
        result = analyzer.analyze(
            pat_metrics={
                "total_sleep_time": 3.0,  # Critically low
                "circadian_rhythm_score": 0.3,
                "activity_fragmentation": 0.8,
            }
        )

        assert result.confidence < 1.0  # Should be 0.8 based on implementation
        assert result.risk_score >= 0.6  # Still significant risk
        assert "PAT estimation" in str(result.contributing_factors)

    def test_recommendations_for_high_risk(self):
        """Test that appropriate recommendations are generated for high risk."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        result = analyzer.analyze(
            sleep_features=SleepFeatures(
                total_sleep_minutes=150,  # 2.5 hours
                sleep_efficiency=0.9,
                sleep_latency=2.0,
                awakenings_count=0.0,
                consistency_score=0.2,
                overall_quality_score=0.3,
                data_coverage_days=7,  # Full week of data
            ),
            pat_metrics={
                "circadian_rhythm_score": 0.3,
                "activity_fragmentation": 0.85,
            },
        )

        assert result.alert_level == "high"
        recommendations = result.recommendations

        # Should have urgent care recommendation first
        assert recommendations[0] == "Contact your healthcare provider within 24 hours"

        # Should have sleep-related recommendations
        assert any("sleep" in r.lower() for r in recommendations)
        assert any("7-8 hours" in r for r in recommendations)

        # Should have activity recommendations
        assert any("activity" in r.lower() for r in recommendations)

        # Should have circadian recommendations
        assert any("consistent wake/sleep times" in r.lower() for r in recommendations)

    @pytest.mark.parametrize(
        ("sleep_hours", "expected_level"),
        [
            (8.0, "none"),  # Normal sleep
            (7.0, "none"),  # Good sleep
            (6.0, "none"),  # Slightly reduced but ok
            (5.5, "none"),  # Just above threshold
            (4.8, "low"),  # Below 5h threshold but not severe
            (4.0, "low"),  # Clearly reduced
            (3.0, "moderate"),  # At critical threshold
            (2.0, "moderate"),  # Severe insomnia
            (0.5, "moderate"),  # Near-zero sleep
        ],
    )
    def test_sleep_duration_thresholds(self, sleep_hours, expected_level):
        """Test various sleep duration thresholds map to correct alert levels."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        result = analyzer.analyze(
            pat_metrics={
                "total_sleep_time": sleep_hours,
                "circadian_rhythm_score": 0.7,  # Normal circadian
                "activity_fragmentation": 0.5,  # Normal activity
            }
        )

        assert (
            result.alert_level == expected_level
        ), f"Expected {expected_level} for {sleep_hours}h sleep, got {result.alert_level}"

    def test_insufficient_data_handling(self):
        """Test graceful handling when insufficient data is available."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        # No sleep data at all
        result = analyzer.analyze()

        assert result.risk_score == 0.0
        assert result.alert_level == "none"
        assert result.confidence == 0.5
        assert "Insufficient sleep data" in result.contributing_factors

    def test_yaml_config_parsing(self):
        """Test that YAML configuration file is parsed correctly."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        # Test with the actual config file
        config_path = Path("config/mania_risk_config.yaml")
        analyzer = ManiaRiskAnalyzer(config_path)

        # Verify thresholds are loaded correctly
        assert analyzer.config.min_sleep_hours == 5.0
        assert analyzer.config.critical_sleep_hours == 3.0
        assert analyzer.config.sleep_loss_percent == 0.4
        assert analyzer.config.circadian_disruption_threshold == 0.5
        assert analyzer.config.phase_advance_hours == 1.0
        assert analyzer.config.activity_surge_ratio == 1.5
        assert analyzer.config.activity_fragmentation_threshold == 0.8
        assert analyzer.config.elevated_resting_hr == 90.0
        assert analyzer.config.low_hrv_threshold == 20.0

        # Verify weights are loaded correctly
        assert analyzer.config.weights["severe_sleep_loss"] == 0.40
        assert analyzer.config.weights["acute_sleep_loss"] == 0.30
        assert analyzer.config.weights["rapid_sleep_onset"] == 0.10
        assert analyzer.config.weights["circadian_disruption"] == 0.25
        assert analyzer.config.weights["sleep_inconsistency"] == 0.10
        assert analyzer.config.weights["circadian_phase_advance"] == 0.15
        assert analyzer.config.weights["activity_fragmentation"] == 0.20
        assert analyzer.config.weights["activity_surge"] == 0.10
        assert analyzer.config.weights["elevated_hr"] == 0.05
        assert analyzer.config.weights["low_hrv"] == 0.05

    def test_custom_yaml_config_parsing(self):
        """Test parsing of custom YAML configuration with different values."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        # Create a temporary YAML config file with custom values
        custom_config = {
            "min_sleep_hours": 6.0,
            "critical_sleep_hours": 4.0,
            "sleep_loss_percent": 0.5,
            "circadian_disruption_threshold": 0.6,
            "phase_advance_hours": 2.0,
            "activity_surge_ratio": 2.0,
            "activity_fragmentation_threshold": 0.9,
            "elevated_resting_hr": 95.0,
            "low_hrv_threshold": 25.0,
            "weights": {
                "severe_sleep_loss": 0.50,
                "acute_sleep_loss": 0.35,
                "rapid_sleep_onset": 0.15,
                "circadian_disruption": 0.30,
                "sleep_inconsistency": 0.15,
                "circadian_phase_advance": 0.20,
                "activity_fragmentation": 0.25,
                "activity_surge": 0.15,
                "elevated_hr": 0.10,
                "low_hrv": 0.10,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.dump(custom_config, f)
            temp_path = Path(f.name)

        try:
            # Load analyzer with custom config
            analyzer = ManiaRiskAnalyzer(temp_path)

            # Verify custom thresholds
            assert analyzer.config.min_sleep_hours == 6.0
            assert analyzer.config.critical_sleep_hours == 4.0
            assert analyzer.config.sleep_loss_percent == 0.5
            assert analyzer.config.circadian_disruption_threshold == 0.6
            assert analyzer.config.phase_advance_hours == 2.0
            assert analyzer.config.activity_surge_ratio == 2.0
            assert analyzer.config.activity_fragmentation_threshold == 0.9
            assert analyzer.config.elevated_resting_hr == 95.0
            assert analyzer.config.low_hrv_threshold == 25.0

            # Verify custom weights
            assert analyzer.config.weights["severe_sleep_loss"] == 0.50
            assert analyzer.config.weights["acute_sleep_loss"] == 0.35
            assert analyzer.config.weights["rapid_sleep_onset"] == 0.15
            assert analyzer.config.weights["circadian_disruption"] == 0.30
            assert analyzer.config.weights["sleep_inconsistency"] == 0.15
            assert analyzer.config.weights["circadian_phase_advance"] == 0.20
            assert analyzer.config.weights["activity_fragmentation"] == 0.25
            assert analyzer.config.weights["activity_surge"] == 0.15
            assert analyzer.config.weights["elevated_hr"] == 0.10
            assert analyzer.config.weights["low_hrv"] == 0.10

        finally:
            # Clean up temp file
            temp_path.unlink()

    def test_data_density_guardrails(self):
        """Test that low data density prevents high alerts."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        # Create sleep features with critically low sleep that would normally trigger high alert
        critical_sleep = SleepFeatures(
            total_sleep_minutes=150,  # 2.5 hours - very low
            sleep_efficiency=0.95,
            sleep_latency=2.0,
            awakenings_count=0.0,
            consistency_score=0.2,
            overall_quality_score=0.4,
            data_coverage_days=2,  # Only 2 days of data
        )

        result = analyzer.analyze(
            sleep_features=critical_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.3,
                "activity_fragmentation": 0.9,
            },
        )

        # Should have lowered confidence due to limited data
        assert result.confidence < 0.7
        # Alert should be capped at moderate due to low confidence
        assert result.alert_level == "moderate"
        assert any("Limited data" in f for f in result.contributing_factors)

    @pytest.mark.parametrize(
        ("test_case", "expected"),
        [
            # Sleep duration boundary tests
            (
                {"sleep_hours": 5.0, "desc": "Exactly at min_sleep_hours threshold"},
                {"has_factor": False, "min_score": 0.0},
            ),
            (
                {"sleep_hours": 4.9, "desc": "Just below min_sleep_hours threshold"},
                {"has_factor": True, "min_score": 0.05},
            ),
            (
                {
                    "sleep_hours": 3.0,
                    "desc": "Exactly at critical_sleep_hours threshold",
                },
                {"has_factor": True, "min_score": 0.4},
            ),
            (
                {
                    "sleep_hours": 2.9,
                    "desc": "Just below critical_sleep_hours threshold",
                },
                {"has_factor": True, "min_score": 0.4},
            ),
            # Circadian rhythm boundary tests
            (
                {"circadian": 0.5, "desc": "Exactly at circadian_disruption_threshold"},
                {"has_circadian_factor": False},
            ),
            (
                {
                    "circadian": 0.49,
                    "desc": "Just below circadian_disruption_threshold",
                },
                {"has_circadian_factor": True},
            ),
            # Activity ratio boundary tests
            (
                {"activity_ratio": 1.5, "desc": "Exactly at activity_surge_ratio"},
                {"has_activity_factor": True},
            ),
            (
                {"activity_ratio": 1.49, "desc": "Just below activity_surge_ratio"},
                {"has_activity_factor": False},
            ),
            # Activity fragmentation boundary tests
            (
                {
                    "fragmentation": 0.8,
                    "desc": "Exactly at activity_fragmentation_threshold",
                },
                {"has_fragmentation_factor": False},
            ),
            (
                {
                    "fragmentation": 0.81,
                    "desc": "Just above activity_fragmentation_threshold",
                },
                {"has_fragmentation_factor": True},
            ),
        ],
    )
    def test_exact_threshold_boundaries(self, test_case, expected):
        """Test exact boundary values for all thresholds."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        analyzer = ManiaRiskAnalyzer()

        # Prepare test data based on test case
        pat_metrics = {
            "total_sleep_time": test_case.get("sleep_hours", 7.0),
            "circadian_rhythm_score": test_case.get("circadian", 0.8),
            "activity_fragmentation": test_case.get("fragmentation", 0.5),
        }

        activity_stats = None
        baseline = None

        if "activity_ratio" in test_case:
            activity_stats = {"avg_daily_steps": 10000}
            baseline = {"avg_steps": 10000 / test_case["activity_ratio"]}

        result = analyzer.analyze(
            pat_metrics=pat_metrics,
            activity_stats=activity_stats,
            historical_baseline=baseline,
        )

        # Check sleep factors
        if "has_factor" in expected:
            sleep_factors = [
                f for f in result.contributing_factors if "sleep" in f.lower()
            ]
            if expected["has_factor"]:
                assert (
                    len(sleep_factors) > 0
                ), f"{test_case['desc']}: Expected sleep factor"
                if "min_score" in expected:
                    assert (
                        result.risk_score >= expected["min_score"]
                    ), f"{test_case['desc']}: Score {result.risk_score} < {expected['min_score']}"
            else:
                assert (
                    len(sleep_factors) == 0
                ), f"{test_case['desc']}: Unexpected sleep factor"

        # Check circadian factors
        if "has_circadian_factor" in expected:
            circadian_factors = [
                f for f in result.contributing_factors if "circadian" in f.lower()
            ]
            if expected["has_circadian_factor"]:
                assert (
                    len(circadian_factors) > 0
                ), f"{test_case['desc']}: Expected circadian factor"
            else:
                assert (
                    len(circadian_factors) == 0
                ), f"{test_case['desc']}: Unexpected circadian factor"

        # Check activity factors
        if "has_activity_factor" in expected:
            activity_factors = [
                f for f in result.contributing_factors if "activity surge" in f.lower()
            ]
            if expected["has_activity_factor"]:
                assert (
                    len(activity_factors) > 0
                ), f"{test_case['desc']}: Expected activity factor"
            else:
                assert (
                    len(activity_factors) == 0
                ), f"{test_case['desc']}: Unexpected activity factor"

        # Check fragmentation factors
        if "has_fragmentation_factor" in expected:
            frag_factors = [
                f for f in result.contributing_factors if "fragmentation" in f.lower()
            ]
            if expected["has_fragmentation_factor"]:
                assert (
                    len(frag_factors) > 0
                ), f"{test_case['desc']}: Expected fragmentation factor"
            else:
                assert (
                    len(frag_factors) == 0
                ), f"{test_case['desc']}: Unexpected fragmentation factor"

    def test_actigraphy_analysis_schema_includes_mania_fields(self):
        """Test that ActigraphyAnalysis schema includes required mania risk fields."""
        from clarity.ml.pat_service import ActigraphyAnalysis

        # Check that the model has the required mania fields
        model_fields = ActigraphyAnalysis.model_fields

        # Verify mania_risk_score field exists and has correct type
        assert "mania_risk_score" in model_fields
        mania_score_field = model_fields["mania_risk_score"]
        assert mania_score_field.default == 0.0
        assert "Mania risk score" in mania_score_field.description

        # Verify mania_alert_level field exists and has correct type
        assert "mania_alert_level" in model_fields
        mania_level_field = model_fields["mania_alert_level"]
        assert mania_level_field.default == "none"
        assert "none/low/moderate/high" in mania_level_field.description

        # Test that we can instantiate with mania fields
        analysis = ActigraphyAnalysis(
            user_id="test_user",
            analysis_timestamp="2024-01-01T00:00:00Z",
            sleep_efficiency=85.0,
            sleep_onset_latency=15.0,
            wake_after_sleep_onset=30.0,
            total_sleep_time=7.5,
            circadian_rhythm_score=0.85,
            activity_fragmentation=0.3,
            depression_risk_score=0.2,
            sleep_stages=["light", "deep", "rem"],
            confidence_score=0.9,
            clinical_insights=["Good sleep quality"],
            embedding=[0.1] * 128,
            mania_risk_score=0.75,  # Test setting mania score
            mania_alert_level="high",  # Test setting mania alert
        )

        assert analysis.mania_risk_score == 0.75
        assert analysis.mania_alert_level == "high"

    def test_high_alert_rate_limiting(self):
        """Test that duplicate high alerts within 24 hours are rate limited."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer

        user_id = "test_user_123"
        analyzer = ManiaRiskAnalyzer(user_id=user_id)

        # First analysis with critically low sleep should trigger high alert
        critical_sleep = SleepFeatures(
            total_sleep_minutes=150,  # 2.5 hours
            sleep_efficiency=0.95,
            sleep_latency=2.0,
            awakenings_count=0.0,
            consistency_score=0.2,
            overall_quality_score=0.4,
            data_coverage_days=7,
        )

        first_result = analyzer.analyze(
            sleep_features=critical_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.3,
                "activity_fragmentation": 0.9,
            },
            user_id=user_id,
        )

        # First alert should be high
        assert first_result.alert_level == "high"
        assert first_result.risk_score >= 0.7

        # Second analysis with same critical conditions
        second_result = analyzer.analyze(
            sleep_features=critical_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.3,
                "activity_fragmentation": 0.9,
            },
            user_id=user_id,
        )

        # Second alert should be downgraded to moderate due to rate limiting
        assert second_result.alert_level == "moderate"
        assert second_result.risk_score >= 0.7  # Score is still high

        # Different user should still get high alert
        other_user_result = analyzer.analyze(
            sleep_features=critical_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.3,
                "activity_fragmentation": 0.9,
            },
            user_id="different_user_456",
        )

        assert other_user_result.alert_level == "high"
