"""Tests for Mania Risk Analyzer - Test-Driven Development.

Following TDD principles, we write tests first to define expected behavior
for our mania risk detection module based on 2024-2025 research.
"""

import pytest
from datetime import datetime, UTC

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
            quality_score=0.85,
        )
        
        result = analyzer.analyze(
            sleep_features=healthy_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.85,
                "activity_fragmentation": 0.3,
            }
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
            sleep_efficiency=0.95,    # Paradoxically high
            sleep_latency=2.0,        # Falls asleep instantly
            awakenings_count=0.0,
            consistency_score=0.2,    # Very irregular schedule
            quality_score=0.4,
        )
        
        result = analyzer.analyze(
            sleep_features=manic_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.3,  # Disrupted
                "activity_fragmentation": 0.9,   # High fragmentation
            }
        )
        
        assert result.risk_score >= 0.7
        assert result.alert_level == "high"
        assert any("critically low sleep" in f.lower() for f in result.contributing_factors)
        assert any("circadian" in f.lower() for f in result.contributing_factors)
        assert "healthcare provider" in result.clinical_insight.lower()
        assert len(result.recommendations) >= 3
        assert any("Contact your healthcare provider" in r for r in result.recommendations)
    
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
            quality_score=0.65,
        )
        
        result = analyzer.analyze(
            sleep_features=borderline_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.6,
                "activity_fragmentation": 0.6,
            }
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
            }
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
        
        assert result.risk_score >= 0.25  # Significant contribution from circadian
        assert any("circadian rhythm" in f.lower() for f in result.contributing_factors)
        assert "0.30" in str(result.contributing_factors) or "0.3" in str(result.contributing_factors)
    
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
                "avg_hrv": 15,     # Low
            }
        )
        
        # Should have base risk from sleep plus physio markers
        assert any("elevated resting hr" in f.lower() for f in result.contributing_factors)
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
                quality_score=0.75,
            ),
            historical_baseline={
                "avg_sleep_hours": 8.0,  # Normal is 8 hours
            }
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
                quality_score=0.6,
            )
        )
        
        assert any("rapid sleep onset" in f.lower() for f in result.contributing_factors)
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
        assert result.risk_score >= 0.7  # Still high risk
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
                quality_score=0.3,
            ),
            pat_metrics={
                "circadian_rhythm_score": 0.3,
                "activity_fragmentation": 0.85,
            }
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
    
    @pytest.mark.parametrize("sleep_hours,expected_level", [
        (8.0, "none"),      # Normal sleep
        (7.0, "none"),      # Good sleep
        (6.0, "none"),      # Slightly reduced but ok
        (5.5, "none"),      # Just above threshold
        (4.8, "moderate"),  # Below 5h threshold
        (4.0, "moderate"),  # Clearly reduced
        (3.0, "moderate"),  # At critical threshold (just sleep alone)
        (2.0, "moderate"),  # Severe insomnia (just sleep alone)
        (0.5, "moderate"),  # Near-zero sleep (just sleep alone)
    ])
    def test_sleep_duration_thresholds(self, sleep_hours, expected_level):
        """Test various sleep duration thresholds map to correct alert levels."""
        from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer
        
        analyzer = ManiaRiskAnalyzer()
        
        result = analyzer.analyze(
            pat_metrics={
                "total_sleep_time": sleep_hours,
                "circadian_rhythm_score": 0.7,  # Normal circadian
                "activity_fragmentation": 0.5,   # Normal activity
            }
        )
        
        assert result.alert_level == expected_level, \
            f"Expected {expected_level} for {sleep_hours}h sleep, got {result.alert_level}"
    
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