"""Unit tests for EnhancedManiaRiskAnalyzer module."""

import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, patch
import numpy as np

from clarity.ml.enhanced_mania_risk_analyzer import (
    EnhancedManiaRiskAnalyzer,
    EnhancedRiskConfig
)
from clarity.ml.mania_risk_analyzer import ManiaRiskResult
from clarity.ml.processors.sleep_processor import SleepFeatures
from clarity.models.health_data import HealthMetric, SleepData, ActivityData, BiometricData


class TestEnhancedManiaRiskAnalyzer:
    """Test suite for enhanced mania risk analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EnhancedManiaRiskAnalyzer(user_id="test_user")
    
    def create_health_metrics_with_phase_advance(self):
        """Create metrics showing circadian phase advance (mania predictor)."""
        metrics = []
        
        # Create 14 days of data with gradual phase advance
        for i in range(14):
            date = datetime.now(UTC) - timedelta(days=i)
            
            # Gradually shift sleep earlier
            if i < 7:  # Recent week - advanced phase
                sleep_hour = 21 - (i * 0.2)  # Getting earlier
                wake_hour = 5 - (i * 0.2)
            else:  # Previous week - normal phase
                sleep_hour = 23
                wake_hour = 7
            
            sleep_start = date.replace(hour=int(sleep_hour), minute=0)
            sleep_end = date.replace(hour=int(wake_hour), minute=0) + timedelta(days=1)
            
            sleep_data = SleepData(
                sleep_start=sleep_start,
                sleep_end=sleep_end,
                total_sleep_minutes=int((sleep_end - sleep_start).total_seconds() / 60)
            )
            
            # Also add reduced sleep duration for recent days
            if i < 3:
                sleep_data.total_sleep_minutes = 240  # 4 hours
            
            activity_data = ActivityData(
                steps=12000 if i < 7 else 10000  # Increased activity
            )
            
            bio_data = BiometricData(
                heart_rate=75 if i < 7 else 65  # Elevated HR
            )
            
            metric = HealthMetric(
                user_id="test_user",
                sleep_data=sleep_data,
                activity_data=activity_data,
                biometric_data=bio_data,
                created_at=date
            )
            metrics.append(metric)
        
        return metrics
    
    def create_health_metrics_with_variability_spike(self):
        """Create metrics showing variability spike."""
        metrics = []
        
        for i in range(14):
            date = datetime.now(UTC) - timedelta(days=i)
            
            # High variability in recent week
            if i < 7:
                # Alternating extremes
                if i % 2 == 0:
                    steps = 20000
                    sleep_hours = 4
                else:
                    steps = 5000
                    sleep_hours = 10
            else:
                # Stable baseline
                steps = 10000
                sleep_hours = 7.5
            
            sleep_data = SleepData(
                total_sleep_minutes=int(sleep_hours * 60),
                sleep_start=date.replace(hour=23),
                sleep_end=date.replace(hour=int(23 + sleep_hours) % 24) + timedelta(days=1 if sleep_hours > 1 else 0)
            )
            
            activity_data = ActivityData(steps=steps)
            
            metric = HealthMetric(
                user_id="test_user",
                sleep_data=sleep_data,
                activity_data=activity_data,
                created_at=date
            )
            metrics.append(metric)
        
        return metrics
    
    def test_enhanced_config_weights(self):
        """Test that enhanced config has updated weights."""
        config = EnhancedRiskConfig()
        
        # Check primary predictors have highest weights
        assert config.weights["circadian_phase_advance"] == 0.40
        assert config.weights["circadian_phase_delay"] == 0.35
        assert config.weights["activity_variability_spike"] == 0.35
        
        # Check other weights are present
        assert "severe_sleep_loss" in config.weights
        assert "sleep_deviation" in config.weights
        assert "activity_deviation" in config.weights
    
    def test_analyze_with_phase_advance(self):
        """Test analysis detecting circadian phase advance."""
        metrics = self.create_health_metrics_with_phase_advance()
        
        # Create sleep features from recent data
        recent_sleep = [m.sleep_data for m in metrics[:7] if m.sleep_data]
        sleep_features = SleepFeatures(
            total_sleep_minutes=sum(s.total_sleep_minutes for s in recent_sleep) / len(recent_sleep),
            sleep_efficiency=0.85,
            consistency_score=0.6
        )
        
        result = self.analyzer.analyze(
            sleep_features=sleep_features,
            recent_health_metrics=metrics[:7],
            baseline_health_metrics=metrics[7:14]
        )
        
        assert result.risk_score > 0.7  # High risk due to phase advance
        assert result.alert_level in ["high", "moderate"]
        assert any("phase advance" in f.lower() for f in result.contributing_factors)
        assert result.confidence > 0.7
    
    def test_analyze_with_variability_spike(self):
        """Test analysis detecting variability spike."""
        metrics = self.create_health_metrics_with_variability_spike()
        
        result = self.analyzer.analyze(
            recent_health_metrics=metrics[:7],
            baseline_health_metrics=metrics[7:14]
        )
        
        assert result.risk_score > 0.5
        assert any("variability" in f.lower() for f in result.contributing_factors)
    
    def test_analyze_without_raw_metrics(self):
        """Test fallback to base analysis without raw metrics."""
        # Only provide processed features
        sleep_features = SleepFeatures(
            total_sleep_minutes=240,  # 4 hours
            sleep_efficiency=0.75,
            consistency_score=0.3
        )
        
        result = self.analyzer.analyze(sleep_features=sleep_features)
        
        # Should still detect sleep issues
        assert result.risk_score > 0.4
        assert any("sleep" in f.lower() for f in result.contributing_factors)
    
    def test_time_to_episode_prediction(self):
        """Test prediction of days until episode."""
        metrics = self.create_health_metrics_with_phase_advance()
        
        result = self.analyzer.analyze(
            recent_health_metrics=metrics[:7],
            baseline_health_metrics=metrics[7:14]
        )
        
        # Should predict imminent risk due to phase advance
        assert "Risk window:" in result.clinical_insight or "Monitor closely" in result.clinical_insight
    
    def test_personal_baseline_integration(self):
        """Test integration with personal baseline tracker."""
        metrics = self.create_health_metrics_with_phase_advance()
        
        # First analysis to establish baseline
        result1 = self.analyzer.analyze(
            recent_health_metrics=metrics[7:14],
            baseline_health_metrics=metrics[7:14],
            user_id="test_user"
        )
        
        # Second analysis with deviations
        sleep_features = SleepFeatures(
            total_sleep_minutes=240  # Much less than baseline
        )
        
        result2 = self.analyzer.analyze(
            sleep_features=sleep_features,
            recent_health_metrics=metrics[:7],
            user_id="test_user"
        )
        
        # Should detect personal deviations
        assert result2.risk_score > result1.risk_score
        assert any("below personal baseline" in f for f in result2.contributing_factors)
    
    def test_enhanced_recommendations(self):
        """Test generation of enhanced recommendations."""
        metrics = self.create_health_metrics_with_phase_advance()
        
        result = self.analyzer.analyze(
            recent_health_metrics=metrics[:7],
            baseline_health_metrics=metrics[7:14]
        )
        
        # Should have phase-specific recommendations
        assert any("bedtime" in r.lower() for r in result.recommendations)
        assert any("light" in r.lower() for r in result.recommendations)
        
        # High risk should have urgent recommendations
        if result.alert_level == "high":
            assert any("24 hours" in r for r in result.recommendations)
            assert any("crisis plan" in r.lower() for r in result.recommendations)
    
    def test_mixed_signals_handling(self):
        """Test handling of mixed mania/depression signals."""
        metrics = []
        
        for i in range(14):
            date = datetime.now(UTC) - timedelta(days=i)
            
            # Phase advance (mania) but high sleep (depression)
            sleep_data = SleepData(
                sleep_start=date.replace(hour=20),  # Early (advance)
                sleep_end=date.replace(hour=8) + timedelta(days=1),  # Long duration
                total_sleep_minutes=720  # 12 hours (depression-like)
            )
            
            # Low activity (depression)
            activity_data = ActivityData(steps=3000)
            
            metric = HealthMetric(
                user_id="test_user",
                sleep_data=sleep_data,
                activity_data=activity_data,
                created_at=date
            )
            metrics.append(metric)
        
        result = self.analyzer.analyze(
            recent_health_metrics=metrics[:7],
            baseline_health_metrics=metrics[7:14]
        )
        
        # Should handle conflicting signals appropriately
        assert result.confidence < 0.8  # Lower confidence due to mixed signals
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Only 2 days of data
        metrics = self.create_health_metrics_with_phase_advance()[:2]
        
        result = self.analyzer.analyze(recent_health_metrics=metrics)
        
        # Should have lower confidence
        assert result.confidence < 0.7
        # Should still provide some analysis if possible
        assert result.risk_score >= 0.0
    
    def test_enhanced_clinical_insights(self):
        """Test enhanced clinical insight generation."""
        metrics = self.create_health_metrics_with_phase_advance()
        
        result = self.analyzer.analyze(
            recent_health_metrics=metrics[:7],
            baseline_health_metrics=metrics[7:14]
        )
        
        # Should include timing prediction if high risk
        if result.alert_level == "high":
            assert "days" in result.clinical_insight or "Risk window" in result.clinical_insight
            assert "Primary indicators" in result.clinical_insight
        
        # Should be more detailed than base analyzer
        assert len(result.clinical_insight) > 50