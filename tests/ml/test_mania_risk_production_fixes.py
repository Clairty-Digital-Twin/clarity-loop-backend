"""Tests for critical production fixes in mania risk module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, UTC

import pytest
import torch

from clarity.ml.analysis_pipeline import HealthAnalysisPipeline, AnalysisResults
from clarity.ml.pat_service import PATModelService
from clarity.models.health_data import HealthMetric, HealthMetricType, BiometricData


class TestManiaRiskProductionFixes:
    """Test suite for critical production fixes."""

    @pytest.mark.asyncio
    async def test_mania_risk_always_emitted_when_disabled(self):
        """Test that mania_risk is ALWAYS included in health_indicators even when disabled."""
        # Ensure mania risk is disabled
        with patch.dict(os.environ, {"MANIA_RISK_ENABLED": "false"}, clear=True):
            # Reset feature flag manager
            import clarity.core.feature_flags
            clarity.core.feature_flags._feature_flag_manager = None
            
            pipeline = HealthAnalysisPipeline()
            
            # Create minimal test data
            test_metrics = [
                HealthMetric(
                    metric_type=HealthMetricType.HEART_RATE,
                    created_at=datetime.now(UTC),
                    biometric_data=BiometricData(heart_rate=70.0),
                    device_id="test_device",
                )
            ]
            
            # Process health data
            results = await pipeline.process_health_data(
                user_id="test_user",
                health_metrics=test_metrics
            )
            
            # Verify mania_risk is present with default values
            assert "health_indicators" in results.summary_stats
            assert "mania_risk" in results.summary_stats["health_indicators"]
            
            mania_data = results.summary_stats["health_indicators"]["mania_risk"]
            assert mania_data["risk_score"] == 0.0
            assert mania_data["alert_level"] == "none"
            assert mania_data["contributing_factors"] == []
            assert mania_data["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_mania_risk_always_emitted_when_enabled(self):
        """Test that mania_risk is included when enabled."""
        with patch.dict(os.environ, {"MANIA_RISK_ENABLED": "true"}):
            # Reset feature flag manager
            import clarity.core.feature_flags
            clarity.core.feature_flags._feature_flag_manager = None
            
            pipeline = HealthAnalysisPipeline()
            
            # Mock the mania risk analyzer
            with patch('clarity.ml.analysis_pipeline.ManiaRiskAnalyzer') as mock_analyzer_class:
                mock_analyzer = MagicMock()
                mock_analyzer_class.return_value = mock_analyzer
                
                # Mock the analyze method
                from clarity.ml.mania_risk_analyzer import ManiaRiskResult
                mock_result = ManiaRiskResult(
                    risk_score=0.7,
                    alert_level="moderate",
                    contributing_factors=["reduced_sleep"],
                    confidence=0.85,
                    clinical_insight="Moderate mania risk detected",
                    recommendations=["Monitor sleep patterns"],
                    user_id="test_user"
                )
                mock_analyzer.analyze.return_value = mock_result
                
                # Create test data
                test_metrics = [
                    HealthMetric(
                        metric_type=HealthMetricType.HEART_RATE,
                        created_at=datetime.now(UTC),
                        biometric_data=BiometricData(heart_rate=70.0),
                        device_id="test_device",
                    )
                ]
                
                # Process health data
                results = await pipeline.process_health_data(
                    user_id="test_user",
                    health_metrics=test_metrics
                )
                
                # Verify mania_risk is present with actual values
                assert "health_indicators" in results.summary_stats
                assert "mania_risk" in results.summary_stats["health_indicators"]
                
                mania_data = results.summary_stats["health_indicators"]["mania_risk"]
                assert mania_data["risk_score"] == 0.7
                assert mania_data["alert_level"] == "moderate"
                assert mania_data["contributing_factors"] == ["reduced_sleep"]
                assert mania_data["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_mania_risk_graceful_failure(self):
        """Test that mania risk analysis failures are handled gracefully."""
        with patch.dict(os.environ, {"MANIA_RISK_ENABLED": "true"}):
            # Reset feature flag manager
            import clarity.core.feature_flags
            clarity.core.feature_flags._feature_flag_manager = None
            
            pipeline = HealthAnalysisPipeline()
            
            # Mock the mania risk analyzer to raise an exception
            with patch('clarity.ml.analysis_pipeline.ManiaRiskAnalyzer') as mock_analyzer_class:
                mock_analyzer = MagicMock()
                mock_analyzer_class.return_value = mock_analyzer
                mock_analyzer.analyze.side_effect = Exception("Mania analysis failed")
                
                # Create test data
                test_metrics = [
                    HealthMetric(
                        metric_type=HealthMetricType.HEART_RATE,
                        created_at=datetime.now(UTC),
                        biometric_data=BiometricData(heart_rate=70.0),
                        device_id="test_device",
                    )
                ]
                
                # Process health data - should not raise exception
                results = await pipeline.process_health_data(
                    user_id="test_user",
                    health_metrics=test_metrics
                )
                
                # Verify mania_risk is present with default values
                assert "health_indicators" in results.summary_stats
                assert "mania_risk" in results.summary_stats["health_indicators"]
                
                mania_data = results.summary_stats["health_indicators"]["mania_risk"]
                assert mania_data["risk_score"] == 0.0
                assert mania_data["alert_level"] == "none"
                assert mania_data["contributing_factors"] == []
                assert mania_data["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_pat_model_integrity_check_blocks_loading(self):
        """Test that PAT model with failed integrity check is not loaded."""
        # Create a test model file in allowed directory
        models_dir = Path.home() / ".clarity" / "models" / "pat"
        models_dir.mkdir(parents=True, exist_ok=True)
        test_model_path = models_dir / "test_pat_model.h5"
        test_model_path.touch()
        
        try:
            # Create PAT service with test model
            service = PATModelService(model_path=str(test_model_path), model_size="small")
            
            # Mock integrity check to fail
            with patch.object(service, '_verify_model_integrity', return_value=False):
                # Attempt to load model - should raise RuntimeError
                with pytest.raises(RuntimeError, match="Model integrity verification FAILED"):
                    await service.load_model()
                
                # Verify model is not loaded
                assert service.model is None
                assert not service.is_loaded
                
        finally:
            # Cleanup
            if test_model_path.exists():
                test_model_path.unlink()

    @pytest.mark.asyncio
    async def test_pat_model_integrity_check_allows_valid_model(self):
        """Test that PAT model with passed integrity check is loaded."""
        # Create a test model file in allowed directory
        models_dir = Path.home() / ".clarity" / "models" / "pat"
        models_dir.mkdir(parents=True, exist_ok=True)
        test_model_path = models_dir / "test_pat_model.h5"
        test_model_path.touch()
        
        try:
            # Create PAT service with test model
            service = PATModelService(model_path=str(test_model_path), model_size="small")
            
            # Mock integrity check to pass and weights loading to succeed
            with patch.object(service, '_verify_model_integrity', return_value=True):
                with patch.object(service, '_load_pretrained_weights', return_value=True):
                    # Load model - should succeed
                    await service.load_model()
                    
                    # Verify model is loaded
                    assert service.model is not None
                    assert service.is_loaded
                    
        finally:
            # Cleanup
            if test_model_path.exists():
                test_model_path.unlink()

    def test_feature_flag_system_environment_integration(self):
        """Test that feature flag system properly integrates with environment variables."""
        # Test disabled state
        with patch.dict(os.environ, {"MANIA_RISK_ENABLED": "false"}, clear=True):
            import clarity.core.feature_flags
            clarity.core.feature_flags._feature_flag_manager = None
            
            from clarity.core.feature_flags import is_feature_enabled
            assert not is_feature_enabled("mania_risk_analysis")
        
        # Test enabled state
        with patch.dict(os.environ, {"MANIA_RISK_ENABLED": "true"}):
            clarity.core.feature_flags._feature_flag_manager = None
            
            from clarity.core.feature_flags import is_feature_enabled
            assert is_feature_enabled("mania_risk_analysis")

    def test_feature_flag_production_environment(self):
        """Test feature flag behavior in production environment."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "MANIA_RISK_ENABLED": "true"}):
            import clarity.core.feature_flags
            clarity.core.feature_flags._feature_flag_manager = None
            
            from clarity.core.feature_flags import get_feature_flag_manager
            manager = get_feature_flag_manager()
            
            # Check production-specific flags
            assert manager.is_enabled("enhanced_security")
            assert manager.is_enabled("graceful_degradation")
            
            # Check mania risk with rollout percentage
            mania_flag = manager.get_flag("mania_risk_analysis")
            assert mania_flag is not None
            assert mania_flag.rollout_percentage == 100.0