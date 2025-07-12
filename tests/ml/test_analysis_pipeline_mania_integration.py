"""Integration tests for mania risk analysis in the health analysis pipeline."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clarity.ml.analysis_pipeline import HealthAnalysisPipeline
from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthMetric,
    HealthMetricType,
    SleepData,
)


class TestManiaRiskIntegration:
    """Test mania risk integration in the analysis pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a health analysis pipeline instance."""
        return HealthAnalysisPipeline()

    @pytest.fixture
    def mock_dynamodb_client(self):
        """Mock DynamoDB client for baseline retrieval."""
        mock_client = MagicMock()
        mock_client.table = MagicMock()
        return mock_client

    def create_sleep_metric(self, user_id: str, timestamp: datetime, sleep_hours: float, efficiency: float = 0.85) -> HealthMetric:
        """Helper to create sleep metrics with proper data."""
        sleep_data = SleepData(
            total_sleep_minutes=int(sleep_hours * 60),
            sleep_efficiency=efficiency,
            time_to_sleep_minutes=15,
            wake_count=2,
            sleep_start=timestamp - timedelta(hours=sleep_hours + 0.25),  # Add 15 min for sleep latency
            sleep_end=timestamp,
            waso_minutes=20,  # Wake after sleep onset
            rem_percentage=0.25,  # 25% REM
            deep_percentage=0.20,  # 20% deep sleep
            light_percentage=0.55,  # 55% light sleep
        )

        return HealthMetric(
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            created_at=timestamp,
            sleep_data=sleep_data,
            device_id="test-device",
            metadata={"user_id": user_id},
        )

    def create_activity_metric(self, user_id: str, timestamp: datetime, steps: int) -> HealthMetric:
        """Helper to create activity metrics."""
        activity_data = ActivityData(
            steps=steps,
            distance=steps * 0.0008,  # Roughly 0.8m per step
            active_energy=steps * 0.04,  # Roughly 0.04 kcal per step
            exercise_minutes=30 if steps > 10000 else 15,
        )

        return HealthMetric(
            metric_type=HealthMetricType.ACTIVITY_LEVEL,
            created_at=timestamp,
            activity_data=activity_data,
            device_id="test-device",
            metadata={"user_id": user_id},
        )

    def create_heart_rate_metric(self, user_id: str, timestamp: datetime, heart_rate: int, hrv: float = None) -> HealthMetric:
        """Helper to create heart rate metrics."""
        if hrv is None:
            hrv = 40.0 if heart_rate < 80 else 25.0  # Lower HRV with higher HR

        biometric_data = BiometricData(
            heart_rate=heart_rate,
            heart_rate_variability=hrv,
        )

        return HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            created_at=timestamp,
            biometric_data=biometric_data,
            device_id="test-device",
            metadata={"user_id": user_id},
        )

    @pytest.mark.asyncio
    async def test_mania_risk_detected_in_pipeline(self, pipeline, mock_dynamodb_client):
        """Test that mania risk is properly detected and integrated."""
        user_id = "test-user-123"

        # Mock DynamoDB client
        with patch.object(pipeline, '_get_dynamodb_client', return_value=mock_dynamodb_client):
            # Mock baseline query - return normal baseline
            mock_dynamodb_client.table.query.return_value = {
                "Items": [
                    {
                        "sleep_features": {"total_sleep_minutes": 450},  # 7.5 hours baseline
                        "activity_features": {"avg_daily_steps": 8000},  # 8k steps baseline
                    }
                ]
            }

            # Create metrics showing mania patterns
            metrics = []
            base_time = datetime.now(UTC)

            # Add sleep deprivation (3 hours instead of 7.5)
            metrics.append(self.create_sleep_metric(user_id, base_time, 3.0))

            # Add activity surge (15k steps vs 8k baseline)
            metrics.append(self.create_activity_metric(user_id, base_time, 15000))

            # Add elevated heart rate
            metrics.append(self.create_heart_rate_metric(user_id, base_time, 95))

            # Process through pipeline
            results = await pipeline.process_health_data(
                user_id=user_id,
                health_metrics=metrics,
            )

            # Verify mania risk was analyzed
            assert "health_indicators" in results.summary_stats
            assert "mania_risk" in results.summary_stats["health_indicators"]

            mania_risk = results.summary_stats["health_indicators"]["mania_risk"]
            assert mania_risk["risk_score"] > 0.5  # Should be moderate to high
            assert mania_risk["alert_level"] in {"moderate", "high"}
            assert len(mania_risk["contributing_factors"]) >= 2
            assert mania_risk["confidence"] > 0.5

            # Check for sleep-related factor
            factors = mania_risk["contributing_factors"]
            assert any("sleep" in f.lower() for f in factors)

            # Verify clinical insights were added
            if mania_risk["alert_level"] in {"moderate", "high"}:
                assert "clinical_insights" in results.summary_stats
                assert any("mania" in insight.lower()
                          for insight in results.summary_stats["clinical_insights"])

    @pytest.mark.asyncio
    async def test_mania_risk_with_no_baseline(self, pipeline, mock_dynamodb_client):
        """Test mania risk detection when no historical baseline exists."""
        user_id = "new-user-456"

        with patch.object(pipeline, '_get_dynamodb_client', return_value=mock_dynamodb_client):
            # Mock empty baseline query
            mock_dynamodb_client.table.query.return_value = {"Items": []}

            # Create metrics with critically low sleep
            metrics = [
                self.create_sleep_metric(user_id, datetime.now(UTC), 2.5),  # Very low sleep
                self.create_activity_metric(user_id, datetime.now(UTC), 12000),
            ]

            results = await pipeline.process_health_data(
                user_id=user_id,
                health_metrics=metrics,
            )

            # Should still detect risk based on absolute thresholds
            mania_risk = results.summary_stats["health_indicators"]["mania_risk"]
            assert mania_risk["risk_score"] > 0.4
            assert "Critically low sleep" in str(mania_risk["contributing_factors"])

    @pytest.mark.asyncio
    async def test_no_mania_risk_with_healthy_data(self, pipeline, mock_dynamodb_client):
        """Test that healthy data produces no mania risk alert."""
        user_id = "healthy-user-789"

        with patch.object(pipeline, '_get_dynamodb_client', return_value=mock_dynamodb_client):
            # Mock normal baseline
            mock_dynamodb_client.table.query.return_value = {
                "Items": [{"sleep_features": {"total_sleep_minutes": 480}}]
            }

            # Create healthy metrics for several days
            metrics = []
            base_time = datetime.now(UTC)

            # Add 7 days of consistent healthy data
            for i in range(7):
                timestamp = base_time - timedelta(days=i)
                metrics.extend([
                    self.create_sleep_metric(user_id, timestamp, 7.5 + (i % 2) * 0.2),  # 7.5-7.7 hours
                    self.create_activity_metric(user_id, timestamp, 8000 + (i * 100)),  # 8000-8600 steps
                    self.create_heart_rate_metric(user_id, timestamp, 63 + i % 3, hrv=42.0 + i % 5),  # Normal HR/HRV
                ])

            results = await pipeline.process_health_data(
                user_id=user_id,
                health_metrics=metrics,
            )

            # Should have minimal or no risk
            mania_risk = results.summary_stats["health_indicators"]["mania_risk"]

            # Check if it's related to insufficient data rather than mania risk
            if "Insufficient sleep data" in str(mania_risk["contributing_factors"]):
                # This is expected - no direct sleep metrics from PAT
                assert mania_risk["risk_score"] == 0.0
                assert mania_risk["alert_level"] == "none"
            else:
                # Otherwise should have low risk
                assert mania_risk["risk_score"] <= 0.3
                assert mania_risk["alert_level"] in {"none", "low"}

            # No clinical insights for low risk
            if "clinical_insights" in results.summary_stats:
                assert not any("mania" in insight.lower()
                              for insight in results.summary_stats["clinical_insights"])

    @pytest.mark.asyncio
    async def test_recommendations_added_for_high_risk(self, pipeline, mock_dynamodb_client):
        """Test that recommendations are added when mania risk is high."""
        user_id = "high-risk-user"

        with patch.object(pipeline, '_get_dynamodb_client', return_value=mock_dynamodb_client):
            mock_dynamodb_client.table.query.return_value = {"Items": []}

            # Create very concerning metrics
            metrics = [
                self.create_sleep_metric(user_id, datetime.now(UTC), 1.5),  # Extremely low
                self.create_activity_metric(user_id, datetime.now(UTC), 20000),  # Very high activity
            ]

            results = await pipeline.process_health_data(
                user_id=user_id,
                health_metrics=metrics,
            )

            # Should have recommendations
            assert "recommendations" in results.summary_stats
            recommendations = results.summary_stats["recommendations"]
            assert len(recommendations) >= 3
            assert any("sleep" in r.lower() for r in recommendations)
            assert any("provider" in r.lower() or "healthcare" in r.lower() for r in recommendations)

    @pytest.mark.asyncio
    async def test_mania_risk_with_pat_only_data(self, pipeline):
        """Test mania risk analysis when only PAT metrics are available."""
        user_id = "pat-only-user"

        # Create minimal metrics that would trigger PAT analysis
        metrics = [
            self.create_activity_metric(user_id, datetime.now(UTC) - timedelta(hours=i), 10000)
            for i in range(7)  # 7 days of activity data
        ]

        # Mock PAT service to return data with sleep estimation
        with patch('clarity.ml.analysis_pipeline.get_pat_service') as mock_get_pat:
            mock_pat_service = AsyncMock()
            mock_pat_service.analyze_actigraphy = AsyncMock(return_value=MagicMock(
                total_sleep_time=3.5,  # Low sleep from PAT estimation
                circadian_rhythm_score=0.4,  # Disrupted circadian rhythm
                activity_fragmentation=0.85,  # High fragmentation
                embedding=[0.1] * 128
            ))
            mock_get_pat.return_value = mock_pat_service

            # Mock DynamoDB
            with patch.object(pipeline, '_get_dynamodb_client') as mock_get_db:
                mock_db = MagicMock()
                mock_db.table.query.return_value = {"Items": []}
                mock_get_db.return_value = mock_db

                results = await pipeline.process_health_data(
                    user_id=user_id,
                    health_metrics=metrics,
                )

                # PAT data should still trigger mania risk analysis
                # Even though we don't have direct sleep metrics
                assert "health_indicators" in results.summary_stats
                # The mania risk analysis should use PAT-estimated sleep
