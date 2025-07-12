"""Tests for enhanced feature flag system with auto-refresh capabilities."""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from circuitbreaker import CircuitBreakerError

from clarity.core.config_provider import ConfigProvider
from clarity.core.feature_flags import FeatureFlag, FeatureFlagConfig, FeatureFlagStatus
from clarity.core.feature_flags_enhanced import (
    EnhancedFeatureFlagConfig,
    EnhancedFeatureFlagManager,
    RefreshMode,
)


class TestEnhancedFeatureFlagManager:
    """Test cases for EnhancedFeatureFlagManager."""
    
    @pytest.fixture
    def mock_config_provider(self):
        """Create mock config provider."""
        provider = Mock(spec=ConfigProvider)
        return provider
    
    @pytest.fixture
    def enhanced_config(self):
        """Create test enhanced configuration."""
        return EnhancedFeatureFlagConfig(
            refresh_interval_seconds=10,
            refresh_mode=RefreshMode.NONE,  # Disable background tasks for tests
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_timeout=10,
            stale_config_threshold_seconds=60,
        )
    
    @pytest.fixture
    def manager(self, mock_config_provider, enhanced_config):
        """Create enhanced feature flag manager."""
        return EnhancedFeatureFlagManager(
            config_provider=mock_config_provider,
            enhanced_config=enhanced_config,
        )
    
    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager._config_provider is not None
        assert manager._enhanced_config is not None
        assert manager._last_refresh_time is None
        assert manager._refresh_failures == 0
        assert manager._circuit_breaker is not None
    
    def test_synchronous_refresh_success(self, manager):
        """Test successful synchronous refresh."""
        # Mock successful config fetch
        test_config = FeatureFlagConfig(
            flags={
                "test_flag": FeatureFlag(
                    name="test_flag",
                    status=FeatureFlagStatus.ENABLED,
                    description="Test flag",
                )
            }
        )
        
        with patch.object(
            manager, "_fetch_config_from_store", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = test_config
            
            # Perform refresh
            result = manager.refresh()
            
            assert result is True
            assert manager._last_refresh_time is not None
            assert manager._refresh_failures == 0
            assert "test_flag" in manager.config.flags
    
    def test_synchronous_refresh_failure(self, manager):
        """Test failed synchronous refresh."""
        with patch.object(
            manager, "_fetch_config_from_store", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.side_effect = Exception("Config store error")
            
            # Perform refresh
            result = manager.refresh()
            
            assert result is False
            assert manager._refresh_failures == 1
    
    @pytest.mark.asyncio
    async def test_async_refresh_success(self, manager):
        """Test successful asynchronous refresh."""
        test_config = FeatureFlagConfig(
            flags={
                "async_flag": FeatureFlag(
                    name="async_flag",
                    status=FeatureFlagStatus.ENABLED,
                )
            }
        )
        
        with patch.object(
            manager, "_fetch_config_from_store", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = test_config
            
            # Perform async refresh
            result = await manager.refresh_async()
            
            assert result is True
            assert manager._last_refresh_time is not None
            assert "async_flag" in manager.config.flags
    
    @pytest.mark.asyncio
    async def test_async_refresh_with_circuit_breaker_open(self, manager):
        """Test refresh when circuit breaker is open."""
        # Force circuit breaker to open
        manager._circuit_breaker._failure_count = 3
        manager._circuit_breaker._state = "open"
        
        result = await manager.refresh_async()
        
        assert result is False
    
    def test_circuit_breaker_trips_after_failures(self, manager):
        """Test circuit breaker trips after consecutive failures."""
        with patch.object(
            manager, "_fetch_config_from_store", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.side_effect = Exception("Config store error")
            
            # First failure
            manager.refresh()
            assert manager._circuit_breaker.current_state == "closed"
            
            # Second failure (threshold reached)
            manager.refresh()
            assert manager._circuit_breaker.current_state == "open"
            
            # Third attempt should fail fast
            result = manager.refresh()
            assert result is False
    
    def test_config_staleness_detection(self, manager):
        """Test configuration staleness detection."""
        # Initially, config should be stale (never refreshed)
        assert manager.is_config_stale() is True
        assert manager.get_config_age_seconds() is None
        
        # Set last refresh time
        manager._last_refresh_time = datetime.utcnow() - timedelta(seconds=30)
        
        # Should not be stale (30s < 60s threshold)
        assert manager.is_config_stale() is False
        age = manager.get_config_age_seconds()
        assert age is not None
        assert 29 <= age <= 31  # Allow for small timing variations
        
        # Make it stale
        manager._last_refresh_time = datetime.utcnow() - timedelta(seconds=120)
        assert manager.is_config_stale() is True
    
    def test_thread_safety_concurrent_refresh(self, manager):
        """Test thread safety during concurrent refresh operations."""
        refresh_count = 0
        
        async def mock_fetch():
            nonlocal refresh_count
            refresh_count += 1
            await asyncio.sleep(0.1)  # Simulate slow fetch
            return FeatureFlagConfig(flags={})
        
        with patch.object(manager, "_fetch_config_from_store", side_effect=mock_fetch):
            # Run multiple refreshes concurrently
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(manager.refresh) for _ in range(5)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            # All refreshes should succeed
            assert all(results)
            # But fetch should be called 5 times due to no deduplication
            assert refresh_count == 5
    
    def test_cache_clearing_on_refresh(self, manager):
        """Test cache is cleared after refresh."""
        # Add some cached values
        manager._cache["test_flag:user1"] = True
        manager._cache["test_flag:user2"] = False
        
        assert len(manager._cache) == 2
        
        # Refresh should clear cache
        with patch.object(
            manager, "_fetch_config_from_store", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = FeatureFlagConfig(flags={})
            manager.refresh()
        
        assert len(manager._cache) == 0
    
    @pytest.mark.asyncio
    async def test_periodic_refresh_loop(self):
        """Test periodic refresh background task."""
        enhanced_config = EnhancedFeatureFlagConfig(
            refresh_interval_seconds=1,  # Fast refresh for testing
            refresh_mode=RefreshMode.PERIODIC,
        )
        
        refresh_count = 0
        
        async def mock_refresh():
            nonlocal refresh_count
            refresh_count += 1
            return True
        
        manager = EnhancedFeatureFlagManager(
            config_provider=None,
            enhanced_config=enhanced_config,
        )
        
        with patch.object(manager, "refresh_async", side_effect=mock_refresh):
            # Start periodic refresh
            task = asyncio.create_task(manager._periodic_refresh_loop())
            
            # Wait for a few refreshes
            await asyncio.sleep(2.5)
            
            # Cancel task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Should have refreshed at least twice
            assert refresh_count >= 2
    
    def test_shutdown(self, manager):
        """Test manager shutdown."""
        # Set up some state
        manager._shutdown_event.clear()
        manager._refresh_task = Mock()
        manager._pubsub_task = Mock()
        
        # Shutdown
        manager.shutdown()
        
        # Verify shutdown event is set
        assert manager._shutdown_event.is_set()
        
        # Verify tasks are cancelled
        manager._refresh_task.cancel.assert_called_once()
        manager._pubsub_task.cancel.assert_called_once()
    
    def test_context_manager(self, mock_config_provider, enhanced_config):
        """Test context manager functionality."""
        with EnhancedFeatureFlagManager(
            config_provider=mock_config_provider,
            enhanced_config=enhanced_config,
        ) as manager:
            assert manager is not None
            assert not manager._shutdown_event.is_set()
        
        # After exiting context, should be shut down
        assert manager._shutdown_event.is_set()
    
    @pytest.mark.asyncio
    async def test_metrics_update_on_refresh(self, manager):
        """Test metrics are updated correctly on refresh."""
        from clarity.core.feature_flags_enhanced import (
            FEATURE_FLAG_REFRESH_SUCCESS,
            FEATURE_FLAG_REFRESH_FAILURE,
            FEATURE_FLAG_STALE_CONFIG,
        )
        
        # Mock metrics
        with patch.object(FEATURE_FLAG_REFRESH_SUCCESS, "inc") as mock_success, \
             patch.object(FEATURE_FLAG_REFRESH_FAILURE, "labels") as mock_failure, \
             patch.object(FEATURE_FLAG_STALE_CONFIG, "set") as mock_stale:
            
            # Successful refresh
            with patch.object(
                manager, "_fetch_config_from_store", new_callable=AsyncMock
            ) as mock_fetch:
                mock_fetch.return_value = FeatureFlagConfig(flags={})
                await manager.refresh_async()
            
            mock_success.assert_called_once()
            mock_stale.assert_called_with(0)
            
            # Failed refresh
            with patch.object(
                manager, "_fetch_config_from_store", new_callable=AsyncMock
            ) as mock_fetch:
                mock_fetch.side_effect = Exception("Error")
                await manager.refresh_async()
            
            mock_failure.assert_called()
    
    def test_get_circuit_breaker_state(self, manager):
        """Test getting circuit breaker state."""
        assert manager.get_circuit_breaker_state() == "closed"
        
        # Force open state
        manager._circuit_breaker._state = "open"
        assert manager.get_circuit_breaker_state() == "open"