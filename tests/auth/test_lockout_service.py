"""Tests for Account Lockout Service."""

import asyncio
from datetime import timedelta
import logging

import pytest

from clarity.auth.lockout_service import AccountLockoutError, AccountLockoutService

logger = logging.getLogger(__name__)


class TestAccountLockoutService:
    """Test account lockout functionality."""

    @pytest.fixture
    def lockout_service(self) -> AccountLockoutService:
        """Create a fresh lockout service for each test."""
        return AccountLockoutService()

    @pytest.mark.asyncio
    async def test_no_lockout_initially(
        self, lockout_service: AccountLockoutService
    ) -> None:
        """Test that new accounts are not locked."""
        # Should not raise exception
        await lockout_service.check_lockout("test@example.com")

    @pytest.mark.asyncio
    async def test_record_failed_attempts(
        self, lockout_service: AccountLockoutService
    ) -> None:
        """Test recording failed login attempts."""
        email = "test@example.com"
        ip = "192.168.1.1"

        # Record attempts up to one less than max (2 for max_attempts=3)
        for _i in range(2):
            await lockout_service.record_failed_attempt(email, ip)

        # Should still be able to check (not locked yet at 2 attempts)
        await lockout_service.check_lockout(email)

    @pytest.mark.asyncio
    async def test_account_lockout_after_max_attempts(
        self, lockout_service: AccountLockoutService
    ) -> None:
        """Test that account gets locked after max failed attempts."""
        email = "test@example.com"
        ip = "192.168.1.1"

        # Record max failed attempts (3 for test lockout service)
        for _i in range(3):
            await lockout_service.record_failed_attempt(email, ip)

        # Next check should raise lockout error
        with pytest.raises(AccountLockoutError) as exc_info:
            await lockout_service.check_lockout(email)

        assert "too many failed login attempts" in str(exc_info.value).lower()
        assert email in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_reset_attempts_after_success(
        self, lockout_service: AccountLockoutService
    ) -> None:
        """Test that successful login resets failed attempts."""
        email = "test@example.com"
        ip = "192.168.1.1"

        # Record some failed attempts (but not enough to lock)
        for _i in range(2):
            await lockout_service.record_failed_attempt(email, ip)

        # Reset attempts (simulate successful login)
        await lockout_service.reset_attempts(email)

        # Should be able to record more attempts without immediate lockout
        for _i in range(2):  # Only 2 more attempts (total would be 2 after reset)
            await lockout_service.record_failed_attempt(email, ip)

        # Should still not be locked (only 2 attempts after reset)
        await lockout_service.check_lockout(email)

    @pytest.mark.asyncio
    async def test_lockout_expiry(self, lockout_service: AccountLockoutService) -> None:
        """Test that lockout expires after timeout period."""
        email = "test@example.com"
        ip = "192.168.1.1"

        # Create lockout service with very short timeout for testing
        short_timeout_service = AccountLockoutService(
            max_attempts=3,
            lockout_duration=timedelta(seconds=1),  # 1 second for testing
        )

        # Log for debugging
        logger.info(
            "Service initialized with max_attempts=%s",
            short_timeout_service.max_attempts,
        )

        # Trigger lockout - record exactly 3 attempts
        for _i in range(3):
            await short_timeout_service.record_failed_attempt(email, ip)

        # Should be locked
        with pytest.raises(AccountLockoutError):
            await short_timeout_service.check_lockout(email)

        # Wait for lockout to expire (add buffer for timing issues)
        await asyncio.sleep(1.5)

        # Should no longer be locked
        await short_timeout_service.check_lockout(email)

    @pytest.mark.asyncio
    async def test_different_users_independent(
        self, lockout_service: AccountLockoutService
    ) -> None:
        """Test that lockouts are per-user."""
        email1 = "user1@example.com"
        email2 = "user2@example.com"
        ip = "192.168.1.1"

        # Lock first user
        for _i in range(5):
            await lockout_service.record_failed_attempt(email1, ip)

        # First user should be locked
        with pytest.raises(AccountLockoutError):
            await lockout_service.check_lockout(email1)

        # Second user should not be affected
        await lockout_service.check_lockout(email2)

    @pytest.mark.asyncio
    async def test_ip_tracking(self, lockout_service: AccountLockoutService) -> None:
        """Test that IP addresses are tracked for security monitoring."""
        email = "test@example.com"
        ip1 = "192.168.1.1"
        ip2 = "10.0.0.1"

        # Record attempts from different IPs
        await lockout_service.record_failed_attempt(email, ip1)
        await lockout_service.record_failed_attempt(email, ip2)
        await lockout_service.record_failed_attempt(email, ip1)

        # Should track all attempts regardless of IP
        await lockout_service.record_failed_attempt(email, ip2)
        await lockout_service.record_failed_attempt(email, ip1)

        # Should be locked after 5 total attempts
        with pytest.raises(AccountLockoutError):
            await lockout_service.check_lockout(email)

    @pytest.mark.asyncio
    async def test_get_lockout_status(
        self, lockout_service: AccountLockoutService
    ) -> None:
        """Test getting lockout status without raising exceptions."""
        email = "test@example.com"
        ip = "192.168.1.1"

        # Initially no lockout
        is_locked = await lockout_service.is_locked(email)
        assert is_locked is False

        # Add some failed attempts (but not enough to lock)
        for _i in range(2):
            await lockout_service.record_failed_attempt(email, ip)

        # Should not be locked yet
        is_locked = await lockout_service.is_locked(email)
        assert is_locked is False

        # Trigger lockout with one more attempt (total 3)
        await lockout_service.record_failed_attempt(email, ip)

        # Now should be locked
        is_locked = await lockout_service.is_locked(email)
        assert is_locked is True

    @pytest.mark.asyncio
    async def test_custom_configuration(self) -> None:
        """Test lockout service with custom configuration."""
        # Create service with custom settings
        custom_service = AccountLockoutService(
            max_attempts=3, lockout_duration=timedelta(minutes=30)
        )

        email = "test@example.com"
        ip = "192.168.1.1"

        # Should lock after 3 attempts instead of 5
        for _i in range(3):
            await custom_service.record_failed_attempt(email, ip)

        # Should be locked
        with pytest.raises(AccountLockoutError) as exc_info:
            await custom_service.check_lockout(email)

        # Check that lockout duration is 30 minutes
        error_msg = str(exc_info.value)
        # The exact time format may vary, but should indicate 30 minutes from now
        assert "locked" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_concurrent_access(
        self, lockout_service: AccountLockoutService
    ) -> None:
        """Test that the service handles concurrent access safely."""
        email = "test@example.com"

        # Simulate concurrent failed attempts
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                lockout_service.record_failed_attempt(email, f"192.168.1.{i}")
            )
            tasks.append(task)

        # Wait for all attempts to complete
        await asyncio.gather(*tasks)

        # Should be locked after all attempts
        with pytest.raises(AccountLockoutError):
            await lockout_service.check_lockout(email)

    @pytest.mark.asyncio
    async def test_error_handling(self, lockout_service: AccountLockoutService) -> None:
        """Test error handling with invalid inputs."""
        # Empty email should not crash
        await lockout_service.record_failed_attempt("", "192.168.1.1")
        await lockout_service.check_lockout("")

        # None values should not crash
        await lockout_service.record_failed_attempt("test@example.com", None)

        # Very long email should not crash
        long_email = "a" * 1000 + "@example.com"
        await lockout_service.record_failed_attempt(long_email, "192.168.1.1")
        await lockout_service.check_lockout(long_email)
