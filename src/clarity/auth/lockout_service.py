"""Account Lockout Protection Service.

Provides brute force protection by tracking failed login attempts
and temporarily locking accounts after too many failures.
"""

import asyncio
from datetime import UTC, datetime, timedelta
import logging
import os
import time
from typing import Any

import redis.asyncio as redis

from clarity.core.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class AccountLockoutError(AuthenticationError):
    """Raised when an account is temporarily locked due to too many failed attempts."""

    def __init__(self, username: str, unlock_time: datetime) -> None:
        self.username = username
        self.unlock_time = unlock_time
        super().__init__(
            f"Account {username} is locked until {unlock_time.isoformat()}. "
            f"Too many failed login attempts."
        )


class AccountLockoutService:
    """Account lockout service with Redis persistence and in-memory fallback."""

    _PREFIX = "lockout:v1:"  # key namespace

    def __init__(
        self,
        max_attempts: int = 3,
        lockout_duration: timedelta = timedelta(minutes=15),
        redis_url: str | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.lockout_secs = int(lockout_duration.total_seconds())
        self._mem: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._r = redis.from_url(redis_url) if redis_url else None

        logger.info(
            "🔒 AccountLockoutService initialized: max_attempts=%d, lockout_duration=%s, persistence=%s",
            self.max_attempts,
            lockout_duration,
            "Redis" if self._r else "in-memory",
        )

    # ---------- public API ----------
    async def is_locked(self, user: str) -> bool:
        if self._r:
            return await self._redis_is_locked(user)
        return await self._mem_is_locked(user)

    async def register_failure(self, user: str) -> None:
        if self._r:
            await self._redis_register_failure(user)
        else:
            await self._mem_register_failure(user)

    async def reset(self, user: str) -> None:
        if self._r is not None:
            await self._r.delete(self._key(user))
        else:
            async with self._lock:
                self._mem.pop(user, None)

    # ---------- Redis impl ----------
    async def _redis_is_locked(self, user: str) -> bool:
        if self._r is None:
            msg = "Redis client not initialized"
            raise RuntimeError(msg)
        key = self._key(user)
        ttl = await self._r.ttl(key)
        if ttl <= 0:
            return False
        locked_value = await self._r.hget(key, "locked")
        return locked_value == b"1"

    async def _redis_register_failure(self, user: str) -> None:
        if self._r is None:
            msg = "Redis client not initialized"
            raise RuntimeError(msg)
        key = self._key(user)
        pipe = self._r.pipeline()
        pipe.hincrby(key, "attempts", 1)
        pipe.hget(key, "attempts")
        res = await pipe.execute()
        attempt_count = int(res[1])
        if attempt_count == 1:
            await self._r.expire(key, self.lockout_secs)
        if attempt_count >= self.max_attempts:
            # mark as locked; keep same TTL
            await self._r.hset(key, mapping={"locked": 1})

    # ---------- in-memory impl (unchanged, but guarded) ----------
    async def _mem_is_locked(self, user: str) -> bool:
        async with self._lock:
            rec = self._mem.get(user)
            if not rec:
                return False

            current_time = time.time()
            locked_until = rec.get("locked_until", 0)

            # If lockout has expired, clean up
            if locked_until > 0 and locked_until <= current_time:
                rec["attempts"] = []
                rec["locked_until"] = 0
                return False

            return bool(locked_until > current_time)

    async def _mem_register_failure(self, user: str) -> None:
        async with self._lock:
            rec = self._mem.setdefault(user, {"attempts": [], "locked_until": 0})

            # Clean up expired attempts
            current_time = time.time()
            if rec["locked_until"] > 0 and rec["locked_until"] <= current_time:
                # Lockout has expired, reset attempts
                rec["attempts"] = []
                rec["locked_until"] = 0

            rec["attempts"].append(current_time)
            if len(rec["attempts"]) >= self.max_attempts:
                rec["locked_until"] = current_time + self.lockout_secs

    # ---------- helpers ----------
    @staticmethod
    def _key(user: str) -> str:
        return f"{AccountLockoutService._PREFIX}{user.lower()}"

    # ---------- compatibility methods for existing API ----------
    async def is_account_locked(self, username: str) -> bool:
        """Alias for is_locked for backward compatibility."""
        return await self.is_locked(username)

    async def record_failed_attempt(
        self, username: str, ip_address: str | None = None
    ) -> None:
        """Alias for register_failure for backward compatibility."""
        await self.register_failure(username)
        if await self.is_locked(username):
            logger.warning(
                "🔒 Account locked: username=%s, max_attempts=%d, ip=%s",
                username,
                self.max_attempts,
                ip_address,
            )
        else:
            logger.info(
                "🚨 Failed attempt recorded: username=%s, ip=%s", username, ip_address
            )

    async def reset_attempts(self, username: str) -> None:
        """Alias for reset for backward compatibility."""
        await self.reset(username)
        logger.info("✅ Reset failed attempts for user: %s", username)

    async def check_lockout(self, username: str) -> None:
        """Check if account is locked and raise exception if so."""
        if await self.is_locked(username):
            # Calculate unlock time (approximate)
            unlock_time = datetime.now(UTC) + timedelta(seconds=self.lockout_secs)
            raise AccountLockoutError(username, unlock_time)


# Global instance - will be initialized in main.py with Redis URL from env
lockout_service: AccountLockoutService | None = None


def get_lockout_service() -> AccountLockoutService:
    """Get the global lockout service instance."""
    global lockout_service  # noqa: PLW0603
    if lockout_service is None:
        # Initialize with environment variables
        redis_url = os.getenv("REDIS_URL")
        lockout_service = AccountLockoutService(
            max_attempts=3,
            lockout_duration=timedelta(minutes=15),
            redis_url=redis_url,
        )
    return lockout_service
