"""
Account Lockout Protection Service

Provides brute force protection by tracking failed login attempts
and temporarily locking accounts after too many failures.
"""

import asyncio
import time
import json
import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from clarity.core.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class AccountLockoutError(AuthenticationError):
    """Raised when an account is temporarily locked due to too many failed attempts."""
    
    def __init__(self, username: str, unlock_time: datetime):
        self.username = username
        self.unlock_time = unlock_time
        super().__init__(
            f"Account {username} is locked until {unlock_time.isoformat()}. "
            f"Too many failed login attempts."
        )


class AccountLockoutService:
    """
    Account lockout service with support for both in-memory and Redis persistence.
    
    Provides brute force protection by tracking failed login attempts
    and temporarily locking accounts after configurable thresholds.
    """
    
    def __init__(
        self,
        max_attempts: int = 5,
        lockout_duration: timedelta = timedelta(minutes=15),
        attempt_window: timedelta = timedelta(minutes=15),
        redis_url: Optional[str] = None,
        redis_key_prefix: str = "clarity:lockout:",
    ):
        """
        Initialize the account lockout service.
        
        Args:
            max_attempts: Maximum failed attempts before lockout
            lockout_duration: How long to lock the account
            attempt_window: Time window to consider for counting attempts
            redis_url: Optional Redis URL for persistence (falls back to in-memory)
            redis_key_prefix: Prefix for Redis keys
        """
        self.max_attempts = max_attempts
        self.lockout_duration = lockout_duration
        self.attempt_window = attempt_window
        self.redis_key_prefix = redis_key_prefix
        
        # Initialize Redis if available and URL provided
        self._redis_client: Optional[redis.Redis] = None
        self._use_redis = False
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self._redis_client = redis.from_url(redis_url, decode_responses=True)
                self._use_redis = True
                logger.info("✅ Redis lockout persistence enabled: %s", redis_url)
            except Exception as e:
                logger.warning("❌ Redis connection failed, using in-memory storage: %s", e)
                self._use_redis = False
        elif redis_url:
            logger.warning("❌ Redis URL provided but redis package not installed, using in-memory storage")
        
        # Fallback in-memory storage
        # Format: {username: {'attempts': [(timestamp, ip), ...], 'locked_until': datetime}}
        self._user_attempts: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        if not self._use_redis:
            logger.info("📝 Using in-memory lockout storage (will reset on restart)")
        
        logger.info(
            "🔒 AccountLockoutService initialized: max_attempts=%d, lockout_duration=%s, persistence=%s",
            self.max_attempts, self.lockout_duration, "Redis" if self._use_redis else "in-memory"
        )
    
    def _cleanup_old_attempts(self, username: str) -> None:
        """Remove attempts older than the attempt window."""
        if username not in self._user_attempts:
            return
            
        cutoff_time = datetime.now() - self.attempt_window
        user_data = self._user_attempts[username]
        
        # Keep only recent attempts
        user_data['attempts'] = [
            (timestamp, ip) for timestamp, ip in user_data['attempts']
            if timestamp > cutoff_time
        ]
        
        # Clean up empty records
        if not user_data['attempts'] and 'locked_until' not in user_data:
            del self._user_attempts[username]
    
    async def record_failed_attempt(self, username: str, ip_address: Optional[str] = None) -> None:
        """
        Record a failed login attempt for the user.
        
        Args:
            username: The username that failed to authenticate
            ip_address: Optional IP address of the attempt
        """
        async with self._lock:
            current_time = datetime.now()
            
            if self._use_redis:
                # Redis-based implementation
                user_data = await self._get_user_data_redis(username)
                
                # Clean up old attempts
                cutoff_time = current_time - self.attempt_window
                user_data["attempts"] = [
                    (timestamp, ip) for timestamp, ip in user_data["attempts"]
                    if timestamp > cutoff_time
                ]
                
                # Add new failed attempt
                user_data["attempts"].append((current_time, ip_address))
                
                # Check if we should lock the account
                if len(user_data["attempts"]) >= self.max_attempts:
                    user_data["locked_until"] = current_time + self.lockout_duration
                    logger.warning(
                        "🔒 Account locked (Redis): username=%s, attempts=%d, locked_until=%s, ip=%s",
                        username, len(user_data["attempts"]), 
                        user_data["locked_until"].isoformat(), ip_address
                    )
                else:
                    logger.info(
                        "🚨 Failed attempt recorded (Redis): username=%s, attempts=%d/%d, ip=%s",
                        username, len(user_data["attempts"]), self.max_attempts, ip_address
                    )
                
                # Save to Redis
                await self._set_user_data_redis(username, user_data)
                
            else:
                # In-memory implementation (existing code)
                # Initialize user data if needed
                if username not in self._user_attempts:
                    self._user_attempts[username] = {'attempts': []}
                
                # Clean up old attempts first
                self._cleanup_old_attempts(username)
                
                # Re-initialize if cleanup removed the user (can happen if no recent attempts)
                if username not in self._user_attempts:
                    self._user_attempts[username] = {'attempts': []}
                
                # Add the new failed attempt
                self._user_attempts[username]['attempts'].append((current_time, ip_address))
                
                # Check if we should lock the account
                attempt_count = len(self._user_attempts[username]['attempts'])
                
                if attempt_count >= self.max_attempts:
                    # Lock the account
                    locked_until = current_time + self.lockout_duration
                    self._user_attempts[username]['locked_until'] = locked_until
                    
                    logger.warning(
                        "🔒 Account locked (memory): username=%s, attempts=%d, locked_until=%s, ip=%s",
                        username, attempt_count, locked_until.isoformat(), ip_address
                    )
                else:
                    logger.info(
                        "🚨 Failed attempt recorded (memory): username=%s, attempts=%d/%d, ip=%s",
                        username, attempt_count, self.max_attempts, ip_address
                    )
    
    async def is_account_locked(self, username: str) -> bool:
        """
        Check if an account is currently locked.
        
        Args:
            username: The username to check
            
        Returns:
            True if the account is locked, False otherwise
        """
        if username not in self._user_attempts:
            return False
        
        user_data = self._user_attempts[username]
        locked_until = user_data.get('locked_until')
        
        if not locked_until:
            return False
        
        # Check if lockout has expired
        if datetime.now() >= locked_until:
            # Lockout expired, clean up
            del user_data['locked_until']
            user_data['attempts'] = []  # Reset attempts after lockout expires
            logger.info(f"Account lockout expired: username={username}")
            return False
        
        return True
    
    async def get_lockout_info(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get lockout information for a user.
        
        Returns:
            Dictionary with lockout info or None if not locked
        """
        if not await self.is_account_locked(username):
            return None
        
        user_data = self._user_attempts[username]
        locked_until = user_data['locked_until']
        
        return {
            'username': username,
            'locked_until': locked_until,
            'attempts_count': len(user_data['attempts']),
            'time_remaining_seconds': int((locked_until - datetime.now()).total_seconds())
        }
    
    async def get_lockout_status(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get lockout status for a user, including attempt count even if not locked.
        
        Returns:
            Dictionary with status info or None if no attempts recorded
        """
        if username not in self._user_attempts:
            return None
        
        # Clean up old attempts first
        self._cleanup_old_attempts(username)
        
        user_data = self._user_attempts[username]
        attempts = len(user_data['attempts'])
        
        if attempts == 0 and 'locked_until' not in user_data:
            return None
        
        is_locked = await self.is_account_locked(username)
        result = {
            'attempts': attempts,
            'locked': is_locked,
            'max_attempts': self.max_attempts
        }
        
        if is_locked:
            locked_until = user_data['locked_until']
            result['unlock_time'] = locked_until
            result['time_remaining_seconds'] = int((locked_until - datetime.now()).total_seconds())
        
        return result
    
    async def reset_attempts(self, username: str) -> None:
        """
        Reset failed attempts for a user (called after successful login).
        
        Args:
            username: The username to reset
        """
        if username in self._user_attempts:
            # Keep the user record but clear attempts and lockout
            self._user_attempts[username] = {'attempts': []}
            logger.info(f"Reset failed attempts for user: {username}")
    
    async def check_lockout(self, username: str) -> None:
        """
        Check if account is locked and raise exception if so.
        
        Args:
            username: The username to check
            
        Raises:
            AccountLockoutError: If the account is currently locked
        """
        if await self.is_account_locked(username):
            lockout_info = await self.get_lockout_info(username)
            raise AccountLockoutError(username, lockout_info['locked_until'])
    
    async def check_and_enforce_lockout(self, username: str) -> None:
        """
        Alias for check_lockout for backward compatibility.
        
        Args:
            username: The username to check
            
        Raises:
            AccountLockoutError: If the account is currently locked
        """
        await self.check_lockout(username)
    
    async def get_attempt_count(self, username: str) -> int:
        """Get the current number of failed attempts for a user."""
        if username not in self._user_attempts:
            return 0
        
        self._cleanup_old_attempts(username)
        return len(self._user_attempts[username]['attempts'])
    
    async def admin_unlock_account(self, username: str) -> bool:
        """
        Admin function to manually unlock an account.
        
        Returns:
            True if account was locked and is now unlocked, False if wasn't locked
        """
        if username not in self._user_attempts:
            return False
        
        user_data = self._user_attempts[username]
        was_locked = 'locked_until' in user_data
        
        if was_locked:
            del user_data['locked_until']
            user_data['attempts'] = []
            logger.warning(f"Admin unlock: username={username}")
        
        return was_locked
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about current lockout state."""
        total_users = len(self._user_attempts)
        locked_users = sum(
            1 for user_data in self._user_attempts.values()
            if 'locked_until' in user_data and user_data['locked_until'] > datetime.utcnow()
        )
        
        return {
            'total_users_with_attempts': total_users,
            'currently_locked_users': locked_users,
            'max_attempts': self.max_attempts,
            'lockout_duration_minutes': self.lockout_duration.total_seconds() / 60,
            'attempt_window_minutes': self.attempt_window.total_seconds() / 60
        }

    def _get_redis_key(self, username: str) -> str:
        """Get Redis key for a username."""
        return f"{self.redis_key_prefix}{username}"
    
    async def _get_user_data_redis(self, username: str) -> Dict[str, Any]:
        """Get user lockout data from Redis."""
        if not self._redis_client:
            return {"attempts": []}
        
        try:
            key = self._get_redis_key(username)
            data = await self._redis_client.get(key)
            if data:
                parsed = json.loads(data)
                # Convert timestamp strings back to datetime objects
                attempts = []
                for attempt in parsed.get("attempts", []):
                    timestamp = datetime.fromisoformat(attempt[0])
                    ip = attempt[1]
                    attempts.append((timestamp, ip))
                
                result = {"attempts": attempts}
                if "locked_until" in parsed:
                    result["locked_until"] = datetime.fromisoformat(parsed["locked_until"])
                
                return result
            return {"attempts": []}
        except Exception as e:
            logger.exception("Failed to get user data from Redis: %s", e)
            return {"attempts": []}
    
    async def _set_user_data_redis(self, username: str, data: Dict[str, Any]) -> None:
        """Set user lockout data in Redis."""
        if not self._redis_client:
            return
        
        try:
            key = self._get_redis_key(username)
            
            # Convert datetime objects to ISO strings for JSON serialization
            serializable_data = {"attempts": []}
            for timestamp, ip in data.get("attempts", []):
                serializable_data["attempts"].append([timestamp.isoformat(), ip])
            
            if "locked_until" in data:
                serializable_data["locked_until"] = data["locked_until"].isoformat()
            
            # Set with TTL based on lockout duration + attempt window for cleanup
            ttl_seconds = int((self.lockout_duration + self.attempt_window).total_seconds())
            await self._redis_client.setex(key, ttl_seconds, json.dumps(serializable_data))
            
        except Exception as e:
            logger.exception("Failed to set user data in Redis: %s", e)
    
    async def _delete_user_data_redis(self, username: str) -> None:
        """Delete user lockout data from Redis."""
        if not self._redis_client:
            return
        
        try:
            key = self._get_redis_key(username)
            await self._redis_client.delete(key)
        except Exception as e:
            logger.exception("Failed to delete user data from Redis: %s", e)


# Global instance - in production this should be injected via DI
lockout_service = AccountLockoutService()