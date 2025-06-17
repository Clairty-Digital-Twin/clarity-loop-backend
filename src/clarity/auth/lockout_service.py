"""
Account Lockout Protection Service

Provides brute force protection by tracking failed login attempts
and temporarily locking accounts after too many failures.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

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
    Lightweight account lockout service using in-memory storage.
    
    For production with multiple instances, this should be backed by Redis
    or DynamoDB, but for now we'll use a simple in-memory approach.
    """
    
    def __init__(
        self,
        max_attempts: int = 5,
        lockout_duration_minutes: int = 15,
        attempt_window_minutes: int = 60
    ):
        self.max_attempts = max_attempts
        self.lockout_duration = timedelta(minutes=lockout_duration_minutes)
        self.attempt_window = timedelta(minutes=attempt_window_minutes)
        
        # In-memory storage for failed attempts and lockouts
        # Format: {username: {'attempts': [(timestamp, ip), ...], 'locked_until': datetime}}
        self._user_attempts: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            f"AccountLockoutService initialized: "
            f"max_attempts={max_attempts}, "
            f"lockout_duration={lockout_duration_minutes}min, "
            f"attempt_window={attempt_window_minutes}min"
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
        current_time = datetime.now()
        
        # Initialize user data if needed
        if username not in self._user_attempts:
            self._user_attempts[username] = {'attempts': []}
        
        # Clean up old attempts first
        self._cleanup_old_attempts(username)
        
        # Add the new failed attempt
        self._user_attempts[username]['attempts'].append((current_time, ip_address))
        
        # Check if we should lock the account
        attempt_count = len(self._user_attempts[username]['attempts'])
        
        if attempt_count >= self.max_attempts:
            # Lock the account
            locked_until = current_time + self.lockout_duration
            self._user_attempts[username]['locked_until'] = locked_until
            
            logger.warning(
                f"Account locked: username={username}, "
                f"attempts={attempt_count}, "
                f"locked_until={locked_until.isoformat()}, "
                f"ip={ip_address}"
            )
        else:
            logger.info(
                f"Failed attempt recorded: username={username}, "
                f"attempts={attempt_count}/{self.max_attempts}, "
                f"ip={ip_address}"
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
            result['time_remaining_seconds'] = int((locked_until - datetime.utcnow()).total_seconds())
        
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


# Global instance - in production this should be injected via DI
lockout_service = AccountLockoutService()