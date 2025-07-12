"""Security module for Clarity Loop Backend.

This module provides secure parameter management, secrets handling,
and other security-related functionality.
"""

from clarity.security.secrets_manager import SecretsManager, get_secrets_manager

__all__ = ["SecretsManager", "get_secrets_manager"]