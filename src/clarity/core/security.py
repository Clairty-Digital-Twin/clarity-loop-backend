"""Security utilities for the Clarity Digital Twin platform.

This module provides secure cryptographic functions for hashing, key generation,
and other security-related operations following industry best practices.

All functions use cryptographically secure algorithms and are designed to be
resistant to common attacks while maintaining good performance.
"""

# removed - breaks FastAPI

import hashlib
import secrets

from clarity.core.constants import CACHE_KEY_TRUNCATION_LENGTH, HASH_ALGORITHM


class SecureHashGenerator:
    """Secure hash generator using cryptographically safe algorithms.

    This class provides a clean interface for generating secure hashes
    with consistent configuration and behavior across the application.
    """

    def __init__(self, algorithm: str = HASH_ALGORITHM) -> None:
        """Initialize the hash generator with specified algorithm.

        Args:
            algorithm: Hash algorithm to use (default: sha256)

        Raises:
            ValueError: If the specified algorithm is not supported
        """
        if algorithm not in hashlib.algorithms_available:
            available = ", ".join(sorted(hashlib.algorithms_available))
            msg = f"Unsupported hash algorithm '{algorithm}'. Available: {available}"
            raise ValueError(msg)

        self.algorithm = algorithm

    def hash_string(self, data: str, *, encoding: str = "utf-8") -> str:
        """Generate a secure hash of a string.

        Args:
            data: String data to hash
            encoding: Text encoding to use (default: utf-8)

        Returns:
            Hexadecimal hash digest
        """
        hash_obj = hashlib.new(self.algorithm)
        hash_obj.update(data.encode(encoding))
        return hash_obj.hexdigest()

    def hash_bytes(self, data: bytes) -> str:
        """Generate a secure hash of bytes.

        Args:
            data: Byte data to hash

        Returns:
            Hexadecimal hash digest
        """
        hash_obj = hashlib.new(self.algorithm)
        hash_obj.update(data)
        return hash_obj.hexdigest()

    def generate_cache_key(self, *components: object, truncate: bool = True) -> str:
        """Generate a secure cache key from multiple components.

        Args:
            *components: Variable components to include in the key
            truncate: Whether to truncate the key to a standard length

        Returns:
            Secure cache key string
        """
        # Convert all components to strings and join
        key_data = "_".join(str(component) for component in components)

        # Generate secure hash
        hash_digest = self.hash_string(key_data)

        # Optionally truncate for readability and consistency
        if truncate:
            return hash_digest[:CACHE_KEY_TRUNCATION_LENGTH]

        return hash_digest


# Global instance for convenience
_default_hasher = SecureHashGenerator()


def create_secure_hash(data: str, *, encoding: str = "utf-8") -> str:
    """Create a secure hash of string data using the default hasher.

    Args:
        data: String data to hash
        encoding: Text encoding to use (default: utf-8)

    Returns:
        Hexadecimal hash digest
    """
    return _default_hasher.hash_string(data, encoding=encoding)


def create_secure_cache_key(*components: object, truncate: bool = True) -> str:
    """Create a secure cache key from multiple components.

    This is a convenience function that uses the default hasher instance
    to generate cache keys for common use cases.

    Args:
        *components: Variable components to include in the key
        truncate: Whether to truncate the key to a standard length

    Returns:
        Secure cache key string

    Example:
        >>> create_secure_cache_key("user123", "actigraphy", 1024)
        "a1b2c3d4e5f6789a"
    """
    return _default_hasher.generate_cache_key(*components, truncate=truncate)


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token.

    Args:
        length: Length of the token in bytes (default: 32)

    Returns:
        Hexadecimal token string

    Example:
        >>> generate_secure_token(16)
        "a1b2c3d4e5f67890abcdef1234567890"
    """
    return secrets.token_hex(length)


def generate_request_id(prefix: str = "req") -> str:
    """Generate a secure request ID with optional prefix.

    Args:
        prefix: Prefix for the request ID (default: "req")

    Returns:
        Secure request ID string

    Example:
        >>> generate_request_id("insights")
        "insights_a1b2c3d4"
    """
    token = secrets.token_hex(4)  # 8 character hex string
    return f"{prefix}_{token}"


class DataIntegrityChecker:
    """Utility class for verifying data integrity using checksums.

    This class provides methods to generate and verify data integrity
    checksums for ensuring data hasn't been corrupted or tampered with.
    """

    def __init__(self, algorithm: str = HASH_ALGORITHM) -> None:
        self.hasher = SecureHashGenerator(algorithm)

    def generate_checksum(self, data: str | bytes) -> str:
        """Generate an integrity checksum for data.

        Args:
            data: Data to generate checksum for

        Returns:
            Checksum string
        """
        if isinstance(data, str):
            return self.hasher.hash_string(data)
        return self.hasher.hash_bytes(data)

    def verify_integrity(self, data: str | bytes, expected_checksum: str) -> bool:
        """Verify data integrity against expected checksum.

        Args:
            data: Data to verify
            expected_checksum: Expected checksum value

        Returns:
            True if data integrity is verified, False otherwise
        """
        actual_checksum = self.generate_checksum(data)
        return secrets.compare_digest(actual_checksum, expected_checksum)


# Singleton instance for common use
data_integrity_checker = DataIntegrityChecker()


def verify_data_integrity(data: str | bytes, expected_checksum: str) -> bool:
    """Verify data integrity using the default checker.

    Args:
        data: Data to verify
        expected_checksum: Expected checksum value

    Returns:
        True if data integrity is verified, False otherwise
    """
    return data_integrity_checker.verify_integrity(data, expected_checksum)


def generate_data_checksum(data: str | bytes) -> str:
    """Generate data integrity checksum using the default checker.

    Args:
        data: Data to generate checksum for

    Returns:
        Checksum string
    """
    return data_integrity_checker.generate_checksum(data)
