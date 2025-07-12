"""Test security module."""

from unittest.mock import Mock, patch

import pytest

from clarity.core.security import (
    DataIntegrityChecker,
    SecureHashGenerator,
    create_secure_cache_key,
    create_secure_hash,
    generate_data_checksum,
    generate_request_id,
    generate_secure_token,
    verify_data_integrity,
)


class TestSecureHashGenerator:
    """Test SecureHashGenerator class."""

    def test_hash_generation(self):
        """Test secure hash generation."""
        generator = SecureHashGenerator()

        hash1 = generator.hash_string("test_data")
        hash2 = generator.hash_string("test_data")
        hash3 = generator.hash_string("different_data")

        # Same input should produce same hash
        assert hash1 == hash2
        # Different input should produce different hash
        assert hash1 != hash3
        # Hash should be non-empty string
        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_hash_with_salt(self):
        """Test hash generation with salt."""
        generator = SecureHashGenerator()

        # Test with different algorithms produces different hashes
        generator_sha256 = SecureHashGenerator(algorithm="sha256")
        generator_sha512 = SecureHashGenerator(algorithm="sha512")

        hash_sha256 = generator_sha256.hash_string("test_data")
        hash_sha512 = generator_sha512.hash_string("test_data")

        # Different algorithms should produce different hashes
        assert hash_sha256 != hash_sha512


class TestSecurityFunctions:
    """Test security utility functions."""

    def test_create_secure_hash(self):
        """Test create_secure_hash function."""
        hash1 = create_secure_hash("test_data")
        hash2 = create_secure_hash("test_data")
        hash3 = create_secure_hash("different_data")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars

    def test_create_secure_cache_key(self):
        """Test create_secure_cache_key function."""
        key1 = create_secure_cache_key("user", 123, "data")
        key2 = create_secure_cache_key("user", 123, "data")
        key3 = create_secure_cache_key("user", 456, "data")

        assert key1 == key2
        assert key1 != key3
        assert isinstance(key1, str)

        # Test truncation
        long_key = create_secure_cache_key("a" * 1000, truncate=True)
        assert len(long_key) <= 250  # Default max length

    def test_generate_secure_token(self):
        """Test generate_secure_token function."""
        token1 = generate_secure_token()
        token2 = generate_secure_token()

        assert token1 != token2  # Should be unique
        assert len(token1) == 64  # Default length * 2 (hex encoding)

        # Test custom length
        token3 = generate_secure_token(16)
        assert len(token3) == 32  # 16 * 2

    def test_generate_request_id(self):
        """Test request ID generation."""
        id1 = generate_request_id()
        id2 = generate_request_id()

        assert isinstance(id1, str)
        assert id1.startswith("req_")
        assert id1 != id2  # Should be unique

        # Test custom prefix
        id3 = generate_request_id("custom")
        assert id3.startswith("custom_")


class TestDataIntegrity:
    """Test data integrity functions."""

    def test_generate_data_checksum(self):
        """Test checksum generation."""
        checksum1 = generate_data_checksum("test data")
        checksum2 = generate_data_checksum("test data")
        checksum3 = generate_data_checksum("different data")

        assert checksum1 == checksum2
        assert checksum1 != checksum3
        assert isinstance(checksum1, str)

        # Test with bytes
        checksum_bytes = generate_data_checksum(b"test data")
        assert checksum_bytes == checksum1

    def test_verify_data_integrity(self):
        """Test data integrity verification."""
        data = "important data"
        checksum = generate_data_checksum(data)

        # Valid checksum
        assert verify_data_integrity(data, checksum) is True

        # Invalid checksum
        assert verify_data_integrity(data, "invalid_checksum") is False

        # Modified data
        assert verify_data_integrity("modified data", checksum) is False

    def test_data_integrity_checker_class(self):
        """Test DataIntegrityChecker class."""
        checker = DataIntegrityChecker()

        data = "test data"
        checksum = checker.generate_checksum(data)

        assert checker.verify_integrity(data, checksum) is True
        assert checker.verify_integrity("wrong data", checksum) is False
