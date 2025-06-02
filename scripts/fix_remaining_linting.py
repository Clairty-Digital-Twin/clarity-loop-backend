#!/usr/bin/env python3
"""Script to fix all remaining linting issues in firestore_client.py."""

from pathlib import Path


def fix_firestore_client() -> None:
    """Fix all remaining linting issues in firestore_client.py."""
    file_path = Path("src/clarity/storage/firestore_client.py")
    content = file_path.read_text(encoding="utf-8")

    # Fix logging calls that still have variable interpolation issues
    fixes = [
        # Fix logging calls that still have variable interpolation issues
        ('logger.exception("Failed to update document %s/%s", collection, document_id, e)',
         'logger.exception("Failed to update document %s/%s", collection, document_id)'),
        ('logger.exception("Failed to delete document %s/%s", collection, document_id, e)',
         'logger.exception("Failed to delete document %s/%s", collection, document_id)'),
        ('logger.exception("Failed to query documents in %s", collection, e)',
         'logger.exception("Failed to query documents in %s", collection)'),
        ('logger.exception("Failed to count documents in %s", collection, e)',
         'logger.exception("Failed to count documents in %s", collection)'),
        ('logger.exception("Failed to delete documents in %s", collection, e)',
         'logger.exception("Failed to delete documents in %s", collection)'),
        ('logger.exception("Failed to batch create documents in %s", collection, e)',
         'logger.exception("Failed to batch create documents in %s", collection)'),
        ('logger.exception("Error closing Firestore client: %s", e)',
         'logger.exception("Error closing Firestore client")'),
        ('logger.exception("Firestore health check failed: %s", e)',
         'logger.exception("Firestore health check failed")'),
        ('logger.exception("Failed to get health data for user %s", user_id, e)',
         'logger.exception("Failed to get health data for user %s", user_id)'),
        ('logger.exception("Failed to get processing status for %s", processing_id, e)',
         'logger.exception("Failed to get processing status for %s", processing_id)'),
        ('logger.exception("Failed to delete health data for user %s", user_id, e)',
         'logger.exception("Failed to delete health data for user %s", user_id)'),
        ('logger.exception("Failed to save health data for user %s", user_id, e)',
         'logger.exception("Failed to save health data for user %s", user_id)'),
        ('logger.exception("Failed to get health data for user %s", user_id, e)',
         'logger.exception("Failed to get health data for user %s", user_id)'),
        ('logger.exception("Failed to initialize FirestoreHealthDataRepository: %s", e)',
         'logger.exception("Failed to initialize FirestoreHealthDataRepository")'),
        ('logger.exception("Failed to cleanup FirestoreHealthDataRepository: %s", e)',
         'logger.exception("Failed to cleanup FirestoreHealthDataRepository")'),
        ('logger.exception("Failed to get health summary for user %s", user_id, e)',
         'logger.exception("Failed to get health summary for user %s", user_id)'),
        ('logger.exception("Failed to delete health data for user %s", user_id, e)',
         'logger.exception("Failed to delete health data for user %s", user_id)'),
    ]

    # Apply all fixes
    for old, new in fixes:
        content = content.replace(old, new)

    # Write the fixed content back
    file_path.write_text(content, encoding="utf-8")
    print("Fixed remaining linting issues in firestore_client.py")


if __name__ == "__main__":
    fix_firestore_client()
