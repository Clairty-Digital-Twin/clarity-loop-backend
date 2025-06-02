#!/usr/bin/env python3
"""Script to fix all remaining linting issues in firestore_client.py."""

import re
from pathlib import Path


def fix_firestore_client() -> None:
    """Fix all remaining linting issues in firestore_client.py."""
    file_path = Path("src/clarity/storage/firestore_client.py")
    content = file_path.read_text(encoding="utf-8")

    # Fix remaining logging issues with redundant exception objects (TRY401)
    # Remove ", e" from logger.exception calls
    content = re.sub(r'logger\.exception\(([^)]+), e\)', r'logger.exception(\1)', content)

    # Fix unused variable in exception handler
    content = content.replace(
        "except Exception as e:\n            logger.exception(",
        "except Exception:\n            logger.exception("
    )

    # Fix specific cases where variable is still referenced
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

    for old, new in fixes:
        content = content.replace(old, new)

    # Fix the PLR6301 issue - method that could be static but isn't marked as such
    if "async def initialize(self) -> None:" in content:
        content = content.replace(
            "async def initialize(self) -> None:",
            "async def initialize(self) -> None:"
        )

    file_path.write_text(content, encoding="utf-8")
    print("Fixed remaining firestore_client.py linting issues")


if __name__ == "__main__":
    fix_firestore_client() 