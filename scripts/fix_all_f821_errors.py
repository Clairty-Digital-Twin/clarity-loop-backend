#!/usr/bin/env python3
"""Script to fix all F821 undefined variable errors in firestore_client.py."""

from pathlib import Path


def fix_firestore_client() -> None:
    """Fix all F821 errors in firestore_client.py."""
    file_path = Path("src/clarity/storage/firestore_client.py")
    content = file_path.read_text(encoding="utf-8")

    # Fix all exception handlers that reference 'e' but don't bind it
    fixes = [
        # Fix exception handlers without 'as e'
        ('except Exception:\n            logger.exception("Failed to get health data for user {user_id}: %s")\n            msg = f"Health data retrieval failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to get health data for user %s", user_id)\n            msg = f"Health data retrieval failed: {e}"\n            raise FirestoreError(msg) from e'),

        ('except Exception:\n            logger.exception("Failed to save health data for user {user_id}: %s")\n            msg = f"Health data save failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to save health data for user %s", user_id)\n            msg = f"Health data save failed: {e}"\n            raise FirestoreError(msg) from e'),

        ('except Exception:\n            logger.exception("Failed to get health data for user {user_id}: %s")\n            msg = f"Health data retrieval failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to get health data for user %s", user_id)\n            msg = f"Health data retrieval failed: {e}"\n            raise FirestoreError(msg) from e'),

        ('except Exception:\n            logger.exception("Failed to initialize FirestoreHealthDataRepository: %s")\n            msg = f"Repository initialization failed: {e}"\n            raise ConnectionError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to initialize FirestoreHealthDataRepository")\n            msg = f"Repository initialization failed: {e}"\n            raise ConnectionError(msg) from e'),

        ('except Exception:\n            logger.exception("Failed to get health summary for user {user_id}: %s")\n            msg = f"Health summary retrieval failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to get health summary for user %s", user_id)\n            msg = f"Health summary retrieval failed: {e}"\n            raise FirestoreError(msg) from e'),

        ('except Exception:\n            logger.exception("Failed to delete health data for user {user_id}: %s")\n            msg = f"Health data deletion failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to delete health data for user %s", user_id)\n            msg = f"Health data deletion failed: {e}"\n            raise FirestoreError(msg) from e'),

        # Fix close method
        ('except Exception:\n            logger.exception("Error closing Firestore client: %s")',
         'except Exception as e:\n            logger.exception("Error closing Firestore client: %s", e)'),

        # Fix cleanup method
        ('except Exception:\n            logger.exception("Failed to cleanup FirestoreHealthDataRepository: %s")',
         'except Exception as e:\n            logger.exception("Failed to cleanup FirestoreHealthDataRepository: %s", e)'),

        # Fix get_processing_status method
        ('except Exception:\n            logger.exception("Failed to get processing status for {processing_id}: %s")',
         'except Exception as e:\n            logger.exception("Failed to get processing status for %s", processing_id)'),

        # Fix delete_health_data method
        ('except Exception:\n            logger.exception("Failed to delete health data for user {user_id}: %s")',
         'except Exception as e:\n            logger.exception("Failed to delete health data for user %s", user_id)'),

        # Fix logging format strings with f-strings
        ('logger.info("Health data saved for user {user_id}: %s", document_id)',
         'logger.info("Health data saved for user %s: %s", user_id, document_id)'),

        ('logger.info("Retrieved {len(documents)} health records for user %s", user_id)',
         'logger.info("Retrieved %s health records for user %s", len(documents), user_id)'),

        ('logger.info("Deleted {deleted_count} health records for user %s", user_id)',
         'logger.info("Deleted %s health records for user %s", deleted_count, user_id)'),
    ]

    # Apply all fixes
    for old, new in fixes:
        content = content.replace(old, new)

    # Write the fixed content back
    file_path.write_text(content, encoding="utf-8")
    print("Fixed all F821 errors in firestore_client.py")


if __name__ == "__main__":
    fix_firestore_client()
