#!/usr/bin/env python3
"""Simple script to fix F821 undefined variable errors in firestore_client.py."""

from pathlib import Path


def fix_f821_errors() -> None:
    """Fix F821 undefined variable errors."""
    file_path = Path("src/clarity/storage/firestore_client.py")
    content = file_path.read_text(encoding="utf-8")

    # Simple replacements to fix F821 errors
    replacements = [
        # Fix exception handlers that reference 'e' but don't bind it
        ('except Exception:\n            logger.exception("Failed to update document {collection}/{document_id}: %s")\n            msg = f"Document update failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to update document %s/%s", collection, document_id)\n            msg = f"Document update failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        ('except Exception:\n            logger.exception("Failed to delete document {collection}/{document_id}: %s")\n            msg = f"Document deletion failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to delete document %s/%s", collection, document_id)\n            msg = f"Document deletion failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        ('except Exception:\n            logger.exception("Failed to store health data: %s")\n            msg = f"Health data storage failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to store health data")\n            msg = f"Health data storage failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        ('except Exception:\n            logger.exception("Failed to query documents in {collection}: %s")\n            msg = f"Query operation failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to query documents in %s", collection)\n            msg = f"Query operation failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        ('except Exception:\n            logger.exception("Failed to count documents in {collection}: %s")\n            msg = f"Count operation failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to count documents in %s", collection)\n            msg = f"Count operation failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        ('except Exception:\n            logger.exception("Failed to delete documents in {collection}: %s")\n            msg = f"Delete operation failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to delete documents in %s", collection)\n            msg = f"Delete operation failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        ('except Exception:\n            logger.exception("Failed to batch create documents in {collection}: %s")\n            msg = f"Batch create operation failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to batch create documents in %s", collection)\n            msg = f"Batch create operation failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        ('except Exception:\n            logger.exception("Firestore health check failed: %s")\n            return {\n                "status": "unhealthy",\n                "error": str(e),\n                "timestamp": datetime.now(UTC),\n            }',
         'except Exception as e:\n            logger.exception("Firestore health check failed")\n            return {\n                "status": "unhealthy",\n                "error": str(e),\n                "timestamp": datetime.now(UTC),\n            }'),
    ]

    # Apply replacements
    for old, new in replacements:
        content = content.replace(old, new)

    # Write back
    file_path.write_text(content, encoding="utf-8")
    print("Fixed F821 errors in firestore_client.py")


if __name__ == "__main__":
    fix_f821_errors() 