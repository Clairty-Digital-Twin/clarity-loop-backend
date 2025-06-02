#!/usr/bin/env python3
"""Final script to fix all remaining linting issues in firestore_client.py."""

import re
from pathlib import Path


def fix_firestore_client() -> None:
    """Fix all remaining linting issues in firestore_client.py."""
    file_path = Path("src/clarity/storage/firestore_client.py")
    content = file_path.read_text(encoding="utf-8")

    # Fix exception handlers that lost their 'as e' but still reference 'e'
    # Pattern: except Exception: followed by code that uses 'e'
    
    # First, restore 'as e' where needed
    patterns_to_fix = [
        # Pattern: except Exception:\n...logger.exception...\n...if isinstance(e,
        (r'except Exception:\n(\s+)logger\.exception\([^)]+\)\n(\s+)if isinstance\(e,', 
         r'except Exception as e:\n\1logger.exception(\1)\n\2if isinstance(e,'),
        
        # Pattern: except Exception:\n...logger.exception...\n...msg = f"...{e}"
        (r'except Exception:\n(\s+)logger\.exception\([^)]+\)\n(\s+)([^}]*\{e\}[^}]*)', 
         r'except Exception as e:\n\1logger.exception(\1)\n\2\3'),
    ]
    
    # More targeted fixes for specific exception handling patterns
    exception_blocks = [
        # Document creation
        ('except Exception:\n            logger.exception("Failed to create document in %s", collection)\n            if isinstance(e, (FirestoreValidationError, FirestoreConnectionError)):\n                raise\n            msg = f"Document creation failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to create document in %s", collection)\n            if isinstance(e, (FirestoreValidationError, FirestoreConnectionError)):\n                raise\n            msg = f"Document creation failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        # Document retrieval
        ('except Exception:\n            logger.exception("Failed to get document %s/%s", collection, document_id)\n            msg = f"Document retrieval failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to get document %s/%s", collection, document_id)\n            msg = f"Document retrieval failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        # Document update
        ('except Exception:\n            logger.exception("Failed to update document %s/%s", collection, document_id)\n            msg = f"Document update failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to update document %s/%s", collection, document_id)\n            msg = f"Document update failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        # Document deletion
        ('except Exception:\n            logger.exception("Failed to delete document %s/%s", collection, document_id)\n            msg = f"Document deletion failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to delete document %s/%s", collection, document_id)\n            msg = f"Document deletion failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        # Health data storage
        ('except Exception:\n            logger.exception("Failed to store health data")\n            msg = f"Health data storage failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to store health data")\n            msg = f"Health data storage failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        # Query documents
        ('except Exception:\n            logger.exception("Failed to query documents in %s", collection)\n            msg = f"Query operation failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to query documents in %s", collection)\n            msg = f"Query operation failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        # Count documents
        ('except Exception:\n            logger.exception("Failed to count documents in %s", collection)\n            msg = f"Count operation failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to count documents in %s", collection)\n            msg = f"Count operation failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        # Delete documents
        ('except Exception:\n            logger.exception("Failed to delete documents in %s", collection)\n            msg = f"Delete operation failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to delete documents in %s", collection)\n            msg = f"Delete operation failed: {e}"\n            raise FirestoreError(msg) from e'),
        
        # Batch create
        ('except Exception:\n            logger.exception("Failed to batch create documents in %s", collection)\n            msg = f"Batch create operation failed: {e}"\n            raise FirestoreError(msg) from e',
         'except Exception as e:\n            logger.exception("Failed to batch create documents in %s", collection)\n            msg = f"Batch create operation failed: {e}"\n            raise FirestoreError(msg) from e'),
    ]
    
    for old, new in exception_blocks:
        content = content.replace(old, new)
    
    # Fix any remaining patterns with regex
    content = re.sub(
        r'except Exception:\n(\s+)logger\.exception\([^)]+\)\n(\s+)msg = f"[^"]*\{e\}[^"]*"\n(\s+)raise \w+Error\(msg\) from e',
        r'except Exception as e:\n\1logger.exception(\1)\n\2msg = f"\2{e}"\n\3raise \1Error(msg) from e',
        content
    )

    file_path.write_text(content, encoding="utf-8")
    print("Fixed all remaining firestore_client.py linting issues")


if __name__ == "__main__":
    fix_firestore_client() 