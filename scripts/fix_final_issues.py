#!/usr/bin/env python3
"""Fix final remaining linting issues in firestore_client.py."""

from pathlib import Path


def fix_firestore_client() -> None:
    """Fix all remaining linting issues in firestore_client.py."""
    file_path = Path("src/clarity/storage/firestore_client.py")
    content = file_path.read_text(encoding="utf-8")

    # Fix TRY401 - Remove redundant exception objects from logger.exception calls
    content = content.replace(
        'logger.exception("Error closing Firestore client: %s", e)',
        'logger.exception("Error closing Firestore client")'
    )

    content = content.replace(
        'logger.exception("Failed to cleanup FirestoreHealthDataRepository: %s", e)',
        'logger.exception("Failed to cleanup FirestoreHealthDataRepository")'
    )

    # Fix F841 - Remove unused exception variables where they're not needed
    content = content.replace(
        'except Exception as e:\n            logger.exception("Failed to get processing status for %s", processing_id)\n            return None',
        'except Exception:\n            logger.exception("Failed to get processing status for %s", processing_id)\n            return None'
    )

    content = content.replace(
        'except Exception as e:\n            logger.exception("Failed to delete health data for user %s", user_id)\n            return False',
        'except Exception:\n            logger.exception("Failed to delete health data for user %s", user_id)\n            return False'
    )

    # Fix TRY300 issues by restructuring try/except blocks with else clauses
    # Fix query_documents method
    content = content.replace(
        '            return results\n\n        except Exception as e:',
        '\n        except Exception as e:'
    )
    content = content.replace(
        'raise FirestoreError(msg) from e\n\n    async def count_documents(',
        'raise FirestoreError(msg) from e\n        else:\n            return results\n\n    async def count_documents('
    )

    # Fix count_documents method
    content = content.replace(
        '            return count\n\n        except Exception as e:\n            logger.exception("Failed to count documents in %s", collection)',
        '\n        except Exception as e:\n            logger.exception("Failed to count documents in %s", collection)'
    )
    content = content.replace(
        'raise FirestoreError(msg) from e\n\n    async def delete_documents(',
        'raise FirestoreError(msg) from e\n        else:\n            return count\n\n    async def delete_documents('
    )

    # Fix delete_documents method
    content = content.replace(
        '            return deleted_count\n\n        except Exception as e:',
        '\n        except Exception as e:'
    )
    content = content.replace(
        'raise FirestoreError(msg) from e\n\n    async def batch_create_documents(',
        'raise FirestoreError(msg) from e\n        else:\n            return deleted_count\n\n    async def batch_create_documents('
    )

    # Fix save_health_data method in FirestoreHealthDataRepository
    content = content.replace(
        '            return True\n\n        except Exception:',
        '\n        except Exception:'
    )
    content = content.replace(
        'return False\n\n    async def get_user_health_data(',
        'return False\n        else:\n            return True\n\n    async def get_user_health_data('
    )

    # Fix delete_health_data method
    content = content.replace(
        '            return True\n\n        except Exception:',
        '\n        except Exception:'
    )
    content = content.replace(
        'return False\n\n    async def save_data(',
        'return False\n        else:\n            return True\n\n    async def save_data('
    )

    # Fix save_data method
    content = content.replace(
        '            return document_id\n\n        except Exception as e:',
        '\n        except Exception as e:'
    )
    content = content.replace(
        'raise FirestoreError(msg) from e\n\n    async def get_data(',
        'raise FirestoreError(msg) from e\n        else:\n            return document_id\n\n    async def get_data('
    )

    # Fix get_data method
    content = content.replace(
        '            return result\n\n        except Exception as e:',
        '\n        except Exception as e:'
    )
    content = content.replace(
        'raise FirestoreError(msg) from e\n\n    async def initialize(',
        'raise FirestoreError(msg) from e\n        else:\n            return result\n\n    async def initialize('
    )

    # Fix delete_user_data method
    content = content.replace(
        '            return deleted_count\n\n        except Exception as e:',
        '\n        except Exception as e:'
    )
    content = content.replace(
        'raise FirestoreError(msg) from e\n',
        'raise FirestoreError(msg) from e\n        else:\n            return deleted_count\n'
    )

    # Write the fixed content back
    file_path.write_text(content, encoding="utf-8")
    print("Fixed final linting issues in firestore_client.py")


if __name__ == "__main__":
    fix_firestore_client()
