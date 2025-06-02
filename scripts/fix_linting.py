#!/usr/bin/env python3
"""Script to fix linting issues in firestore_client.py."""

from pathlib import Path


def fix_firestore_client() -> None:
    """Fix linting issues in firestore_client.py."""
    file_path = Path("src/clarity/storage/firestore_client.py")
    content = file_path.read_text(encoding="utf-8")

    # Fix f-string logging issues
    patterns = [
        ('logger.info(f"Health data stored with processing ID: {processing_id}")',
         'logger.info("Health data stored with processing ID: %s", processing_id)'),
        ('logger.info(f"Health data saved: {processing_id} with {len(metrics)} metrics")',
         'logger.info("Health data saved: %s with %s metrics", processing_id, len(metrics))'),
        ('logger.info(f"Deleted health data for user {user_id}, processing {processing_id}")',
         'logger.info("Deleted health data for user %s, processing %s", user_id, processing_id)'),
    ]

    for pattern, replacement in patterns:
        content = content.replace(pattern, replacement)

    # Fix specific complex logging cases
    content = content.replace(
        'logger.info("Health data stored with processing ID: %s", processing_id)',
        'logger.info("Health data stored with processing ID: %s", processing_id)'
    )

    content = content.replace(
        'logger.info(\n                f"Health data saved: {processing_id} with {len(metrics)} metrics"\n            )',
        'logger.info(\n                "Health data saved: %s with %s metrics", processing_id, len(metrics)\n            )'
    )

    content = content.replace(
        'logger.info(\n                f"Deleted health data for user {user_id}, processing {processing_id}"\n            )',
        'logger.info(\n                "Deleted health data for user %s, processing %s", user_id, processing_id\n            )'
    )

    # Fix exception logging with redundant exception objects (TRY401)
    content = content.replace(
        'logger.exception(f"Failed to create audit log: {e}")',
        'logger.exception("Failed to create audit log")'
    )

    # Fix exception chaining (B904) - add "from e"
    content = content.replace('raise FirestoreError(msg)$', 'raise FirestoreError(msg) from e')
    content = content.replace('raise ConnectionError(msg)$', 'raise ConnectionError(msg) from e')

    # Fix ValueError exception chaining
    content = content.replace(
        'raise FirestoreValidationError(msg)',
        'raise FirestoreValidationError(msg) from None'
    )

    # Fix unused variable in exception handler
    content = content.replace(
        'except Exception as e:\n            logger.exception("Failed to create audit log")',
        'except Exception:\n            logger.exception("Failed to create audit log")'
    )

    # Fix specific type ignore
    content = content.replace(
        'self._db = firestore.AsyncClient(',
        'self._db = firestore.AsyncClient(  # type: ignore[misc]'
    )

    # Fix private member access
    content = content.replace(
        'if not firebase_admin._apps:',
        'if not firebase_admin._apps:  # type: ignore[misc]  # noqa: SLF001'
    )

    # Fix methods that could be static
    content = content.replace(
        'def _cache_key(self, collection: str, doc_id: str) -> str:',
        '@staticmethod\n    def _cache_key(collection: str, doc_id: str) -> str:'
    )

    content = content.replace(
        'async def _validate_health_data(self, data: dict[str, Any]) -> None:',
        '@staticmethod\n    async def _validate_health_data(data: dict[str, Any]) -> None:'
    )

    # Remove test comment at end
    content = content.replace('# Test comment\n', '')

    file_path.write_text(content, encoding="utf-8")
    print("Fixed firestore_client.py linting issues")


if __name__ == "__main__":
    fix_firestore_client() 