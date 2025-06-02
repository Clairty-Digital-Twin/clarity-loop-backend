#!/usr/bin/env python3
"""Script to fix all remaining linting issues in firestore_client.py"""

import re

def fix_firestore_client():
    """Fix all linting issues in firestore_client.py"""
    
    with open('src/clarity/storage/firestore_client.py', 'r') as f:
        content = f.read()
    
    # Fix logging f-strings (G004) - convert to % formatting
    logging_patterns = [
        (r'logger\.info\(f"([^"]*){([^}]+)}([^"]*)"\)', r'logger.info("\1%s\3", \2)'),
        (r'logger\.debug\(f"([^"]*){([^}]+)}([^"]*)"\)', r'logger.debug("\1%s\3", \2)'),
        (r'logger\.warning\(f"([^"]*){([^}]+)}([^"]*)"\)', r'logger.warning("\1%s\3", \2)'),
        (r'logger\.error\(f"([^"]*){([^}]+)}([^"]*)"\)', r'logger.error("\1%s\3", \2)'),
        (r'logger\.exception\(f"([^"]*){([^}]+)}([^"]*)"\)', r'logger.exception("\1%s\3", \2)'),
    ]
    
    for pattern, replacement in logging_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Fix specific complex logging cases
    content = content.replace(
        'logger.info(f"Health data stored with processing ID: {processing_id}")',
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
    content = re.sub(r'logger\.exception\(([^)]+): \{e\}"\)', r'logger.exception(\1")', content)
    
    # Fix exception chaining (B904) - add "from e"
    content = re.sub(r'raise FirestoreError\(msg\)$', r'raise FirestoreError(msg) from e', content, flags=re.MULTILINE)
    content = re.sub(r'raise ConnectionError\(msg\)$', r'raise ConnectionError(msg) from e', content, flags=re.MULTILINE)
    
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
        'self._db = firestore.AsyncClient(  # type: ignore',
        'self._db = firestore.AsyncClient(  # type: ignore[misc]'
    )
    
    # Fix private member access
    content = content.replace(
        'if not firebase_admin._apps:  # type: ignore[misc]',
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
    
    with open('src/clarity/storage/firestore_client.py', 'w') as f:
        f.write(content)
    
    print("Fixed firestore_client.py linting issues")

if __name__ == "__main__":
    fix_firestore_client() 