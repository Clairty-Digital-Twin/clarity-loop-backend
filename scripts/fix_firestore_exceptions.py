#!/usr/bin/env python3
"""Fix exception handling in firestore_client.py."""

import re
from pathlib import Path


def fix_firestore_exceptions() -> None:
    """Fix exception handling issues in firestore_client.py."""
    file_path = Path("src/clarity/storage/firestore_client.py")
    content = file_path.read_text(encoding="utf-8")

    # Fix all "except Exception:" that should be "except Exception as e:"
    # Look for patterns where Exception is caught but 'e' is used later
    content = re.sub(
        r'except Exception:\n(\s+)logger\.exception\([^)]+\)\n(\s+)([^}]*\{e\}[^}]*|.*msg = f"[^"]*\{e\}[^"]*"|.*from e)',
        r'except Exception as e:\n\1logger.exception(\1)\n\2\3',
        content,
        flags=re.MULTILINE
    )
    
    # More specific pattern for the health check exception
    content = content.replace(
        'except Exception:\n            logger.exception("Firestore health check failed: %s")\n            return {\n                "status": "unhealthy",\n                "error": str(e),',
        'except Exception as e:\n            logger.exception("Firestore health check failed: %s")\n            return {\n                "status": "unhealthy",\n                "error": str(e),'
    )

    # Fix specific exception blocks that reference 'e'
    patterns = [
        (r'except Exception:\n(\s+)logger\.exception\([^)]+\)\n(\s+)if isinstance\(e,', 
         r'except Exception as e:\n\1logger.exception(\1)\n\2if isinstance(e,'),
        (r'except Exception:\n(\s+)logger\.exception\([^)]+\)\n(\s+)msg = f"[^"]*\{e\}[^"]*"', 
         r'except Exception as e:\n\1logger.exception(\1)\n\2msg = f"\2{e}"'),
        (r'except Exception:\n(\s+)logger\.exception\([^)]+\)\n(\s+)raise [^(]+\([^)]+\) from e', 
         r'except Exception as e:\n\1logger.exception(\1)\n\2raise \2(\2) from e'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    file_path.write_text(content, encoding="utf-8")
    print("Fixed exception handling in firestore_client.py")


if __name__ == "__main__":
    fix_firestore_exceptions() 