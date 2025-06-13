#!/usr/bin/env python3
import re


def fix_caplog_properly(file_path):
    """Fix caplog usage by changing at_level to set_level with logger."""
    with open(file_path) as f:
        content = f.read()

    # Replace with caplog.at_level() with the proper logger-specific version
    # This maintains the with block structure but uses set_level instead
    content = re.sub(
        r"with caplog\.at_level\(logging\.(INFO|DEBUG|WARNING)\):",
        r'caplog.set_level(logging.\1, logger="clarity.core.decorators")\n        with caplog.at_level(logging.\1):',
        content,
    )

    with open(file_path, "w") as f:
        f.write(content)

    print(f"Fixed {file_path}")


# Fix both files
fix_caplog_properly("tests/core/test_decorators_comprehensive.py")
fix_caplog_properly("tests/core/test_decorators_production.py")
