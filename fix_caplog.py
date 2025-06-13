#!/usr/bin/env python3
import re

# Fix test_decorators_comprehensive.py
with open("tests/core/test_decorators_comprehensive.py", encoding="utf-8") as f:
    content = f.read()

# Replace caplog.at_level patterns
content = re.sub(
    r"with caplog\.at_level\(logging\.(INFO|DEBUG|WARNING)\):",
    r'caplog.set_level(logging.\1, logger="clarity.core.decorators")',
    content,
)

# Write back
with open("tests/core/test_decorators_comprehensive.py", "w", encoding="utf-8") as f:
    f.write(content)


# Fix test_decorators_production.py
with open("tests/core/test_decorators_production.py", encoding="utf-8") as f:
    content = f.read()

# Replace caplog.at_level patterns
content = re.sub(
    r"with caplog\.at_level\(logging\.(INFO|DEBUG|WARNING)\):",
    r'caplog.set_level(logging.\1, logger="clarity.core.decorators")',
    content,
)

# Write back
with open("tests/core/test_decorators_production.py", "w", encoding="utf-8") as f:
    f.write(content)
