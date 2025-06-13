#!/usr/bin/env python3
import re


def fix_caplog_and_indentation(file_path) -> None:
    """Fix caplog usage and indentation in decorator test files."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # First pass: Replace caplog.at_level patterns
    content = re.sub(
        r"with caplog\.at_level\(logging\.(INFO|DEBUG|WARNING)\):",
        r'caplog.set_level(logging.\1, logger="clarity.core.decorators")',
        content,
    )

    # Second pass: Fix indentation by removing 4 spaces from lines that were inside with blocks
    lines = content.split("\n")
    fixed_lines = []
    in_caplog_block = False

    for line in lines:
        # Detect caplog.set_level lines
        if "caplog.set_level(" in line:
            fixed_lines.append(line)
            in_caplog_block = True
            continue

        # If we're in a caplog block and hit a line with same or less indentation as caplog line, end the block
        if in_caplog_block:
            # Look for the next method definition or assert to end the block
            if (
                line.strip().startswith("assert ")
                or line.strip().startswith("def ")
                or line.strip().startswith("class ")
                or (line.strip() and not line.startswith("    "))
            ):
                in_caplog_block = False
                fixed_lines.append(line)
                continue

            # Remove 4 spaces of indentation from lines that were inside with blocks
            if line.startswith("            "):  # 12 spaces (was inside with block)
                line = line[4:]  # Remove 4 spaces
            elif line.startswith("        "):  # 8 spaces
                # This might be normal indentation, keep as is
                pass

        fixed_lines.append(line)

    content = "\n".join(fixed_lines)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# Fix both files
fix_caplog_and_indentation("tests/core/test_decorators_comprehensive.py")
fix_caplog_and_indentation("tests/core/test_decorators_production.py")
