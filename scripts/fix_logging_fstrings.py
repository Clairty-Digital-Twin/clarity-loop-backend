#!/usr/bin/env python3
"""Fix G004 logging f-string errors by converting to lazy formatting."""

from pathlib import Path
import re
import sys


def fix_logging_fstring(line: str) -> str:
    """Convert f-string logging to lazy formatting."""
    # Match logging statements with f-strings
    patterns = [
        (r'(logger\.\w+)\(f"([^"]+)"\)', r'\1("\2")'),
        (r"(logger\.\w+)\(f'([^']+)'\)", r'\1("\2")'),
        (r'(logging\.\w+)\(f"([^"]+)"\)', r'\1("\2")'),
        (r"(logging\.\w+)\(f'([^']+)'\)", r'\1("\2")'),
    ]

    for pattern, _replacement in patterns:
        if re.search(pattern, line):
            # Extract the f-string content
            match = re.search(pattern, line)
            if match:
                method = match.group(1)
                fstring_content = match.group(2)

                # Find all {expressions} in the f-string
                expr_pattern = r"\{([^}]+)\}"
                expressions = re.findall(expr_pattern, fstring_content)

                if expressions:
                    # Replace {expr} with %s in the string
                    format_string = re.sub(expr_pattern, "%s", fstring_content)

                    # Build the new logging call
                    args = ", ".join(expressions)
                    return re.sub(pattern, f'{method}("{format_string}", {args})', line)

    return line


def process_file(filepath: Path) -> int:
    """Process a single file and return number of changes."""
    changes = 0

    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

    new_lines = []
    for line in lines:
        new_line = fix_logging_fstring(line)
        if new_line != line:
            changes += 1
        new_lines.append(new_line)

    if changes > 0:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print(f"Fixed {changes} f-string(s) in {filepath}")
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
            return 0

    return changes


def main() -> None:
    """Main function."""
    # Get all Python files
    root = Path.cwd()
    python_files = list(root.rglob("*.py"))

    # Exclude venv, node_modules, etc.
    exclude_dirs = {
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".git",
        "htmlcov",
        "dist",
    }
    python_files = [
        f for f in python_files if not any(ex in f.parts for ex in exclude_dirs)
    ]

    total_changes = 0
    for filepath in python_files:
        changes = process_file(filepath)
        total_changes += changes

    print(f"\n✅ Total f-strings fixed: {total_changes}")


if __name__ == "__main__":
    main()
