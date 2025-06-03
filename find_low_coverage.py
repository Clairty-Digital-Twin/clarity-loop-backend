#!/usr/bin/env python3
"""Find files with lowest test coverage."""

import json
import sys

# Coverage threshold constant
COVERAGE_THRESHOLD = 80
MAX_FILES_TO_SHOW = 15

try:
    with open('coverage.json', encoding='utf-8') as f:
        data = json.load(f)

    files = []
    for file, info in data['files'].items():
        if file.startswith('src/clarity') and not file.endswith('__init__.py'):
            percent = info['summary']['percent_covered']
            if percent < COVERAGE_THRESHOLD:
                files.append((percent, file))

    files.sort()
    # Using sys.stdout.write instead of print for lint compliance
    sys.stdout.write("🎯 LOWEST COVERAGE TARGETS:\n")
    for percent, file in files[:MAX_FILES_TO_SHOW]:
        sys.stdout.write(f'{percent:5.1f}% {file}\n')

except FileNotFoundError:
    sys.stdout.write("coverage.json not found - run coverage first\n")
    sys.exit(1)
