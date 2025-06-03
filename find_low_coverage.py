#!/usr/bin/env python3
"""Find files with lowest test coverage."""

import json
import sys

try:
    with open('coverage.json') as f:
        data = json.load(f)

    files = []
    for file, info in data['files'].items():
        if file.startswith('src/clarity') and not file.endswith('__init__.py'):
            percent = info['summary']['percent_covered']
            if percent < 80:  # Only show files below 80%
                files.append((percent, file))

    files.sort()
    print("🎯 LOWEST COVERAGE TARGETS:")
    for percent, file in files[:15]:
        print(f'{percent:5.1f}% {file}')

except FileNotFoundError:
    print("coverage.json not found - run coverage first")
    sys.exit(1)
