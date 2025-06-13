#!/usr/bin/env python3
import re
import os

def fix_caplog_in_file(file_path):
    """Fix caplog usage in decorator test files by adding logger specification."""
    if not os.path.exists(file_path):
        print(f'File not found: {file_path}')
        return
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add caplog.set_level before with caplog.at_level blocks
    content = re.sub(
        r'(\s+)with caplog\.at_level\(logging\.(INFO|DEBUG|WARNING)\):',
        r'\1caplog.set_level(logging.\2, logger="clarity.core.decorators")\n\1with caplog.at_level(logging.\2):',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f'Fixed {file_path}')

# Fix all decorator test files
files_to_fix = [
    'tests/core/test_decorators.py',
    'tests/core/test_decorators_production.py'
]

for file_path in files_to_fix:
    fix_caplog_in_file(file_path)