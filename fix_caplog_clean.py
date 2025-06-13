#!/usr/bin/env python3
import re

def fix_caplog_clean(file_path):
    """Fix caplog usage by adding set_level before with blocks."""
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

# Fix both files
fix_caplog_clean('tests/core/test_decorators_comprehensive.py')
fix_caplog_clean('tests/core/test_decorators_production.py')