#!/usr/bin/env python3
"""Script to fix TYPE_CHECKING imports that are actually used at runtime."""

import ast
import os
from pathlib import Path
from typing import Set, Dict, List, Tuple


class TypeCheckingAnalyzer(ast.NodeVisitor):
    """Analyze Python AST to find runtime usage of TYPE_CHECKING imports."""
    
    def __init__(self):
        self.type_checking_imports: Dict[str, List[str]] = {}
        self.runtime_type_uses: Set[str] = set()
        self.in_type_checking = False
        self.current_module = None
        
    def visit_If(self, node):
        """Track if we're inside a TYPE_CHECKING block."""
        # Check if this is an if TYPE_CHECKING: block
        if (isinstance(node.test, ast.Name) and node.test.id == 'TYPE_CHECKING') or \
           (isinstance(node.test, ast.Attribute) and node.test.attr == 'TYPE_CHECKING'):
            self.in_type_checking = True
            self.generic_visit(node)
            self.in_type_checking = False
        else:
            self.generic_visit(node)
            
    def visit_ImportFrom(self, node):
        """Track imports inside TYPE_CHECKING blocks."""
        if self.in_type_checking and node.module:
            module = node.module
            names = []
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                names.append(name)
            if module not in self.type_checking_imports:
                self.type_checking_imports[module] = []
            self.type_checking_imports[module].extend(names)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Check function parameters and return types."""
        # Check parameters
        for arg in node.args.args:
            if arg.annotation:
                self._extract_type_names(arg.annotation)
        
        # Check return type
        if node.returns:
            self._extract_type_names(node.returns)
            
        self.generic_visit(node)
        
    def visit_AnnAssign(self, node):
        """Check variable annotations."""
        if node.annotation:
            self._extract_type_names(node.annotation)
        self.generic_visit(node)
        
    def visit_arg(self, node):
        """Check function argument annotations."""
        if node.annotation:
            self._extract_type_names(node.annotation)
        self.generic_visit(node)
        
    def _extract_type_names(self, node):
        """Extract type names from annotation nodes."""
        if isinstance(node, ast.Name):
            self.runtime_type_uses.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For things like typing.Dict
            pass
        elif isinstance(node, ast.Subscript):
            # For things like List[str], Dict[str, Any]
            self._extract_type_names(node.value)
            if isinstance(node.slice, ast.Index):
                # Python < 3.9
                self._extract_type_names(node.slice.value)
            elif isinstance(node.slice, ast.Tuple):
                for elt in node.slice.elts:
                    self._extract_type_names(elt)
            else:
                # Python >= 3.9
                self._extract_type_names(node.slice)
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                self._extract_type_names(elt)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union types (e.g., str | None)
            self._extract_type_names(node.left)
            self._extract_type_names(node.right)


def analyze_file(file_path: Path) -> Tuple[Dict[str, List[str]], Set[str]]:
    """Analyze a Python file for TYPE_CHECKING imports and runtime usage."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        analyzer = TypeCheckingAnalyzer()
        analyzer.visit(tree)
        
        # Get all imported names from TYPE_CHECKING blocks
        all_type_checking_names = set()
        for names in analyzer.type_checking_imports.values():
            all_type_checking_names.update(names)
        
        # Find which TYPE_CHECKING imports are used at runtime
        runtime_used = all_type_checking_names & analyzer.runtime_type_uses
        
        return analyzer.type_checking_imports, runtime_used
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return {}, set()


def fix_file(file_path: Path, type_checking_imports: Dict[str, List[str]], runtime_used: Set[str]):
    """Fix TYPE_CHECKING imports in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if not runtime_used:
        return False
        
    # Process each module that has runtime-used imports
    modified = False
    for module, names in type_checking_imports.items():
        runtime_names = [n for n in names if n in runtime_used]
        type_only_names = [n for n in names if n not in runtime_used]
        
        if not runtime_names:
            continue
            
        # Find the TYPE_CHECKING import for this module
        import_pattern = f"from {module} import"
        
        # Build the new runtime import
        if len(runtime_names) == 1:
            runtime_import = f"from {module} import {runtime_names[0]}"
        else:
            runtime_import = f"from {module} import (\n"
            for name in runtime_names[:-1]:
                runtime_import += f"    {name},\n"
            runtime_import += f"    {runtime_names[-1]},\n)"
            
        # Build the new TYPE_CHECKING import (if any names remain)
        if type_only_names:
            if len(type_only_names) == 1:
                type_import = f"    from {module} import {type_only_names[0]}"
            else:
                type_import = f"    from {module} import (\n"
                for name in type_only_names[:-1]:
                    type_import += f"        {name},\n"
                type_import += f"        {type_only_names[-1]},\n    )"
        else:
            type_import = None
            
        # Replace the import in TYPE_CHECKING block
        lines = content.split('\n')
        in_type_checking = False
        import_added = False
        
        for i in range(len(lines)):
            line = lines[i]
            
            if 'if TYPE_CHECKING:' in line:
                in_type_checking = True
                continue
                
            if in_type_checking and line and not line[0].isspace():
                # Exit TYPE_CHECKING block
                in_type_checking = False
                
            if in_type_checking and import_pattern in line:
                # Found the import to modify
                if type_import:
                    lines[i] = type_import
                else:
                    # Remove the line if no type-only imports remain
                    lines[i] = ''
                    
                # Add runtime import before TYPE_CHECKING block
                # Find the if TYPE_CHECKING line
                for j in range(i, -1, -1):
                    if 'if TYPE_CHECKING:' in lines[j]:
                        # Insert before this line
                        lines.insert(j, runtime_import)
                        import_added = True
                        modified = True
                        break
                break
                
        if import_added:
            content = '\n'.join(lines)
    
    if modified:
        # Clean up empty lines
        lines = content.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            if not line.strip():
                if not prev_empty:
                    cleaned_lines.append(line)
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
                
        content = '\n'.join(cleaned_lines)
        
        # Ensure TYPE_CHECKING block has pass if empty
        if 'if TYPE_CHECKING:' in content:
            lines = content.split('\n')
            for i in range(len(lines)):
                if 'if TYPE_CHECKING:' in lines[i]:
                    # Check if block is empty
                    j = i + 1
                    block_empty = True
                    while j < len(lines) and lines[j] and lines[j][0].isspace():
                        if lines[j].strip() and lines[j].strip() != 'pass':
                            block_empty = False
                            break
                        j += 1
                    
                    if block_empty:
                        # Ensure there's a pass
                        has_pass = False
                        for k in range(i + 1, j):
                            if lines[k].strip() == 'pass':
                                has_pass = True
                                break
                        if not has_pass:
                            lines.insert(i + 1, '    pass')
                    break
            
            content = '\n'.join(lines)
        
        # Write the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return True
        
    return False


def main():
    """Main function to process all Python files."""
    root_dir = Path("/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/clarity-loop-backend")
    
    # Files with TYPE_CHECKING imports from the grep search
    files_to_check = [
        "src/clarity/api/v1/auth.py",
        "src/clarity/api/v1/websocket/chat_handler.py",
        "tests/core/test_exceptions_comprehensive.py",
        "src/clarity/auth/decorators.py",
        "src/clarity/services/health_data_service.py",
        "src/clarity/auth/modal_auth_fix.py",
        "tests/storage/test_mock_repository_comprehensive.py",
        "src/clarity/ml/preprocessing.py",
        "src/clarity/auth/dependencies.py",
        "src/clarity/utils/decorators.py",
        "src/clarity/services/s3_storage_service.py",
        "src/clarity/services/cognito_auth_service.py",
        "src/clarity/api/v1/metrics.py",
        "src/clarity/api/v1/websocket/app_example.py",
        "src/clarity/ml/proxy_actigraphy.py",
        "src/clarity/core/config_provider.py",
        "src/clarity/api/v1/websocket/lifespan.py",
        "src/clarity/api/v1/debug.py",
        "src/clarity/ml/inference_engine.py",
        "src/clarity/storage/mock_repository.py",
        "src/clarity/api/v1/pat_analysis.py",
        "src/clarity/api/v1/health_data.py",
        "src/clarity/ml/analysis_pipeline.py",
        "src/clarity/storage/dynamodb_client.py",
        "src/clarity/api/v1/websocket/connection_manager.py",
        "src/clarity/integrations/apple_watch.py",
        "src/clarity/core/container.py",
        "tests/entrypoints/test_analysis_service.py",
        "src/clarity/api/v1/healthkit_upload.py",
        "src/clarity/core/secure_logging.py",
        "src/clarity/ml/processors/respiration_processor.py",
        "src/clarity/services/messaging/insight_subscriber.py",
        "src/clarity/core/exceptions.py",
        "src/clarity/ml/processors/activity_processor.py",
        "src/clarity/integrations/healthkit.py",
        "tests/ml/test_nhanes_stats.py",
        "src/clarity/main.py",
        "tests/ml/test_pat_optimization.py",
        "tests/integration/test_e2e_health_data_flow.py",
        "src/clarity/services/messaging/analysis_subscriber.py",
        "src/clarity/api/v1/test.py",
        "src/clarity/ports/data_ports.py",
        "tests/ml/test_proxy_actigraphy.py",
        "src/clarity/ml/processors/sleep_processor.py",
        "src/clarity/core/container_aws.py",
        "tests/conftest.py",
        "src/clarity/api/v1/gemini_insights.py",
        "src/clarity/ml/pat_optimization.py",
        "src/clarity/ml/processors/cardio_processor.py",
        "src/clarity/models/auth.py",
        "src/clarity/auth/aws_auth_provider.py",
        "tests/test_startup.py",
        "src/clarity/services/dynamodb_service.py",
        "src/clarity/ports/middleware_ports.py",
        "src/clarity/ports/config_ports.py",
        "src/clarity/core/cloud.py",
        "src/clarity/auth/aws_cognito_provider.py",
        "scripts/api_test_suite.py",
    ]
    
    fixed_files = []
    
    for file_path in files_to_check:
        full_path = root_dir / file_path
        if not full_path.exists():
            print(f"File not found: {full_path}")
            continue
            
        print(f"Analyzing {file_path}...")
        type_checking_imports, runtime_used = analyze_file(full_path)
        
        if runtime_used:
            print(f"  Found runtime-used TYPE_CHECKING imports: {runtime_used}")
            if fix_file(full_path, type_checking_imports, runtime_used):
                fixed_files.append(file_path)
                print(f"  ✓ Fixed {file_path}")
            else:
                print(f"  ✗ Failed to fix {file_path}")
    
    print(f"\nFixed {len(fixed_files)} files:")
    for f in fixed_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()