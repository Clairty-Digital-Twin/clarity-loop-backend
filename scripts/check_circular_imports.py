#!/usr/bin/env python3
"""Check for circular imports in the CLARITY codebase.

This script analyzes the import structure to detect potential circular
dependencies that could cause import hanging issues.
"""

import ast
from collections import defaultdict
from pathlib import Path
import sys


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to extract import information from Python files."""

    def __init__(self, module_path: str):
        self.module_path = module_path
        self.imports: list[str] = []
        self.from_imports: list[tuple[str, list[str]]] = []

    def visit_import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_import_from(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements."""
        if node.module:
            names = [alias.name for alias in node.names]
            self.from_imports.append((node.module, names))


def get_python_files(src_dir: Path) -> list[Path]:
    """Get all Python files in the source directory."""
    return list(src_dir.rglob("*.py"))


def get_module_name(file_path: Path, src_dir: Path) -> str:
    """Convert file path to module name."""
    relative_path = file_path.relative_to(src_dir)
    if relative_path.name == "__init__.py":
        module_parts = relative_path.parts[:-1]
    else:
        module_parts = relative_path.parts[:-1] + (relative_path.stem,)

    return ".".join(module_parts) if module_parts else ""


def analyze_imports(file_path: Path) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Analyze imports in a Python file."""
    try:
        content = file_path.read_text(encoding="utf-8")

        tree = ast.parse(content)
        analyzer = ImportAnalyzer(str(file_path))
        analyzer.visit(tree)

    except (OSError, UnicodeDecodeError, SyntaxError) as e:
        print(f"Warning: Could not analyze {file_path}: {e}")
        return [], []
    else:
        return analyzer.imports, analyzer.from_imports


def build_dependency_graph(src_dir: Path) -> dict[str, set[str]]:
    """Build a dependency graph of modules."""
    python_files = get_python_files(src_dir)
    dependency_graph: dict[str, set[str]] = defaultdict(set)

    for file_path in python_files:
        module_name = get_module_name(file_path, src_dir)
        if not module_name:
            continue

        imports, from_imports = analyze_imports(file_path)

        # Process regular imports
        for import_name in imports:
            if import_name.startswith("clarity"):
                dependency_graph[module_name].add(import_name)

        # Process from-imports
        for from_module, _ in from_imports:
            if from_module and from_module.startswith("clarity"):
                dependency_graph[module_name].add(from_module)

    return dependency_graph


def detect_circular_imports(dependency_graph: dict[str, set[str]]) -> list[list[str]]:
    """Detect circular imports using DFS."""
    visited: set[str] = set()
    rec_stack: set[str] = set()
    cycles: list[list[str]] = []

    def dfs(node: str, path: list[str]) -> bool:
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return True

        if node in visited:
            return False

        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in dependency_graph.get(node, set()):
            if dfs(neighbor, path.copy()):
                pass  # Continue to find all cycles

        rec_stack.remove(node)
        return False

    for node in dependency_graph:
        if node not in visited:
            dfs(node, [])

    return cycles


def main() -> None:
    """Main function to check for circular imports."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist")
        sys.exit(1)

    print("🔍 Analyzing import dependencies...")
    dependency_graph = build_dependency_graph(src_dir)

    print(f"📊 Found {len(dependency_graph)} modules")

    print("🔄 Checking for circular imports...")
    cycles = detect_circular_imports(dependency_graph)

    if cycles:
        print(f"❌ Found {len(cycles)} circular import(s):")
        for i, cycle in enumerate(cycles, 1):
            print(f"\n  Cycle {i}:")
            for j, module in enumerate(cycle):
                if j < len(cycle) - 1:
                    print(f"    {module} →")
                else:
                    print(f"    {module}")

        print("\n💡 To fix circular imports:")
        print("  1. Use dependency injection")
        print("  2. Move shared code to a common module")
        print("  3. Use lazy imports (import at function level)")
        print("  4. Create abstract interfaces")

        sys.exit(1)
    else:
        print("✅ No circular imports detected!")

        # Show dependency summary
        if dependency_graph:
            print("\n📈 Dependency Summary:")
            for module, deps in sorted(dependency_graph.items()):
                if deps:
                    print(f"  {module} → {', '.join(sorted(deps))}")

    print("\n🎉 Import analysis completed successfully!")


if __name__ == "__main__":
    main()
