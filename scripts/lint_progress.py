#!/usr/bin/env python3
"""Track and report lint progress for CLARITY Digital Twin Platform."""

from collections import Counter
from datetime import datetime
import operator
from pathlib import Path
import subprocess
import sys


def run_ruff_check() -> tuple[int, dict[str, int]]:
    """Run ruff check and return total errors and breakdown by code."""
    result = subprocess.run(
        ["ruff", "check", "."],
        capture_output=True,
        text=True, check=False
    )

    # Parse output to count errors by type
    error_counts = Counter()
    total_errors = 0

    for line in result.stdout.splitlines():
        # Look for error codes like "F401", "ANN001", etc.
        if ":" in line and any(char.isdigit() for char in line):
            parts = line.split(":")
            if len(parts) >= 3:
                # Extract error code from format "file.py:line:col: CODE message"
                code_part = parts[-1].strip()
                if " " in code_part:
                    code = code_part.split()[0]
                    if code and code[0].isupper():
                        error_counts[code] += 1

    # Get total from summary line
    for line in result.stdout.splitlines():
        if line.startswith("Found") and "error" in line:
            total_errors = int(line.split()[1])
            break

    return total_errors, dict(error_counts)


def load_baseline() -> dict[str, int]:
    """Load baseline error counts."""
    return {
        "total": 947,
        "breakdown": {
            "F401": 143,  # Unused imports
            "ANN001": 140,  # Missing type annotations
            "PLC0415": 74,  # Import outside top-level
            "S311": 70,  # Random for security
            "NPY002": 68,  # Legacy numpy
            "PLR2004": 59,  # Magic values
            "RUF029": 38,  # Async without await
            "ANN202": 35,  # Missing return annotations
            "S105": 29,  # Hardcoded passwords
            "F841": 28,  # Unused variables
        }
    }


def generate_report(current_total: int, current_breakdown: dict[str, int]) -> None:
    """Generate progress report."""
    baseline = load_baseline()

    print("\n" + "=" * 60)
    print("CLARITY DIGITAL TWIN - Lint Progress Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Overall progress
    improvement = baseline["total"] - current_total
    percent_fixed = (improvement / baseline["total"]) * 100 if baseline["total"] > 0 else 0

    print("\n📊 Overall Progress:")
    print(f"   Baseline errors: {baseline['total']:,}")
    print(f"   Current errors:  {current_total:,}")
    print(f"   Fixed:          {improvement:,} ({percent_fixed:.1f}%)")

    # Progress bar
    progress_width = 40
    filled = int(progress_width * percent_fixed / 100)
    bar = "█" * filled + "░" * (progress_width - filled)
    print(f"   Progress:       [{bar}] {percent_fixed:.1f}%")

    # Top issues
    print("\n🔍 Top 10 Issues:")
    top_issues = sorted(current_breakdown.items(), key=operator.itemgetter(1), reverse=True)[:10]

    for i, (code, count) in enumerate(top_issues, 1):
        baseline_count = baseline["breakdown"].get(code, 0)
        change = baseline_count - count if baseline_count else -count
        change_str = f"+{abs(change)}" if change < 0 else f"-{change}" if change > 0 else "±0"

        # Issue descriptions
        descriptions = {
            "F401": "Unused imports",
            "ANN001": "Missing type annotation",
            "PLC0415": "Import outside top-level",
            "S311": "Random for security",
            "NPY002": "Legacy numpy.random",
            "PLR2004": "Magic value comparison",
            "RUF029": "Async without await",
            "ANN202": "Missing return annotation",
            "S105": "Hardcoded password",
            "F841": "Unused variable",
            "UP035": "Deprecated typing",
            "G004": "f-string in logging",
            "BLE001": "Blind except",
            "DTZ005": "datetime.now() no tz",
            "SIM117": "Multiple with statements",
        }

        desc = descriptions.get(code, "")
        print(f"   {i:2d}. {code:<8} {count:>4} ({change_str:>4})  {desc}")

    # Recommendations
    print("\n💡 Recommendations:")

    if current_breakdown.get("F401", 0) > 50:
        print("   • Run: ruff check --fix --select F401 .")
        print("     Fix unused imports automatically")

    if current_breakdown.get("UP035", 0) > 20:
        print("   • Run: ruff check --fix --select UP035 .")
        print("     Update deprecated typing imports")

    if current_breakdown.get("NPY002", 0) > 30:
        print("   • Use: rng = np.random.default_rng()")
        print("     Replace legacy numpy.random calls")

    if percent_fixed > 10:
        print(f"   • Great progress! Update CI baseline to {current_total}")

    print("\n" + "=" * 60)


def main() -> None:
    """Main entry point."""
    print("🔍 Running ruff check...")
    total, breakdown = run_ruff_check()

    if total == 0:
        print("🎉 Congratulations! No lint errors found!")
        return

    generate_report(total, breakdown)

    # Exit with error if regression detected
    baseline = load_baseline()
    if total > baseline["total"]:
        print("\n❌ ERROR: Lint regression detected!")
        print(f"   Current errors ({total}) exceed baseline ({baseline['total']})")
        sys.exit(1)


if __name__ == "__main__":
    main()
