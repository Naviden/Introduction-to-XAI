#!/usr/bin/env python3
"""Execute marked Python examples and compare their documented stdout.

Only run this on trusted repository documentation: the marked blocks are code.
Examples run in separate Python processes and temporary directories, using only
the standard library. No notebook runtime, model download or optional extra is
needed. A missing required example, malformed block or output mismatch fails.
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EXAMPLES = {"shap-logit", "lime-kernel", "rulefit-rules"}
EXAMPLE = re.compile(
    r"<!-- checked-example: ([a-z0-9-]+) -->\s*"
    r"```python\n(.*?)\n```\s*```text\n(.*?)\n```\s*"
    r"<!-- /checked-example -->",
    re.DOTALL,
)


def check_document(path: Path, timeout: float = 10.0) -> list[str]:
    """Check each marked code/output pair; raise ValueError on any failure."""
    source = path.read_text(encoding="utf-8")
    matches = list(EXAMPLE.finditer(source))
    if (source.count("<!-- checked-example:") != len(matches)
            or source.count("<!-- /checked-example -->") != len(matches)):
        raise ValueError(f"{path}: malformed checked-example block")
    names = []
    for match in matches:
        name, code, expected = match.groups()
        with tempfile.TemporaryDirectory(prefix="xai-worked-example-") as cwd:
            try:
                result = subprocess.run(
                    [sys.executable, "-I", "-c", code], cwd=cwd,
                    text=True, capture_output=True, timeout=timeout, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError(f"{path} [{name}]: exceeded {timeout}s") from exc
        if result.returncode:
            raise ValueError(f"{path} [{name}]: execution failed\n{result.stderr}")
        actual_lines = result.stdout.strip().splitlines()
        expected_lines = expected.strip().splitlines()
        if actual_lines != expected_lines:
            diff = "\n".join(difflib.unified_diff(
                expected_lines, actual_lines,
                fromfile="documented output", tofile="computed output", lineterm="",
            ))
            raise ValueError(f"{path} [{name}]: output mismatch\n{diff}")
        names.append(name)
    return names


def check_repository(root: Path = ROOT, timeout: float = 10.0) -> list[str]:
    names = []
    for path in sorted(root.glob("Tools/*/How *.md")):
        names.extend(check_document(path, timeout=timeout))
    missing = REQUIRED_EXAMPLES - set(names)
    if missing:
        raise ValueError(f"Missing required worked examples: {', '.join(sorted(missing))}")
    if len(names) != len(set(names)):
        raise ValueError("Worked example identifiers must be unique")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Time limit in seconds for each code block")
    args = parser.parse_args()
    try:
        names = check_repository(timeout=args.timeout)
    except (ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    for name in names:
        print(f"PASS {name}")
    print(f"{len(names)} documented worked examples passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
