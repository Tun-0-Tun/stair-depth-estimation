#!/usr/bin/env python3
"""Pre-commit hook: enforce the one-metric-implementation rule.

Rule 1 of CONTRIBUTING.md says AbsRel / RMSE / delta / FPS are computed in
``metrics/`` and nowhere else.  A rule nobody checks is a rule nobody follows,
so this hook greps staged Python files for the shapes those formulas take.

It is a heuristic, not a proof: it looks for the arithmetic, not for the intent.
False positives are expected occasionally -- silence one with a trailing
``# noqa: metrics`` and say in the PR why the exception is justified.

Usage (pre-commit passes the filenames):
    python scripts/check_single_metrics.py path/to/file.py ...
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Directories allowed to contain metric arithmetic.
ALLOWED = ("metrics/", "tests/", "third_party/", "notebooks/")

#: (pattern, what it looks like) - deliberately narrow, to keep noise down.
PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(abs_rel|absrel|rel_err|rmse_log|rmselog)\s*="), "a metric being computed"),
    (re.compile(r"\bdelta[_ ]?1\s*="), "a delta-1 threshold"),
    (re.compile(r"1\.25\s*\*\*\s*[23]"), "delta-2 / delta-3 thresholds"),
    (re.compile(r"torch\.max\s*\(\s*\w+\s*/\s*\w+\s*,\s*\w+\s*/\s*\w+"), "a delta ratio"),
    (re.compile(r"np\.maximum\s*\(\s*\w+\s*/\s*\w+\s*,\s*\w+\s*/\s*\w+"), "a delta ratio"),
    (re.compile(r"sqrt\s*\(\s*(np\.|torch\.)?mean\s*\(.*\*\*\s*2"), "an RMSE"),
    (re.compile(r"\bfps\s*=\s*1(\.0)?\s*/"), "an FPS computation"),
)

HINT = (
    "    -> import it instead:\n"
    "         from metrics.depth_metrics import compute_depth_metrics, DepthMetricAccumulator\n"
    "         from metrics.runtime_metrics import measure_runtime\n"
    "       See CONTRIBUTING.md rule 1 and docs/metrics_protocol.md.\n"
    "       If this is a genuine false positive, append '# noqa: metrics' to the line."
)


def check(path: Path) -> list[str]:
    try:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    if rel.startswith(ALLOWED) or path.name == Path(__file__).name:
        return []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    problems = []
    for n, line in enumerate(lines, 1):
        if "# noqa: metrics" in line:
            continue
        for pattern, what in PATTERNS:
            if pattern.search(line):
                problems.append(f"{rel}:{n}: looks like {what}\n      {line.strip()}")
                break
    return problems


def main(argv: list[str]) -> int:
    problems: list[str] = []
    for name in argv:
        problems += check(Path(name))

    if not problems:
        return 0

    print("Metric code found outside metrics/ -- there is exactly one implementation:\n")
    for p in problems:
        print(f"  {p}")
    print(f"\n{HINT}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
