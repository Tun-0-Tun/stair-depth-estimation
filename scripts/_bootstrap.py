"""Make ``scripts/*.py`` runnable as plain files without installing the package.

``python scripts/run_baseline.py ...`` puts ``scripts/`` on ``sys.path``, not the
repo root, so ``import metrics`` would fail.  Every script imports this first.
(If you ``pip install -e .`` this is a no-op.)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
