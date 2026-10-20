#!/usr/bin/env python3
"""Fine-tune A' from a yaml config.   python scripts/train_aprime.py CONFIG [k.sub=v ...] [--out DIR] [--smoke N]

``--smoke N``: N steps on ONE repeated batch, asserts all losses finite and total decreasing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from models.aprime_runner import load_cfg, train
from utils.misc import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.overrides)
    sev = cfg["degradation"]["severity"]
    out = Path(a.out) if a.out else REPO_ROOT / "outputs" / "aprime" / sev / cfg["name"]
    train(cfg, out, smoke=a.smoke)


if __name__ == "__main__":
    main()
