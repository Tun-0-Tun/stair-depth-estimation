#!/usr/bin/env python3
"""Overlay cached StairNet masks / edge band / lines on the RGB for a visual check (~50 frames).

    python scripts/stairnet_viz.py [--per-stairs 2] [--out outputs/stairnet_viz]

Picks up to ``--per-stairs`` shots from every staircase (first and middle shot).  Colours:
tread = green, riser = red, low-confidence (< conf_thr) = grey hatch, edge band = yellow outline,
convex lines blue, concave lines magenta.  The tread/riser assignment is UNVERIFIED until you
looked at these images; if swapped, re-run stairnet_cache.py with --tread-class/--riser-class.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import yaml

from data.degradation import IdentityDegradation
from data.loaders import build_dataset
from data.stairnet.cache import load_masks
from data.stairnet.edges import build_edge_maps
from utils.misc import REPO_ROOT, data_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="ours/stairnet_cache")
    ap.add_argument("--out", default="outputs/stairnet_viz")
    ap.add_argument("--per-stairs", type=int, default=2)
    ap.add_argument("--mode", default="gaussian")
    ap.add_argument("--band-px", type=float, default=6)
    ap.add_argument("--conf-thr", type=float, default=0.7)
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / "configs/dataset/ours.yaml").read_text())
    ds = build_dataset({k: cfg[k] for k in ("name", "loader", "root", "crop", "target_size")},
                       degradation=IdentityDegradation())
    by = defaultdict(list)
    for i, r in enumerate(ds.records):
        by[r["meta"].get("stairs_id")].append(i)
    picks = [i for v in by.values() for i in sorted({v[0], v[len(v) // 2]})[: a.per_stairs]]
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i in picks:
        s = ds[i]
        sid = s["meta"]["sample_id"]
        m = load_masks(data_root() / a.cache, sid, s["rgb"].shape[:2])
        if m is None:
            print("no cache for", sid)
            continue
        img = (s["rgb"][..., ::-1] * 255).astype(np.uint8).copy()  # BGR
        ov = img.copy()
        ov[m["cls"] == 1] = (0, 200, 0)
        ov[m["cls"] == 2] = (0, 0, 220)
        img = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
        lowc = m["conf"] < a.conf_thr
        img[lowc & ((np.add.outer(np.arange(img.shape[0]), np.arange(img.shape[1])) % 8) == 0)] = (160, 160, 160)
        e = build_edge_maps(m["cls"], m["conf"], a.mode, a.band_px, a.conf_thr)
        cnt, _ = cv2.findContours(e["band"].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, (0, 255, 255), 1)
        for c, x1, y1, x2, y2, _ in m["lines"]:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0) if c == 0 else (255, 0, 255), 2)
        cv2.putText(img, f"{sid} stairs {s['meta'].get('stairs_id')}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"{sid}.png"), img)
        tiles.append(cv2.resize(img, (344, 264)))
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[k : k + 6]) for k in range(0, len(tiles), 6)])
        cv2.imwrite(str(out / "contact_sheet.png"), sheet)
    print(f"{len(tiles)} frames -> {out}")


if __name__ == "__main__":
    main()
