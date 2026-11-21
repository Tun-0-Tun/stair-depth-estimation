#!/usr/bin/env python3
"""Cut ONE frame out of the column strips that make_viz.py wrote, one PNG per image (for slides).

    python scripts/crop_frame.py --dir outputs/viz_fail20_zeroshot --id s24      # or --row 4 (1-based, order of --ids)

Writes <dir>/frame_<id>/{rgb,input,gt,<method>...,err_<method>...}.png at the original 688x528 size.
"""
import argparse
import json
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--id")
ap.add_argument("--row", type=int, help="1-based row of the contact sheet")
ap.add_argument("--gap", type=int, default=12, help="same as make_viz.py --gap")
a = ap.parse_args()
d = Path(a.dir)
ids = [f["id"] for f in json.loads((d / "viz_meta.json").read_text())["frames"]]
r = ids.index(a.id) if a.id else a.row - 1
out = d / f"frame_{ids[r]}"
out.mkdir(exist_ok=True)
for p in d.glob("*.png"):
    if p.name.startswith("sheet_"):
        continue
    img = cv2.imread(str(p))
    h = (img.shape[0] - (len(ids) - 1) * a.gap) // len(ids)
    cv2.imwrite(str(out / p.name), img[r * (h + a.gap) : r * (h + a.gap) + h])
print(f"row {r + 1} = {ids[r]} -> {out}")
