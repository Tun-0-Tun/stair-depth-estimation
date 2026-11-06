#!/usr/bin/env python3
"""Check the StairNet-DepthOut port against the authors' own val labels (Mendeley 6kffmjt7g2).

    python scripts/stairnet_check_mendeley.py --images DIR/images --seg DIR/segmentation

Reports pixel accuracy / per-class IoU of the 3-class segmentation and the confusion matrix
(rows = label value 0/128/255, cols = net channel) -- the latter shows which channel is which
surface.  Needs the weights; NOT run in the repo yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from data.stairnet.infer import StairNetSeg
from utils.misc import data_root, resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(data_root() / "RGB-D_stair_dataset/val/images"))
    ap.add_argument("--seg", default=str(data_root() / "RGB-D_stair_dataset/val/segmentation"))
    ap.add_argument("--weights", default="weights/stairnet/depth_out/best.pth")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    # identity class map: report the net's raw channels
    net = StairNetSeg(a.weights, device=resolve_device(a.device), tread_class=1, riser_class=2)  # identity: report the net's own channels
    conf = np.zeros((3, 3), np.int64)
    files = sorted(Path(a.images).glob("*.jpg"))[: a.limit or None]
    for f in files:
        seg = cv2.imread(str(Path(a.seg) / (f.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        pred = net.predict(rgb)["cls"]
        lab = np.select([seg == 0, seg == 128, seg == 255], [0, 1, 2], -1)
        ok = lab >= 0
        np.add.at(conf, (lab[ok], pred[ok]), 1)
    iou = np.diag(conf) / (conf.sum(0) + conf.sum(1) - np.diag(conf)).clip(1)
    print(f"frames {len(files)}  PA {np.trace(conf) / conf.sum():.4f}  IoU per class {np.round(iou, 4)}")
    print("confusion (rows label 0/128/255, cols net channel 0/1/2):\n", conf)


if __name__ == "__main__":
    main()
