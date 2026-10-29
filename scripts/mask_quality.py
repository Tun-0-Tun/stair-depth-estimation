#!/usr/bin/env python3
"""How trustworthy are the StairNet masks on each test frame?  (There is no ground-truth mask: use the GT geometry.)

    python scripts/mask_quality.py [--split test] [--out mask_quality.json] [k.sub=v ...]

Per frame, with the SAME cleaned masks the losses and the stair/edge zones use:
  stair_share    share of the frame labelled tread | riser
  tread_inlier   share of tread pixels whose GT surface normal agrees with the frame's "up" vector (|n.u| > 0.85);
                 up = the dominant normal of the tread pixels (as in the planar loss) -- low = the "tread" region is
                 not made of parallel horizontal planes, i.e. the mask is wrong or the scene is not a staircase
  riser_inlier   share of riser pixels perpendicular to up (|n.u| < 0.5)
  score          tread_inlier (NaN without tread / up vector)
Use with ``compare_rows.py --subset mask_quality.json --min-score 0.6`` to compare rows on the frames with good masks.
Needs the frame intrinsics K (configs/dataset/ours.yaml).
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import torch

from data.aprime_dataset import APrimeDataset
from models.aprime_runner import intrinsics, load_cfg
from models.planar_loss import box_smooth, normals_from_depth, robust_up
from utils.misc import REPO_ROOT


def frame_quality(gt, valid, m, K) -> dict:
    tread, riser, band, ign = (np.asarray(m[k], bool) for k in ("tread", "riser", "band", "ignore"))
    out = {"stair_share": float((tread | riser).mean()), "tread_inlier": float("nan"),
           "riser_inlier": float("nan"), "score": float("nan")}
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None, None]  # noqa: E731
    z, v = box_smooth(t(gt.astype(np.float32)), t(valid), 5)
    n, ok = normals_from_depth(z, K, v)
    good = t(valid & ~band & ~ign)
    sel = t(tread) & ok & good
    up, has = robust_up(n, sel, 300)
    if not has[0]:
        return out
    cos = (n * up[:, :, None, None]).sum(1, keepdim=True).abs()
    out["tread_inlier"] = float((cos[sel] > 0.85).float().mean())
    rsel = t(riser) & ok & good
    if rsel.sum() > 100:
        out["riser_inlier"] = float((cos[rsel] < 0.5).float().mean())
    out["score"] = out["tread_inlier"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--config", default="configs/aprime/a1_validity.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="mask_quality.json")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.overrides)
    K = intrinsics(cfg)
    if K is None:
        raise SystemExit("needs data.intrinsics (frame K) in configs/dataset/ours.yaml")
    ds = APrimeDataset(cfg, a.split)
    res = {}
    for i in range(len(ds)):
        s = ds.base[i]
        res[s["meta"]["sample_id"]] = {**frame_quality(s["gt_depth"], s["mask"], ds.masks(s), K),
                                       "stairs_id": s["meta"].get("stairs_id")}
    sc = np.array([r["score"] for r in res.values()])
    ss = np.array([r["stair_share"] for r in res.values()])
    ok = np.isfinite(sc)
    print(f"{len(res)} frames; stair share median {np.median(ss):.2f}; frames with no usable tread region: {(~ok).sum()}")
    if ok.any():
        print("tread_inlier (higher = masks agree with the GT geometry): "
              + "  ".join(f"p{p}={np.percentile(sc[ok], p):.2f}" for p in (10, 25, 50, 75, 90)))
        for thr in (0.4, 0.6, 0.8):
            print(f"  frames with score >= {thr}: {(sc[ok] >= thr).sum()} of {len(res)}")
    (REPO_ROOT / a.out).write_text(json.dumps(res, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
