#!/usr/bin/env python3
"""Find, convert and sanity-check the Astra 2 RGB intrinsics.  Read-only on the data; prints everything.

  1. inspect  -- look into the raw archives / .bag files, print every camera_info, compare bags:
       python scripts/check_intrinsics.py inspect --raw /path/to/raw [--max-bags 6] [--archives stair1 stair40] [--workdir /tmp/intr]
       (zip / tar / 7z archives are opened one .bag at a time; needs a `7z` binary or `pip install py7zr`)
  2. convert  -- full-resolution RGB K -> K of the 528x688 frame (crop + resize from configs/dataset/ours.yaml):
       python scripts/check_intrinsics.py convert --K 1050.2 1049.8 960.1 540.7 [--size 1920 1080]
  3. check    -- geometry test on real frames (needs the StairNet cache): treads must be flat and parallel:
       python scripts/check_intrinsics.py check --K 1050.2 1049.8 960.1 540.7 [--frames 30]

`check` unprojects the sensor depth with K and, per confident StairNet region, reports:
  * plane rms of treads (mm)       -> detects lens distortion / depth-RGB misalignment.  A wrong fx, fy or
                                      principal point is a LINEAR map of the points: planes stay planes and
                                      parallel planes stay parallel, so these two numbers are BLIND to scale errors.
  * tread-riser angle              -> should be ~90 deg; reacts to a wrong principal point / camera tilt.
  * step rise / run in metres      -> scale-sensitive: rise ~ 1/fy.  Real stairs are ~0.14-0.19 m rise,
                                      ~0.25-0.32 m run; far outside = suspicious fy or alignment.
So the numbers from the .bag are authoritative; this only catches gross problems.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import yaml

from utils.misc import REPO_ROOT, data_root
from utils.rosbag_min import is_ros_bag, parse_camera_info, parse_ob_disparity, parse_ob_profile, read_bag


# ------------------------------------------------------------------ convert
def convert_K(K, size, crop, target_size):
    """Full-frame K (fx, fy, cx, cy) -> K after ``crop=(y0,y1,x0,x1)`` and resize to ``target_size=(h,w)``.

    Pixel-centre convention (what cv2/torch resize does): u' = (u - x0 + 0.5) * s - 0.5.
    """
    fx, fy, cx, cy = K
    y0, y1, x0, x1 = crop if crop else (0, size[1], 0, size[0])
    sy, sx = (target_size[0] / (y1 - y0), target_size[1] / (x1 - x0)) if target_size else (1.0, 1.0)
    return (fx * sx, fy * sy, (cx - x0 + 0.5) * sx - 0.5, (cy - y0 + 0.5) * sy - 0.5)


def dataset_geometry():
    c = yaml.safe_load((REPO_ROOT / "configs/dataset/ours.yaml").read_text())
    return c.get("crop"), c.get("target_size")


# ------------------------------------------------------------------ inspect
TEXT_EXT = {".json", ".txt", ".yaml", ".yml", ".ini", ".xml", ".csv", ".cfg", ".md", ".log"}
_NAT = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]  # noqa: E731


def _7z_bin():
    return next((b for b in ("7z", "7za", "7zr", "7zz") if shutil.which(b)), None)


def list_7z(p: Path) -> list[tuple[str, int]]:
    """[(member path, size)] via the 7z binary, else py7zr."""
    b = _7z_bin()
    if b:
        out = subprocess.run([b, "l", "-slt", str(p)], capture_output=True, text=True, check=True).stdout
        res, cur = [], {}
        for line in out.splitlines() + [""]:
            if not line.strip():
                if cur.get("Path") and cur.get("Folder", "-") != "+" and "Size" in cur:
                    res.append((cur["Path"], int(cur["Size"] or 0)))
                cur = {}
            elif " = " in line:
                k, _, v = line.partition(" = ")
                cur[k.strip()] = v
        return res[1:] if res and res[0][0] == str(p) else res
    try:
        import py7zr
    except ImportError:
        sys.exit("no 7z binary and no py7zr: `pip install py7zr` (or install p7zip) and re-run")
    with py7zr.SevenZipFile(p) as z:
        return [(i.filename, i.uncompressed) for i in z.list() if not i.is_directory]


def extract_7z(p: Path, members: list[str], dst: Path) -> list[Path]:
    b = _7z_bin()
    if b:
        subprocess.run([b, "x", "-y", f"-o{dst}", str(p), *members], capture_output=True, check=True)
    else:
        import py7zr

        with py7zr.SevenZipFile(p) as z:
            z.extract(path=dst, targets=members)
    return [dst / m for m in members]


def candidates(raw: Path, workdir: Path, max_bags: int, member: str | None, only: list[str] | None = None):
    """Yield ``(bag path, origin)``: loose .bag files, or one bag extracted at a time from zip / tar / 7z archives.

    Archives are taken evenly spread over the sorted list (unless ``only`` names them), so a few
    archives cover different staircases / sessions.
    """
    loose = [p for p in sorted(raw.rglob("*.bag"), key=_NAT) if p.is_file()]
    arch = [p for p in sorted(raw.rglob("*"), key=_NAT)
            if p.is_file() and (p.suffix in (".7z", ".zip", ".tar", ".tgz") or p.name.endswith((".tar.gz", ".tar.bz2")))]
    if only:
        arch = [p for p in arch if any(o == p.stem or o == p.name for o in only)]
    elif len(arch) > max_bags:
        arch = [arch[round(i * (len(arch) - 1) / max(max_bags - 1, 1))] for i in range(max_bags)]
    for p in loose[:max_bags]:
        yield p, None
    for p in arch:
        try:
            if p.suffix == ".7z":
                mem = list_7z(p)
                bags = [m for m, _ in mem if m.endswith(".bag") and (not member or member in m)]
                print(f"[archive] {p.name}: {len(mem)} files, {len(bags)} .bag; contents (first 25): "
                      f"{[(m, f'{sz / 1e6:.1f}MB') for m, sz in mem[:25]]}")
                texts = [m for m, sz in mem if Path(m).suffix.lower() in TEXT_EXT and sz < 300_000]
                if texts:
                    d = workdir / p.stem
                    for t in extract_7z(p, texts, d):
                        body = t.read_text(errors="replace") if t.exists() else ""
                        hit = re.search(r"(?i)intrinsic|focal|\bfx\b|principal", body)
                        print(f"  [text] {t.name}: {len(body)} chars{'  <-- mentions intrinsics:' if hit else ''}")
                        if hit:
                            print("    " + body[:1200].replace("\n", "\n    "))
                for m in bags[:1]:
                    dst = extract_7z(p, [m], workdir / p.stem)[0]
                    yield dst, f"{p.name}:{m}"
            elif zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as z:
                    names = [i for i in z.infolist() if i.filename.endswith(".bag")]
                    print(f"[archive] {p.name}: {len(z.infolist())} entries, {len(names)} .bag")
                    for info in names[:1]:
                        dst = workdir / Path(info.filename).name
                        with z.open(info) as s_, open(dst, "wb") as o:
                            shutil.copyfileobj(s_, o, 1 << 24)
                        yield dst, f"{p.name}:{info.filename}"
            elif tarfile.is_tarfile(p):
                with tarfile.open(p) as t:
                    for m in t:
                        if m.name.endswith(".bag"):
                            dst = workdir / Path(m.name).name
                            with t.extractfile(m) as s_, open(dst, "wb") as o:
                                shutil.copyfileobj(s_, o, 1 << 24)
                            yield dst, f"{p.name}:{m.name}"
                            break
        except Exception as e:  # noqa: BLE001
            print(f"[archive] {p}: {type(e).__name__}: {e}")


def inspect(a) -> None:
    raw, wd = Path(a.raw), Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    if raw.is_file():
        raw = raw.parent if raw.suffix not in (".zip", ".tar", ".tgz", ".gz") else raw
    report, rows = [], []
    for bag, origin in candidates(raw if raw.is_dir() else raw.parent, wd, a.max_bags, a.member, a.archives):
        print(f"\n=== {bag}  ({origin or 'loose'}, {bag.stat().st_size / 1e6:.0f} MB)")
        if not is_ros_bag(bag):
            head = open(bag, "rb").read(64)
            print("  NOT a ROS1 bag. first bytes:", head[:32], "\n  -> Orbbec native format? send me this line; "
                  "try pyorbbecsdk playback (Pipeline(playback_device=...)) instead.")
            report.append({"bag": str(bag), "format": "unknown", "head": head[:32].hex()})
            continue
        info = read_bag(bag)
        print(f"  {len(info['connections'])} connections, {info['n_chunks']} chunks")
        entry = {"bag": str(bag), "origin": origin, "topics": [], "camera_info": {}}
        for c in sorted(info["connections"].values(), key=lambda c: c["topic"]):
            n = info["counts_in_scanned_chunks"].get(c["topic"], 0)
            print(f"  {c['topic']:<55} {c['type']:<35} msgs(first chunks)={n}")
            entry["topics"].append([c["topic"], c["type"]])
            if c["topic"] not in info["first"]:
                continue
            msg = info["first"][c["topic"]]
            try:
                if c["type"].endswith("CameraInfo"):
                    ci = parse_camera_info(msg)
                    kind = "color" if any(w in c["topic"].lower() for w in ("color", "rgb")) else "other"
                elif c["type"].endswith("OBStreamProfileInfo"):
                    p = parse_ob_profile(msg)
                    fx, fy, cx, cy = p["K4"]
                    ci = {"width": p["width"], "height": p["height"], "fx": fx, "fy": fy, "cx": cx, "cy": cy, "D": p["D"],
                          "distortion_model": f"ob:{p['distortion_model']}", "stream_type": p["stream_type"],
                          "rotation": p["rotation"], "translation": p["translation"], "fps": p["fps"]}
                    kind = {2: "color", 3: "depth"}.get(p["stream_type"], f"type{p['stream_type']}")
                    if not (0.3 < cx / p["width"] < 0.7 and 0.3 < cy / p["height"] < 0.7):
                        print("    WARNING: cameraIntrinsic order is not fx,fy,cx,cy (cx,cy not near the image centre):", p["K4"])
                elif c["type"].endswith("OBDisparityParam"):
                    print("    -> disparity engine:", parse_ob_disparity(msg))
                    continue
                else:
                    continue
            except Exception as e:  # noqa: BLE001
                print("    cannot decode:", type(e).__name__, e)
                continue
            ci["kind"] = kind
            entry["camera_info"][c["topic"]] = ci
            print(f"    -> [{kind}] {ci['width']}x{ci['height']} fx={ci['fx']:.3f} fy={ci['fy']:.3f} cx={ci['cx']:.3f} "
                  f"cy={ci['cy']:.3f} model={ci['distortion_model']} D={np.round(ci['D'], 5).tolist()}")
            if "rotation" in ci:
                print(f"       extrinsics (to the other sensor): R={np.round(ci['rotation'], 4).tolist()} t={np.round(ci['translation'], 4).tolist()}")
            if kind == "color":
                rows.append((Path(bag).name, ci))
        if not entry["camera_info"]:
            print("  no sensor_msgs/CameraInfo topic. message definitions that mention intrinsics:")
            for c in info["connections"].values():
                if any(w in c["definition"].lower() for w in ("intrinsic", "focal", "fx", "principal")):
                    print(f"  [{c['topic']}] {c['type']}\n{c['definition'][:600]}")
        report.append(entry)
    if rows:
        print("\n=== RGB intrinsics across bags (same camera => identical numbers)")
        for name, ci in rows:
            print(f"  {name:<40} {ci['width']}x{ci['height']}  fx={ci['fx']:.3f} fy={ci['fy']:.3f} cx={ci['cx']:.3f} cy={ci['cy']:.3f}")
        K = np.array([[c["fx"], c["fy"], c["cx"], c["cy"]] for _, c in rows])
        print("  spread (max-min) over bags:", np.round(K.max(0) - K.min(0), 3), "px  (0 = one calibration for all)")
        _print_convert(tuple(K[0]), (rows[0][1]["width"], rows[0][1]["height"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("\nreport ->", out)


def _print_convert(K, size):
    crop, tgt = dataset_geometry()
    if tuple(size) != (1920, 1080):
        print(f"  WARNING: stream is {size[0]}x{size[1]}, the saved RGB PNGs are 1920x1080: rescale K first "
              f"(fx,cx by 1920/{size[0]}, fy,cy by 1080/{size[1]}) or tell me.")
    k = convert_K(K, size, crop, tgt)
    print(f"  K for the {tgt[1]}x{tgt[0]} frame (crop {crop}, resize): fx={k[0]:.3f} fy={k[1]:.3f} cx={k[2]:.3f} cy={k[3]:.3f}")
    print(f"  -> configs/dataset/ours.yaml:   intrinsics: [{k[0]:.3f}, {k[1]:.3f}, {k[2]:.3f}, {k[3]:.3f}]")


# ------------------------------------------------------------------- check
def unproject(depth, K):
    fx, fy, cx, cy = K
    v, u = np.mgrid[: depth.shape[0], : depth.shape[1]].astype(np.float64)
    return np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1)


def fit_plane(P):
    """Robust plane fit; returns (normal, rms_m, inlier_frac)."""
    keep = np.ones(len(P), bool)
    for _ in range(3):
        c = P[keep].mean(0)
        n = np.linalg.svd(P[keep] - c, full_matrices=False)[2][-1]
        r = (P - c) @ n
        s = max(np.sqrt((r[keep] ** 2).mean()), 1e-4)
        keep = np.abs(r) < 3 * s
    return n, float(np.sqrt((r[keep] ** 2).mean())), float(keep.mean())


def check(a) -> None:
    from scipy import ndimage

    from data.degradation import IdentityDegradation
    from data.loaders import build_dataset
    from data.stairnet.cache import load_masks

    crop, tgt = dataset_geometry()
    K = tuple(a.K_frame) if a.K_frame else convert_K(a.K, tuple(a.size), crop, tgt)
    print("K used for the frame:", np.round(K, 3).tolist())
    cfg = yaml.safe_load((REPO_ROOT / "configs/dataset/ours.yaml").read_text())
    ds = build_dataset({k: cfg[k] for k in ("name", "loader", "root", "crop", "target_size", "min_depth", "max_depth")},
                       degradation=IdentityDegradation())
    cache = data_root() / a.cache
    variants = {"K": K, "fx,fy -10%": (K[0] * .9, K[1] * .9, K[2], K[3]), "fx,fy +10%": (K[0] * 1.1, K[1] * 1.1, K[2], K[3]),
                "fx only +10%": (K[0] * 1.1, K[1], K[2], K[3])}
    res = {k: {"rms_mm": [], "rel_pct": [], "angle_deg": [], "tr_dev": [], "rise": [], "run": []} for k in variants}
    from metrics.stair_geometry import step_geometry, up_vector
    idx = np.linspace(0, len(ds) - 1, min(a.frames, len(ds))).astype(int)
    used = 0
    for i in idx:
        s = ds[i]
        m = load_masks(cache, s["meta"]["sample_id"], s["gt_depth"].shape)
        if m is None:
            print("no StairNet cache for", s["meta"]["sample_id"], "- run scripts/stairnet_cache.py")
            continue
        tread = (m["cls"] == 1) & (m["conf"] >= 0.7) & s["mask"]
        tread = ndimage.binary_erosion(tread, iterations=3)  # keep away from the mask border
        lab, n = ndimage.label(tread)
        comps = [lab == j for j in range(1, n + 1) if (lab == j).sum() >= a.min_px]
        if not comps:
            continue
        used += 1
        for name, k in variants.items():
            P = unproject(s["gt_depth"].astype(np.float64), k)
            ns = []
            risers = ndimage.binary_erosion((m["cls"] == 2) & (m["conf"] >= 0.7) & s["mask"], iterations=3)
            rl, rn = ndimage.label(risers)
            rnorm = [fit_plane(P[rl == j])[0] for j in range(1, rn + 1) if (rl == j).sum() >= a.min_px]
            for c in comps:
                nrm, rms, _ = fit_plane(P[c])
                zc = float(np.median(s["gt_depth"][c]))
                res[name]["rms_mm"].append(rms * 1000)
                res[name]["rel_pct"].append(100 * rms / zc)
                ns.append(nrm)
            for nt in ns[:1]:  # nearest-to-label-order tread vs every riser
                for nr in rnorm:
                    res[name]["tr_dev"].append(abs(90 - np.degrees(np.arccos(min(1.0, abs(float(nt @ nr)))))))
            gt = s["gt_depth"]
            upv = up_vector(gt, tread & (gt > 0), k, min_px=a.min_px)
            if upv is not None:
                g = step_geometry(gt, tread, k, upv, min_px=a.min_px)
                res[name]["rise"] += [abs(x) for x in g["rise"] if np.isfinite(x)]
                res[name]["run"] += [x for x in g["run"] if np.isfinite(x)]
            for p in range(len(ns)):
                for q in range(p + 1, len(ns)):
                    res[name]["angle_deg"].append(np.degrees(np.arccos(min(1.0, abs(float(ns[p] @ ns[q]))))))
    print(f"\nframes with usable treads: {used}/{len(idx)}")
    print(f"{'K variant':<16}{'plane rms mm (median/p90)':<30}{'rms % of z (median)':<22}{'angle between treads deg (median/p90)'}")
    for name, r in res.items():
        if not r["rms_mm"]:
            continue
        q = lambda x, p: np.percentile(x, p) if len(x) else float("nan")  # noqa: E731
        print(f"{name:<16}{q(r['rms_mm'], 50):7.2f} / {q(r['rms_mm'], 90):<18.2f}{q(r['rel_pct'], 50):<22.3f}"
              f"{q(r['angle_deg'], 50):6.2f} / {q(r['angle_deg'], 90):.2f}")
    print(f"\n{'K variant':<16}{'|tread-riser angle - 90| deg (median/p90)':<46}{'step rise m (median)':<24}{'step run m (median)'}")
    for name, r in res.items():
        q = lambda x, p: np.percentile(x, p) if len(x) else float("nan")  # noqa: E731
        print(f"{name:<16}{q(r['tr_dev'], 50):6.2f} / {q(r['tr_dev'], 90):<36.2f}{q(r['rise'], 50):<24.3f}{q(r['run'], 50):.3f}")
    print("\nhow to read: plane rms ~ sensor noise (<~1 % of z) and tread-tread angle of a few degrees => no lens distortion /"
          " misalignment problem (these cannot see a wrong focal length). Rise should be ~0.14-0.19 m and move with the fy variants:"
          " that is the only scale-sensitive line. Thin far treads make the angle columns noisy; trust the medians.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect")
    i.add_argument("--raw", required=True)
    i.add_argument("--max-bags", type=int, default=6)
    i.add_argument("--member", help="only archive members whose name contains this")
    i.add_argument("--archives", nargs="*", help="only these archives, e.g. stair1 stair40 (default: evenly spread)")
    i.add_argument("--workdir", default="/tmp/intrinsics_inspect")
    i.add_argument("--out", default="outputs/intrinsics_report.json")
    c = sub.add_parser("convert")
    c.add_argument("--K", type=float, nargs=4, required=True, metavar=("FX", "FY", "CX", "CY"))
    c.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"))
    k = sub.add_parser("check")
    k.add_argument("--K", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"), help="full-resolution RGB K")
    k.add_argument("--K-frame", type=float, nargs=4, help="K already in the 528x688 frame")
    k.add_argument("--size", type=int, nargs=2, default=(1920, 1080))
    k.add_argument("--frames", type=int, default=30)
    k.add_argument("--min-px", type=int, default=400)
    k.add_argument("--cache", default="ours/stairnet_cache")
    a = ap.parse_args()
    if a.cmd == "inspect":
        inspect(a)
    elif a.cmd == "convert":
        _print_convert(tuple(a.K), tuple(a.size))
    else:
        if not (a.K or a.K_frame):
            sys.exit("give --K (full-res) or --K-frame")
        check(a)


if __name__ == "__main__":
    main()
