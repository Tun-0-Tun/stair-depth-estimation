#!/usr/bin/env python3
"""Fetch, or explain how to fetch, every dataset and checkpoint we use.

Usage::

    python scripts/download_data.py --list
    python scripts/download_data.py --dataset arkitscenes
    python scripts/download_data.py --dataset hammer --inspect   # show what's on disk
    python scripts/download_data.py --verify                     # check every root
    python scripts/download_data.py --weights depth_anything_v2_vits

Policy
------
Several of these datasets cannot be fetched by a script: they need a form, an
email to the authors, or acceptance of a licence.  For those this tool prints
**exact manual steps** -- the URL, what to click, and where to put the file --
and exits non-zero.  It never fabricates a stand-in, and it never leaves you
guessing whether a run used real data.

Nothing here writes into the repo: everything lands under ``data/raw/`` and
``checkpoints/``, both git-ignored.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401

from utils.misc import REPO_ROOT

DATA_ROOT = REPO_ROOT / "data" / "raw"
CKPT_ROOT = REPO_ROOT / "checkpoints"


@dataclass
class Source:
    key: str
    title: str
    root: Path
    auto: bool
    size: str = "?"
    url: str = ""
    expect: tuple[str, ...] = ()
    """Paths, relative to ``root``, that must exist for the loader to work."""
    steps: tuple[str, ...] = ()
    notes: str = ""


DATASETS: dict[str, Source] = {
    "nyuv2": Source(
        key="nyuv2",
        title="NYU Depth v2",
        root=DATA_ROOT / "nyuv2",
        auto=False,
        size="~4 GB (h5 completion pack) / ~1 GB (npy DSR pack)",
        url="https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html",
        expect=("nyudepthv2/val",),
        steps=(
            "Completion layout (used by DEPTHOR, and by our default config):",
            "  the preprocessed per-sample h5 pack distributed with NLSPN /",
            "  CompletionFormer -- https://github.com/zzangjinsun/NLSPN_ECCV20",
            "  (README -> 'NYU Depth V2' -> Google Drive link).",
            "  Unpack so that data/raw/nyuv2/nyudepthv2/{train,val}/<scene>/*.h5 exists.",
            "Super-resolution layout (used by DuCos): the nyu_data pack from DKN",
            "  https://github.com/cvlab-yonsei/dkn -- needs test_images_v2.npy,",
            "  test_depth.npy AND test_minmax.npy. Put them directly in data/raw/nyuv2/",
            "  and set dataset.layout=npy.",
        ),
    ),
    "arkitscenes": Source(
        key="arkitscenes",
        title="ARKitScenes (depth upsampling subset)",
        root=DATA_ROOT / "arkitscenes",
        auto=True,
        size="~30 GB for the full upsampling split; use --videos to take a subset",
        url="https://github.com/apple/ARKitScenes",
        expect=("upsampling/Validation",),
        steps=(
            "Apple's own downloader is used (it is a plain python script, no account):",
            "  git clone https://github.com/apple/ARKitScenes /tmp/ARKitScenes",
            "  python /tmp/ARKitScenes/download_data.py upsampling --split Validation \\",
            "      --download_dir data/raw/arkitscenes",
            "Add --video_id <id> to fetch a single scene first (~200 MB) and check the",
            "loader before committing to the full split.",
        ),
    ),
    "zju_l5": Source(
        key="zju_l5",
        title="ZJU-L5 (real VL53L5CX dToF + RGB)",
        root=DATA_ROOT / "ZJUL5",
        auto=False,
        size="~2 GB",
        url="https://github.com/zju3dv/deltar",
        expect=("data.json",),
        steps=(
            "Released with Deltar (ECCV 2022): https://github.com/zju3dv/deltar",
            "  README -> 'Dataset' -> the ZJU-L5 download link.",
            "Unpack keeping the original layout so that",
            "  data/raw/ZJUL5/{data.json,theater/,lab1/,cafe1/,cafe2/} exists.",
            "This is the dataset the DEPTHOR / DEPTHOR++ headline numbers are on.",
        ),
    ),
    "hammer": Source(
        key="hammer",
        title="HAMMER (RGB + I-ToF + D-ToF + active stereo + laser GT)",
        root=DATA_ROOT / "hammer",
        auto=True,
        size="~50 GB",
        url="http://www.campar.in.tum.de/public_datasets/2022_arxiv_jung/_dataset_processed.zip",
        expect=(),
        notes=(
            "The archive's internal folder names are not documented upstream. After "
            "unpacking run this tool with --inspect and set the *_subdir keys in "
            "configs/dataset/hammer.yaml to match."
        ),
        steps=(
            "Direct download, no registration:",
            "  http://www.campar.in.tum.de/public_datasets/2022_arxiv_jung/"
            "_dataset_processed.zip",
            "Unzip into data/raw/hammer/ then run:",
            "  python scripts/download_data.py --dataset hammer --inspect",
        ),
    ),
    "rgbdd": Source(
        key="rgbdd",
        title="RGB-D-D",
        root=DATA_ROOT / "RGBDD",
        auto=False,
        size="~15 GB",
        url="https://github.com/lingzhi96/RGB-D-D-Dataset",
        expect=("Test/RGBDD_RGB", "Test/RGBDD_GT"),
        steps=(
            "Access is by request: fill in the form / email the authors as described at",
            "  https://github.com/lingzhi96/RGB-D-D-Dataset",
            "Unpack so that data/raw/RGBDD/{Train,Test}/{RGBDD_RGB,RGBDD_GT,RGBDD_LR}/ exists.",
        ),
    ),
    "tofdc": Source(
        key="tofdc",
        title="TOFDC / TOFDSR",
        root=DATA_ROOT / "TOFDC",
        auto=False,
        size="~10 GB",
        url="https://yanzq95.github.io/projectpage/TOFDC/index.html",
        expect=(),
        steps=(
            "Download from the project page above.",
            "The split lists DuCos used are already vendored:",
            "  third_party/ducos/data/TOFDC_Filled_{Train,Test}.txt",
        ),
    ),
    "hypersim": Source(
        key="hypersim",
        title="Hypersim",
        root=DATA_ROOT / "hypersim",
        auto=True,
        size="~250 GB full; the DEPTHOR training subset is much smaller",
        url="https://github.com/apple/ml-hypersim",
        expect=("scenes",),
        steps=(
            "  git clone https://github.com/apple/ml-hypersim /tmp/ml-hypersim",
            "  python /tmp/ml-hypersim/code/python/tools/dataset_download_images.py \\",
            "      --downloads_dir data/raw/hypersim",
            "Restrict to the scenes listed in third_party/depthor/assets/hypersim_train.txt",
            "so our training split matches DEPTHOR's.",
        ),
    ),
    "tartanair": Source(
        key="tartanair",
        title="TartanAir",
        root=DATA_ROOT / "tartanair",
        auto=False,
        size="varies by environment",
        url="https://theairlab.org/tartanair-dataset/",
        expect=(),
        steps=(
            "Use the official azcopy-based downloader linked from the project page.",
            "Start with the stair-bearing indoor environments (office, hospital).",
        ),
    ),
    "stair_dataset": Source(
        key="stair_dataset",
        title="RGB-D Stair Dataset",
        root=DATA_ROOT / "rgbd_stair",
        auto=False,
        size="?",
        url="",
        expect=(),
        notes="URL and licence still to be confirmed - see docs/datasets.md.",
        steps=(
            "TODO(team): confirm the canonical download URL and licence, then fill in",
            "this entry and docs/datasets.md.",
        ),
    ),
    "astra2_custom": Source(
        key="astra2_custom",
        title="Our Orbbec Astra 2 stair captures",
        root=DATA_ROOT / "astra2",
        auto=False,
        size="internal",
        url="internal",
        expect=(),
        steps=(
            "Not a public dataset. Sync it from the team storage (see the SOW channel),",
            "or record new sessions with the capture script.",
            "Layout: data/raw/astra2/<session>/{rgb,depth,gt}/<frame>.png + meta.json",
        ),
    ),
}

WEIGHTS: dict[str, Source] = {
    "depth_anything_v2_vits": Source(
        key="depth_anything_v2_vits",
        title="Depth Anything V2 (ViT-S) - required by BOTH baselines",
        root=CKPT_ROOT,
        auto=True,
        size="~100 MB",
        url=(
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
            "depth_anything_v2_vits.pth"
        ),
        expect=("depth_anything_v2_vits.pth",),
    ),
    "ducos": Source(
        key="ducos",
        title="DuCos pretrained checkpoints",
        root=CKPT_ROOT / "ducos",
        auto=False,
        size="~500 MB",
        url="https://huggingface.co/RaynWu2002/DuCos/tree/main",
        expect=("x4.pth.tar",),
        notes="Verified 2026-09-01: the files live under DuCos/ckpts/ inside the HF repo.",
        steps=(
            "  pip install huggingface_hub",
            "  python - <<'EOF'",
            "  from huggingface_hub import hf_hub_download; import shutil, pathlib",
            "  for f in ['x4.pth.tar', 'RealRGBDD.pth.tar', 'RealTOFDSR.pth.tar']:",
            "      p = hf_hub_download('RaynWu2002/DuCos', f'DuCos/ckpts/{f}')",
            "      d = pathlib.Path('checkpoints/ducos'); d.mkdir(parents=True, exist_ok=True)",
            "      shutil.copy(p, d / f)",
            "  EOF",
            "Available: x{1.5,2,2.7,3.4,4,5.3,8,11.6,16}.pth.tar, Compress.pth.tar,",
            "           RealRGBDD[Noisy].pth.tar, RealTOFDSR[Noisy].pth.tar  (~215 MB each)",
        ),
    ),
    "depthor": Source(
        key="depthor",
        title="DEPTHOR (v1) pretrained checkpoints",
        root=CKPT_ROOT / "depthor",
        auto=False,
        size="~1 GB",
        url="https://github.com/ShadowBbBb/Depthor",
        expect=(),
        notes="Google Drive links, so no scriptable download.",
        steps=(
            "Open third_party/depthor/README.md and follow the two Google Drive links",
            "(Depthor-ZJU-Large / Depthor-ZJU-Small).",
            "Save them as checkpoints/depthor/depthor_zju_{large,small}.pt and point",
            "configs/model/depthor.yaml :: checkpoint at the file.",
            "NOTE: DEPTHOR++ weights do NOT exist publicly - see docs/metrics_protocol.md.",
        ),
    ),
}


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}\n        -> {dest}")

    def hook(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, 100.0 * block * block_size / total)
            print(f"\r  {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.replace(dest)
    return dest


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive.name} -> {dest}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.suffixes[-2:] in ([".tar", ".gz"], [".tar", ".xz"]) or archive.suffix == ".tar":
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        print(f"  (not an archive, left as is: {archive})")


def cmd_list() -> int:
    print(f"{'key':<16} {'auto':<6} {'present':<9} {'size':<28} title")
    print("-" * 100)
    for src in list(DATASETS.values()) + list(WEIGHTS.values()):
        present = "yes" if verify(src, quiet=True) else "no"
        print(f"{src.key:<16} {src.auto!s:<6} {present:<9} {src.size:<28} {src.title}")
    print("\ndata roots are git-ignored; splits in data/splits/ are the only tracked data.")
    return 0


def verify(src: Source, quiet: bool = False) -> bool:
    ok = src.root.exists() and all((src.root / e).exists() for e in src.expect)
    if not quiet:
        mark = "OK " if ok else "MISSING"
        print(f"[{mark}] {src.key}: {src.root}")
        for e in src.expect:
            print(f"        {'+' if (src.root / e).exists() else '-'} {e}")
    return ok


def inspect(src: Source, depth: int = 3, limit: int = 40) -> int:
    if not src.root.exists():
        print(f"{src.root} does not exist")
        return 1
    print(f"tree of {src.root} (depth {depth}):")
    shown = 0
    for path in sorted(src.root.rglob("*")):
        rel = path.relative_to(src.root)
        if len(rel.parts) > depth:
            continue
        print(f"  {'  ' * (len(rel.parts) - 1)}{rel.name}{'/' if path.is_dir() else ''}")
        shown += 1
        if shown >= limit:
            print(f"  ... ({limit} entries shown)")
            break
    return 0


def instructions(src: Source) -> None:
    print(f"\n=== {src.title} ({src.key}) ===")
    print(f"target directory : {src.root}")
    print(f"approximate size : {src.size}")
    if src.url:
        print(f"source           : {src.url}")
    if src.notes:
        print(f"note             : {src.notes}")
    if src.steps:
        print("steps:")
        for s in src.steps:
            print(f"  {s}")
    print("\nafter unpacking, check it with:")
    print(f"  python scripts/download_data.py --dataset {src.key} --verify")


def fetch(src: Source) -> int:
    if not src.auto or not src.url or src.url.startswith("http") is False:
        instructions(src)
        return 1
    if src.url.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".pth", ".pt", ".pth.tar")):
        dest = src.root / Path(src.url).name
        if dest.exists():
            print(f"already downloaded: {dest}")
        else:
            _download(src.url, dest)
        if dest.suffix in (".zip", ".tar", ".gz", ".tgz"):
            _extract(dest, src.root)
        return 0
    instructions(src)  # auto but not a single-file URL -> it is a scripted procedure
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", help=f"one of {sorted(DATASETS)}")
    p.add_argument("--weights", help=f"one of {sorted(WEIGHTS)}")
    p.add_argument("--list", action="store_true")
    p.add_argument("--verify", action="store_true", help="check what is on disk")
    p.add_argument("--inspect", action="store_true", help="print the directory tree")
    args = p.parse_args(argv)

    if args.list or not (args.dataset or args.weights or args.verify):
        return cmd_list()

    if args.verify and not (args.dataset or args.weights):
        missing = [s for s in list(DATASETS.values()) + list(WEIGHTS.values()) if not verify(s)]
        print(f"\n{len(missing)} of {len(DATASETS) + len(WEIGHTS)} sources missing")
        return 0

    src = DATASETS.get(args.dataset) if args.dataset else WEIGHTS.get(args.weights)
    if src is None:
        p.error(f"unknown key; datasets={sorted(DATASETS)} weights={sorted(WEIGHTS)}")

    if args.inspect:
        return inspect(src)
    if args.verify:
        return 0 if verify(src) else 1
    return fetch(src)


if __name__ == "__main__":
    sys.exit(main())
