#!/usr/bin/env python3
"""Download every baseline's weights into ``checkpoints/`` where the configs expect them.

    uv run --group baselines python scripts/download_weights.py --list
    uv run --group baselines python scripts/download_weights.py              # everything
    uv run --group baselines python scripts/download_weights.py promptda omni_dc
    uv run --group baselines python scripts/download_weights.py --force dornet

Files that already exist are skipped.  Hugging Face downloads need no login;
the Google Drive ones (DEPTHOR, WAVE) go through ``gdown`` and can hit Drive's
download quota -- rerun later, finished files are kept.  DORNet needs nothing:
its weights ship inside ``third_party/dornet/checkpoints/``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import _bootstrap

CKPT = _bootstrap.REPO_ROOT / "checkpoints"

DAV2 = "depth-anything/Depth-Anything-V2-{}"


def hf(repo: str, filename: str, dest: Path) -> None:
    """One Hub file -> ``dest`` (flat; ``hf_hub_download`` would keep the repo's subdirs)."""
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / ".hf_tmp"
    src = Path(hf_hub_download(repo, filename, local_dir=tmp))
    shutil.move(src, dest)
    shutil.rmtree(tmp, ignore_errors=True)


def gdrive_file(file_id: str, dest: Path) -> None:
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not gdown.download(id=file_id, output=str(dest), quiet=False):
        raise RuntimeError(f"gdown failed for {file_id} (Drive quota? retry later)")


# --------------------------------------------------------------------------
# one entry per baseline: (files it provides, download function)
# --------------------------------------------------------------------------


def ducos() -> None:
    for name in ("x4", "RealRGBDD", "RealTOFDSR"):
        dest = CKPT / "ducos" / f"{name}.pth.tar"
        if not dest.exists() or FORCE:
            hf("RaynWu2002/DuCos", f"DuCos/ckpts/{name}.pth.tar", dest)


def depthor() -> None:
    # Drive ids from third_party/depthor/README.md
    for size, fid in (
        ("large", "1oZByVUklbjQHlZTdKFQdwkMKkugi4l6-"),
        ("small", "1P5I54TZctkilvRMTAzMJHzZs9OdN1fBp"),
    ):
        dest = CKPT / "depthor" / f"depthor_zju_{size}.pt"
        if not dest.exists() or FORCE:
            gdrive_file(fid, dest)


def wave() -> None:
    import gdown

    # upstream file -> the name configs/model/wave.yaml expects
    wanted = {
        "HYPERSIM_Patch_448_Scale_8": "NYU8x.pth",
        "NYUV2_Patch_448_Scale_16": "NYU16x.pth",
        "NYUV2_Patch_448_Scale_32": "NYU32x.pth",
    }
    out = CKPT / "wave"
    if all((out / n).exists() for n in wanted.values()) and not FORCE:
        return
    up = CKPT / "wave_upstream"
    gdown.download_folder(
        "https://drive.google.com/drive/folders/1uY5uzU8AAKafeoN_hbuC3bxVN3WJXhin",
        output=str(up),
        quiet=False,
    )
    out.mkdir(parents=True, exist_ok=True)
    files = [p for p in up.rglob("*") if p.is_file()]
    for key, name in wanted.items():
        hit = [p for p in files if key in p.name]
        if len(hit) != 1:
            raise RuntimeError(f"wave: expected one file matching {key!r} in {up}, got {hit}")
        shutil.copy(hit[0], out / name)


def dornet() -> None:
    src = _bootstrap.REPO_ROOT / "third_party" / "dornet" / "checkpoints" / "NYU_X4.pth"
    if not src.exists():
        raise RuntimeError("third_party/dornet is empty: git submodule update --init --recursive")
    print("dornet: weights ship with the submodule, nothing to download")


def promptda() -> None:
    dest = CKPT / "promptda" / "promptda_vitl.ckpt"
    if not dest.exists() or FORCE:
        hf("depth-anything/prompt-depth-anything-vitl", "model.ckpt", dest)


def prior_da() -> None:
    d = CKPT / "prior_depth_anything"
    for name in (
        "prior_depth_anything_vitb_1_1.pth",
        "depth_anything_v2_vitb.pth",
    ):  # names fixed by upstream
        if not (d / name).exists() or FORCE:
            hf("Rain729/Prior-Depth-Anything", name, d / name)


def omni_dc() -> None:
    """HF mirror (a directory the adapter loads with ``from_pretrained``) + DAv2 vitl."""
    from huggingface_hub import snapshot_download

    d = CKPT / "omni_dc"
    if not (d / "model.safetensors").exists() or FORCE:
        snapshot_download("zuoym15/OMNI-DC", local_dir=d)
    dav2("Large", "vitl")


def dav2(size: str, enc: str) -> None:
    dest = CKPT / f"depth_anything_v2_{enc}.pth"
    if not dest.exists() or FORCE:
        hf(DAV2.format(size), f"depth_anything_v2_{enc}.pth", dest)


def dav2_backbones() -> None:
    """DuCos / DEPTHOR / local_bilateral use vits; OMNI-DC (via omni_dc) uses vitl."""
    dav2("Small", "vits")


TASKS: dict[str, Callable[[], None]] = {
    "dav2": dav2_backbones,
    "ducos": ducos,
    "depthor": depthor,
    "wave": wave,
    "dornet": dornet,
    "promptda": promptda,
    "prior_da": prior_da,
    "omni_dc": omni_dc,
}
FORCE = False


def main(argv: list[str] | None = None) -> int:
    global FORCE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"subset of: {' '.join(TASKS)} (default: all)")
    ap.add_argument("--list", action="store_true", help="print the names and exit")
    ap.add_argument("--force", action="store_true", help="download even if the file exists")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(TASKS))
        return 0
    unknown = [n for n in args.names if n not in TASKS]
    if unknown:
        ap.error(f"unknown: {unknown}; choose from {list(TASKS)}")
    FORCE = args.force

    failed = []
    for name in args.names or TASKS:
        print(f"== {name}")
        try:
            TASKS[name]()
        except Exception as e:  # one failed Drive quota must not lose the other downloads
            print(f"   FAILED: {e}", file=sys.stderr)
            failed.append(name)
    if failed:
        print(f"failed: {' '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
