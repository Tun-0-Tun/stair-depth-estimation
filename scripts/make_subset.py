#!/usr/bin/env python3
"""Make a small copy of a dataset that the existing loaders read unchanged.

The copy keeps the original directory layout and file names but holds only
``-n`` samples per split, so it runs on a laptop and in quick checks:

    uv run python scripts/make_subset.py --dataset zju_l5 --output_dir /tmp/zju_small -n 5

    # then point any script at it
    ... dataset=zju_l5 dataset.root=/tmp/zju_small \
        dataset.split_file=/tmp/zju_small/splits/test.json

How files are found
-------------------
Nothing here knows a dataset's layout.  Building a loader runs its
``_build_index()``, which lists every sample as a record of file paths; the
script copies those paths relative to ``ds.root``.  What the loader indexed is
exactly what it will look for in the copy.  Packed datasets (NYUv2 ``mat`` /
``npy``: one file holds all samples) are the exception -- a smaller file with
the same name and keys is written instead.

Train / test split
------------------
The authors' split is kept wherever one exists (docs/datasets.md, "Split
protocol"), and written back in the dataset's own format:

* ``zju_l5``      ``data.json`` train/test         -> filtered ``data.json``
* ``hammer``      scenes 12-14 are test            -> scene folder names kept
* ``nyuv2`` h5    ``nyudepthv2/{train,val}/``      -> folders kept
* ``void_stairs`` ``train_*.txt`` / ``test_*.txt`` -> the same lists, filtered
* ``minjiang``    none; time blocks per camera, first 80% train, last 20% test
  (never a random per-frame split: neighbouring frames are near-duplicates)
* ``middlebury``, ``lu``, ``nyuv2`` mat/npy: test sets, no train part

Every copy also gets ``splits/{train,test}.json`` in the ``data/splits`` format,
which works through ``split_file=`` for every loader, including the ones that
ignore ``split=`` (void_stairs, minjiang).  A split with no samples gets no file.
``manifest.json`` records the source, seed and ids.

Before exiting, the script reopens the copy with the same loader, checks the
ids, and loads one sample per split through ``validate_sample``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

SPLITS = ("train", "test")
#: loaders whose own ``split=`` already selects the authors' train/test lists
NATIVE_SPLIT = {"zju_l5", "hammer"}
#: test-only benchmark sets: everything is test, there is no train part
TEST_ONLY = {"middlebury", "lu"}
#: loaders whose samples have sibling streams worth keeping (void validity_map,
#: absolute_pose; hammer depth_l515, depth_tof, pol, ...), so input_subdir
#: overrides keep working on the copy. value = the record key of the main file
SIBLING_STREAMS = {"void_stairs": "image", "hammer": "rgb"}
#: minjiang time-block split
MINJIANG_TRAIN_FRACTION = 0.8
MANIFEST = "manifest.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset config in configs/dataset/")
    p.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        required=True,
        type=Path,
        help="root of the new, small copy",
    )
    p.add_argument("-n", type=int, default=5, help="samples per split (default: 5)")
    p.add_argument("--seed", type=int, default=0, help="seed for picking samples (default: 0)")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace output_dir if it holds an earlier subset (has manifest.json)",
    )
    args = p.parse_args(argv)
    if args.n < 1:
        p.error("-n must be >= 1")
    return args


# ---------------------------------------------------------------- loaders


def _degradation():
    """GT-only loaders refuse to build without one; only the index is used here."""
    from data.degradation import build_degradation
    from utils.config import load_config

    return build_degradation(dict(load_config("degradation", "astra2_uncalibrated")))


def _build(cfg: Mapping[str, Any], **overrides: Any):
    from data.loaders import build_dataset

    return build_dataset(dict(cfg), degradation=_degradation(), **overrides)


def _build_or_none(cfg: Mapping[str, Any], split: str):
    """A split can legitimately be absent (e.g. no HAMMER test scenes on disk)."""
    try:
        return _build(cfg, split=split)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"note: no {split} samples: {str(exc).splitlines()[0]}", file=sys.stderr)
        return None


def loader_key(cfg: Mapping[str, Any]) -> str:
    return str(cfg.get("loader", cfg.get("name")))


def is_packed(cfg: Mapping[str, Any]) -> bool:
    return loader_key(cfg) == "nyuv2" and cfg.get("layout", "h5") in ("mat", "npy")


# ------------------------------------------------------------------ split


def split_records(cfg: Mapping[str, Any]) -> tuple[Any, dict[str, list], str]:
    """Return (source dataset, {split: records}, where the split came from)."""
    key = loader_key(cfg)
    if key == "synthetic_stairs":
        raise SystemExit("synthetic_stairs is generated on the fly; there are no files to copy")

    if key in NATIVE_SPLIT or (key == "nyuv2" and not is_packed(cfg)):
        dss = {s: _build_or_none(cfg, s) for s in SPLITS}
        present = [d for d in dss.values() if d is not None]
        if not present:
            raise SystemExit(f"{key}: no samples in any split")
        rid = present[0].record_id
        out = {s: list(d.records) if d is not None else [] for s, d in dss.items()}
        # zju_l5 falls back to "test" when data.json has no "train": that is
        # not a train split, and one frame must never sit in both
        test_ids = {rid(r) for r in out["test"]}
        out["train"] = [r for r in out["train"] if rid(r) not in test_ids]
        return present[0], out, "authors' split, read by the loader itself"

    ds = _build(cfg)
    if key == "void_stairs":
        return ds, _void_split(ds), "authors' train_image.txt / test_image.txt"
    if key == "minjiang":
        return (
            ds,
            _time_block_split(ds.records),
            (
                f"no authors' split: time blocks per camera, first {MINJIANG_TRAIN_FRACTION:.0%} train"
            ),
        )
    if key in TEST_ONLY or is_packed(cfg):
        return ds, {"train": [], "test": list(ds.records)}, "test-only set, no train part"
    raise SystemExit(f"no split rule for loader {key!r}; add one to scripts/make_subset.py")


def _read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _void_line_index(ds: Any, split: str) -> dict[str, int]:
    """Map record id -> line number in the upstream ``<split>_image.txt``.

    Lines hold image paths with a prefix that differs between releases
    (``void_1500/data/...`` or ``data/...``), so match on the path suffix.
    """
    path = ds.root / f"{split}_image.txt"
    if not path.exists():
        raise SystemExit(f"void_stairs: {path} is missing, so the authors' split is unknown")
    by_rel = {r["image"].relative_to(ds.root).as_posix(): ds.record_id(r) for r in ds.records}
    found: dict[str, int] = {}
    for i, line in enumerate(_read_list(path)):
        parts = Path(line).as_posix().split("/")
        # try the shortest suffix that can be a relative image path first
        for k in range(len(parts)):
            rel = "/".join(parts[k:])
            if rel in by_rel:
                found[by_rel[rel]] = i
                break
    return found


def _void_split(ds: Any) -> dict[str, list]:
    out = {}
    for split in SPLITS:
        lines = _void_line_index(ds, split)
        out[split] = [r for r in ds.records if ds.record_id(r) in lines]
    if not out["train"] and not out["test"]:
        raise SystemExit(
            "void_stairs: no indexed frame appears in train_image.txt or test_image.txt; "
            "check the path format in those files"
        )
    return out


def _time_block_split(records: Sequence[Any]) -> dict[str, list]:
    groups: dict[tuple[str, str], list] = {}
    for r in records:  # the loader returns them sorted by timestamp within a camera
        groups.setdefault((r["scene"], r["camera"]), []).append(r)
    out: dict[str, list] = {"train": [], "test": []}
    for recs in groups.values():
        cut = round(len(recs) * MINJIANG_TRAIN_FRACTION)
        out["train"] += recs[:cut]
        out["test"] += recs[cut:]
    return out


def pick(records: Sequence[Any], n: int, rng: np.random.Generator, split: str) -> list:
    if len(records) <= n:
        if records:
            print(f"note: {split} has only {len(records)} samples, taking all", file=sys.stderr)
        return list(records)
    idx = sorted(rng.choice(len(records), n, replace=False))
    return [records[i] for i in idx]


# ------------------------------------------------------------------- copy


def _files_of(record: Mapping[str, Any], key: str) -> list[Path]:
    files = [v for v in record.values() if isinstance(v, Path) and v.is_file()]
    main = record.get(SIBLING_STREAMS.get(key, ""))
    if isinstance(main, Path):
        # <seq>/image/<stem>.png -> <seq>/*/<stem>.*
        for stream in main.parent.parent.iterdir():
            if stream.is_dir():
                files += [f for f in stream.glob(f"{main.stem}.*") if f.is_file()]
    return files


def copy_files(ds: Any, chosen: Sequence[Any], out: Path, key: str) -> int:
    done: set[Path] = set()
    for record in chosen:
        for src in _files_of(record, key):
            if src in done:
                continue
            try:
                rel = src.relative_to(ds.root)
            except ValueError as exc:
                raise SystemExit(f"{src} lies outside the dataset root {ds.root}") from exc
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            done.add(src)
    return len(done)


def write_zju_index(ds: Any, chosen: dict[str, list], out: Path) -> None:
    """Keep ``data.json`` as is, minus the entries of frames not copied."""
    payload = json.loads((ds.root / ds.index_file).read_text(encoding="utf-8"))
    keep = {ds.record_id(r) for recs in chosen.values() for r in recs}
    for k, v in payload.items():
        if isinstance(v, list) and all(isinstance(e, dict) and "filename" in e for e in v):
            payload[k] = [e for e in v if e["filename"] in keep]
    (out / ds.index_file).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def write_void_lists(ds: Any, chosen: dict[str, list], out: Path) -> None:
    """Filter every ``<split>_*.txt``; they are parallel lists, one line per frame."""
    for split in SPLITS:
        lines = _void_line_index(ds, split)
        keep = sorted(lines[ds.record_id(r)] for r in chosen[split])
        n_lines = len(_read_list(ds.root / f"{split}_image.txt"))
        for path in sorted(ds.root.glob(f"{split}_*.txt")):
            rows = _read_list(path)
            if len(rows) != n_lines:
                print(
                    f"note: {path.name} is not parallel to {split}_image.txt, skipped",
                    file=sys.stderr,
                )
                continue
            (out / path.name).write_text("".join(rows[i] + "\n" for i in keep), encoding="utf-8")


def write_packed_nyu(ds: Any, chosen: Sequence[Any], out: Path) -> None:
    idx = sorted(int(r["index"]) for r in chosen)
    if ds.layout == "mat":
        import h5py

        with h5py.File(ds.root / ds.mat_file, "r") as src, h5py.File(out / ds.mat_file, "w") as dst:
            n = src["depths"].shape[0]
            for name, obj in src.items():
                # per-sample numeric arrays only; object references (names,
                # scenes) point into the source file and cannot be carried over
                if (
                    isinstance(obj, h5py.Dataset)
                    and obj.shape[:1] == (n,)
                    and obj.dtype.kind != "O"
                ):
                    dst[name] = obj[idx]
        return
    for name in ("test_images_v2.npy", "test_depth.npy"):
        np.save(out / name, np.load(ds.root / name)[idx])
    if ds.minmax_file and (ds.root / ds.minmax_file).exists():
        np.save(out / ds.minmax_file, np.load(ds.root / ds.minmax_file)[:, idx])


# ------------------------------------------------------------ check + main


def verify_and_write_splits(
    cfg: Mapping[str, Any], out: Path, chosen: dict[str, list], src_ds: Any
) -> dict[str, list[str]]:
    """Reopen the copy with the same loader and freeze its split files."""
    from data.loaders import validate_sample

    new_cfg = {**cfg, "root": str(out.resolve())}
    ids: dict[str, list[str]] = {}
    for split in SPLITS:
        if not chosen[split]:
            continue
        ds_new = _build(new_cfg, split=split)
        on_disk = [ds_new.record_id(r) for r in ds_new.records]
        if is_packed(cfg):
            want = on_disk  # packed ids are positions, renumbered in the smaller file
            if len(want) != len(chosen[split]):
                raise SystemExit(f"copy has {len(want)} samples, expected {len(chosen[split])}")
        else:
            want = [src_ds.record_id(r) for r in chosen[split]]
            missing = set(want) - set(on_disk)
            if missing:
                raise SystemExit(f"copy is missing {split} ids, e.g. {sorted(missing)[:3]}")
        split_path = out / "splits" / f"{split}.json"
        ds_new.write_split_file(split_path, want)
        check = _build(new_cfg, split=split, split_file=str(split_path))
        validate_sample(check[0], f"{loader_key(cfg)}/{split}")
        ids[split] = want
    return ids


def prepare_output(out: Path, src_root: Path, overwrite: bool) -> None:
    out_r, src_r = out.resolve(), src_root.resolve()
    if out_r == src_r or src_r in out_r.parents or out_r in src_r.parents:
        raise SystemExit(f"output_dir {out} overlaps the source dataset {src_root}")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise SystemExit(f"{out} is not empty; pass --overwrite to replace an earlier subset")
        if not (out / MANIFEST).exists():
            raise SystemExit(
                f"{out} has no {MANIFEST}, so it was not made by this script; "
                "refusing to delete it"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def make_subset(
    cfg: Mapping[str, Any], out: Path, n: int = 5, seed: int = 0, overwrite: bool = False
) -> dict[str, Any]:
    key = loader_key(cfg)
    src_ds, records, split_source = split_records(cfg)
    prepare_output(out, src_ds.root, overwrite)

    rng = np.random.default_rng(seed)
    chosen = {s: pick(records[s], n, rng, s) for s in SPLITS}
    if not any(chosen.values()):
        raise SystemExit(f"{key}: nothing to copy")

    if is_packed(cfg):
        write_packed_nyu(src_ds, chosen["test"] + chosen["train"], out)
        n_files = 1
    else:
        n_files = copy_files(src_ds, chosen["train"] + chosen["test"], out, key)
    if key == "zju_l5":
        write_zju_index(src_ds, chosen, out)
    if key == "void_stairs":
        write_void_lists(src_ds, chosen, out)

    ids = verify_and_write_splits(cfg, out, chosen, src_ds)
    manifest = {
        "dataset": key,
        "source_root": str(src_ds.root),
        "n_per_split": n,
        "seed": seed,
        "split_source": split_source,
        "counts": {s: len(ids.get(s, [])) for s in SPLITS},
        "ids": ids,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    manifest["n_files"] = n_files
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from utils.config import load_config

    cfg = dict(load_config("dataset", args.dataset))
    m = make_subset(cfg, args.output_dir, args.n, args.seed, args.overwrite)

    print(f"wrote {args.dataset} subset to {args.output_dir}  ({m['n_files']} files copied)")
    print(f"  split: {m['split_source']}")
    for s in SPLITS:
        print(f"  {s:5s}: {m['counts'][s]} samples")
    split = "test" if m["counts"]["test"] else "train"
    root = args.output_dir.resolve()
    print(
        f"use it with:\n  dataset={args.dataset} dataset.root={root} "
        f"dataset.split_file={root}/splits/{split}.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
