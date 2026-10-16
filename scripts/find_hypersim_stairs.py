#!/usr/bin/env python3
"""Find staircase frames in Hypersim: exact, noise-free GT on realistic interiors.

Hypersim has no stair label, so frames are ranked by zero-shot CLIP ("a photo of a staircase"
against a list of ordinary rooms), the best few per camera trajectory are kept, and contact sheets
are written so a human can accept or reject by eye.  Three steps, each resumable:

    # 1. score every 4th frame (needs `transformers`; ~5-10 min on a GPU for the full set)
    uv run --group baselines --with transformers python scripts/find_hypersim_stairs.py score --stride 4
    # 2. candidates + contact sheets to look at (outputs/hypersim_stairs/sheet_*.png)
    uv run --group baselines python scripts/find_hypersim_stairs.py sheets --top 120
    # 3. write the ranks you accepted as a split the hypersim loader understands
    uv run --group baselines python scripts/find_hypersim_stairs.py split --keep 1 2 5-9 14

The split lands in ``data/splits/hypersim/stairs.json``; use it with
``dataset.split_file=hypersim/stairs.json`` (frame_stride must stay 1 or the ids will not exist).
CLIP only proposes: it also likes balconies and ladders, and misses stairs seen from far away.
"""

from __future__ import annotations

import argparse
import csv
import json
import re

import _bootstrap

from data.degradation import build_degradation
from data.loaders import build_dataset
from utils.config import load_config

OUT = _bootstrap.REPO_ROOT / "outputs" / "hypersim_stairs"
SPLIT = _bootstrap.REPO_ROOT / "data" / "splits" / "hypersim" / "stairs.json"
MODEL = "openai/clip-vit-base-patch16"
STAIRS = [
    "a photo of a staircase",
    "a photo of stairs with steps",
    "a flight of steps indoors",
    "a stairway with a handrail",
]
ROOMS = [
    "a photo of a living room",
    "a photo of a bedroom",
    "a photo of a kitchen",
    "a photo of a bathroom",
    "a photo of an office",
    "a photo of a hallway",
    "a photo of a dining room",
    "a photo of a room with furniture",
    "a photo of a lobby",
    "a photo of a closet",
    "a photo of a empty room",
]


def records(stride: int) -> list[dict]:
    cfg = dict(load_config("dataset", "hypersim"))
    cfg["frame_stride"] = stride
    cfg["strict"] = False
    ds = build_dataset(cfg, degradation=build_degradation({"type": "identity"}))
    return ds.records


def split_id(rid: str) -> tuple[str, str, int]:
    scene, cam, frame = rid.split("/")
    return scene, cam, int(frame)


def cmd_score(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    recs = records(args.stride)
    if not recs:
        raise SystemExit("no Hypersim frames found; is STAIR_DATA_ROOT set?")
    print(f"[score] {len(recs)} frames (stride {args.stride})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    texts = STAIRS + ROOMS
    # the plain forward pass: logits_per_image is scaled cosine similarity, and unlike
    # get_text_features/get_image_features its output type is the same in every transformers version
    t = proc(text=texts, return_tensors="pt", padding=True).to(dev)

    class Imgs(torch.utils.data.Dataset):
        def __len__(self):
            return len(recs)

        def __getitem__(self, i):
            im = Image.open(recs[i]["rgb"]).convert("RGB")
            return proc(images=im, return_tensors="pt")["pixel_values"][0]

    dl = torch.utils.data.DataLoader(Imgs(), batch_size=128, num_workers=args.workers)
    rows, i = [], 0
    with torch.no_grad():
        for batch in dl:
            logits = model(**t, pixel_values=batch.to(dev)).logits_per_image
            p = logits.softmax(-1).cpu().numpy()
            for row in p:
                r = recs[i]
                rows.append(
                    (
                        r["id"],
                        float(row[: len(STAIRS)].sum()),
                        STAIRS[int(row[: len(STAIRS)].argmax())],
                    )
                )
                i += 1
            if i % (128 * 20) == 0:
                print(f"  {i}/{len(recs)}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "stairs_prob", "best_prompt"])
        w.writerows(rows)
    print(f"[score] {OUT / 'scores.csv'}")


def cmd_sheets(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    with (OUT / "scores.csv").open() as fh:
        scored = [(r["id"], float(r["stairs_prob"])) for r in csv.DictReader(fh)]
    # best frames first; per trajectory keep at most --per-traj, at least --gap frames apart
    kept, taken = [], {}
    for rid, p in sorted(scored, key=lambda x: -x[1]):
        scene, cam, frame = split_id(rid)
        mine = taken.setdefault((scene, cam), [])
        if len(mine) >= args.per_traj or any(abs(frame - f) < args.gap for f in mine):
            continue
        mine.append(frame)
        kept.append((rid, p))
        if len(kept) >= args.top:
            break

    recs = {r["id"]: r for r in records(1)}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "candidates.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "id", "stairs_prob"])
        for k, (rid, p) in enumerate(kept, 1):
            w.writerow([k, rid, f"{p:.3f}"])

    per_sheet, cols = 30, 6
    for s in range(0, len(kept), per_sheet):
        chunk = kept[s : s + per_sheet]
        rows_n = -(-len(chunk) // cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 2.5 * rows_n), squeeze=False)
        for a in axes.ravel():
            a.axis("off")
        for j, (rid, p) in enumerate(chunk):
            im = Image.open(recs[rid]["rgb"]).convert("RGB")
            im.thumbnail((320, 240))
            a = axes[j // cols, j % cols]
            a.imshow(im)
            a.set_title(f"#{s + j + 1}  {p:.2f}  {rid.split('/')[0]}", fontsize=7)
        fig.tight_layout()
        path = OUT / f"sheet_{s // per_sheet + 1:02d}.png"
        fig.savefig(path, dpi=70)
        plt.close(fig)
        print(f"[sheets] {path}")
    print(f"[sheets] {len(kept)} candidates -> {OUT / 'candidates.csv'}")


def parse_keep(tokens: list[str]) -> list[int]:
    out: list[int] = []
    for t in tokens:
        m = re.fullmatch(r"(\d+)-(\d+)", t)
        out += list(range(int(m[1]), int(m[2]) + 1)) if m else [int(t)]
    return out


def cmd_split(args: argparse.Namespace) -> None:
    with (OUT / "candidates.csv").open() as fh:
        by_rank = {int(r["rank"]): r["id"] for r in csv.DictReader(fh)}
    keep = parse_keep(args.keep)
    missing = [k for k in keep if k not in by_rank]
    if missing:
        raise SystemExit(f"no such rank(s): {missing}; candidates have 1..{max(by_rank)}")
    ids = sorted(by_rank[k] for k in keep)
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps({"dataset": "hypersim", "split": "stairs", "ids": ids}, indent=1))
    scenes = {i.split("/")[0] for i in ids}
    print(f"[split] {len(ids)} frames from {len(scenes)} scenes -> {SPLIT}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="CLIP-score frames")
    s.add_argument("--stride", type=int, default=4)
    s.add_argument("--workers", type=int, default=8)
    c = sub.add_parser("sheets", help="candidates + contact sheets")
    c.add_argument("--top", type=int, default=120)
    c.add_argument("--per-traj", type=int, default=2)
    c.add_argument("--gap", type=int, default=15, help="min frame distance within a trajectory")
    p = sub.add_parser("split", help="write the accepted ranks as a split file")
    p.add_argument("--keep", nargs="+", required=True, help="ranks, e.g. 1 2 5-9 14")
    args = ap.parse_args()
    {"score": cmd_score, "sheets": cmd_sheets, "split": cmd_split}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
