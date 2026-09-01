# Datasets

Every dataset we benchmark on, what it actually contains, how to get it, and
where it is used in the literature we are being compared against.

**Nothing in this table is committed to git.** Raw data lives under `data/raw/`
(git-ignored). The only data artefacts in the repo are the split files in
`data/splits/`, which are small and which must be identical for all three of us.

```bash
python scripts/download_data.py --list              # what is on disk
python scripts/download_data.py --dataset zju_l5    # fetch, or print manual steps
python scripts/download_data.py --verify            # check every root at once
```

---

## Overview

| Dataset | Role | Input modality | RGB size | Depth size | Depth source | GT source | Loader |
|---|---|---|---|---|---|---|---|
| [NYU Depth v2](#nyu-depth-v2) | benchmark, pre-training | GT only (simulate) | 640×480 | 640×480 | Kinect v1 (structured light) | Levin-colourisation filled Kinect | `nyuv2` ✅ |
| [ARKitScenes](#arkitscenes) | benchmark (real LR) | native LR | 1920×1440 | LR 256×192, HR 1920×1440 | Apple lidar (dToF) | high-res fused depth | `arkitscenes` ✅ |
| [ZJU-L5](#zju-l5) | **DEPTHOR headline benchmark** | native sparse | 640×480 | 8×8 zones | ST VL53L5CX dToF | stereo reconstruction | `zju_l5` ✅ |
| [HAMMER](#hammer) | **sensor-gap study** | native sparse | 1224×1024 (pol.) | per-sensor | D435 active stereo / L515 dToF / Helios I-ToF | laser-accurate | `hammer` ✅ |
| [RGB-D-D](#rgb-d-d) | **DuCos real benchmark** | native LR | 512×384 | LR 192×144, HR 512×384 | Huawei P30 Pro ToF | Lucid Helios | `rgbdd` ✅ |
| [Mirror3D-NYU](#mirror3d-nyu) | DEPTHOR++ hard case | GT only | 640×480 | 640×480 | Kinect v1 | manually corrected mirrors | ⬜ TODO |
| [Hypersim](#hypersim) | **DEPTHOR training set** | GT only (simulate) | 1024×768 | 1024×768 | rendered | perfect (ray distance) | `hypersim` ⬜ stub |
| [TartanAir](#tartanair) | pre-training, stair scenes | GT only (simulate) | 640×480 | 640×480 | rendered (AirSim) | perfect | `tartanair` ⬜ stub |
| [TOFDC / TOFDSR](#tofdc--tofdsr) | DuCos real benchmark | native LR | 512×384 | 512×384 | smartphone dToF | structured light | `tofdc` ⬜ stub |
| [RGB-D Stair Dataset](#rgb-d-stair-dataset) | **stair domain** | TBD | TBD | TBD | consumer RGB-D | TBD | `stair_dataset` ⬜ stub |
| [Astra 2 (ours)](#our-astra-2-captures) | **acceptance target** | native sparse | 640×480 | 640×480 | Orbbec Astra 2 (active stereo) | TBD — see below | `astra2_custom` ⬜ stub |

✅ implemented · ⬜ stub (contract + TODO list in the loader file)

Sizes marked TBD are ones we have not verified against the actual download yet;
fill them in when you first unpack the archive, and do not guess.

---

## Split protocol

Splits are frozen as JSON in `data/splits/<dataset>/<split>.json`:

```json
{"dataset": "zju_l5", "split": "test", "ids": ["theater/1645696174.476698.h5", "..."]}
```

Rules:

1. **Splits are committed.** They are lists of ids, a few KB each. If your
   numbers differ from a teammate's, the split is the first thing to rule out.
2. **Use the authors' split whenever one exists**, so our numbers stay
   comparable with the published tables. Concretely: ZJU-L5 uses `data.json`;
   Hypersim uses `third_party/depthor/assets/hypersim_{train,val}.txt`; TOFDC
   uses `third_party/ducos/data/TOFDC_Filled_{Train,Test}.txt`; HAMMER uses
   scenes 2–11 / 12–14.
3. **Never split by frame on sequential data.** Consecutive frames of one
   staircase are near-duplicates; a random frame split leaks the test set into
   training and inflates every metric. Split by scene, video, or capture
   session. This applies to ARKitScenes, HAMMER, TartanAir and our own captures.
4. Regenerate a split with `BaseDepthDataset.write_split_file(...)` and commit
   the result in the same PR as the change that motivated it.

---

## NYU Depth v2

* **Page:** <https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html>
* **Depth range used:** 0.001–10 m
* **Used by:** essentially every depth completion and DSR paper, including
  DEPTHOR (`configs/test_nyu.txt`) and DuCos (`test_Sync.py`).

Two incompatible preprocessed packs circulate, and the loader supports both:

| `dataset.layout` | Pack | Files | Depth units |
|---|---|---|---|
| `h5` (default) | completion (NLSPN / CompletionFormer / Deltar) | `nyudepthv2/{train,val}/<scene>/*.h5` with `rgb`, `depth`, sometimes `raw` | metres |
| `npy` | super-resolution (DKN / FDSR / DuCos) | `test_images_v2.npy`, `test_depth.npy`, `test_minmax.npy` | **normalised [0,1]**, denormalised by the loader with `test_minmax.npy` |

> ⚠️ The `npy` pack stores depth min-max-normalised per image. Without
> `test_minmax.npy` every metric is silently wrong by a per-image scale factor.
> The loader refuses to run rather than guess.

The `h5` pack's `raw` field, where present, is the *unfilled* Kinect depth — a
real sparse sensor input, which is strictly better than any simulation. The
loader uses it automatically.

**Get it:** `python scripts/download_data.py --dataset nyuv2` prints the links
(NLSPN README for the h5 pack, DKN README for the npy pack).

## ARKitScenes

* **Page:** <https://github.com/apple/ARKitScenes>
* **Depth range used:** 0.1–10 m · **Depth encoding:** uint16 millimetres
* **Why we use it:** the closest public analogue of our setup — a *real* consumer
  low-resolution depth sensor with a *real* high-resolution reference. No
  simulation is involved, so it is where a method's ability to fix genuine
  sensor error is measurable.

Layout (the `depth_upsampling` subset):

```
data/raw/arkitscenes/upsampling/{Training,Validation}/<video_id>/
    wide/<video_id>_<timestamp>.png            # RGB 1920x1440
    highres_depth/<video_id>_<timestamp>.png   # uint16 mm, GT
    lowres_depth/<video_id>_<timestamp>.png    # uint16 mm 256x192, input
    confidence/<video_id>_<timestamp>.png      # ARKit confidence 0/1/2
    wide_intrinsics/<video_id>_<timestamp>.pincam
```

The loader resizes to 480×640 by default (`dataset.target_size`) and can drop
low-confidence LR pixels with `dataset.min_confidence`.

**Get it:** scripted, via Apple's own downloader — see
`python scripts/download_data.py --dataset arkitscenes`. Fetch a single
`--video_id` first (~200 MB) before committing to the ~30 GB split.

## ZJU-L5

* **Page:** <https://github.com/zju3dv/deltar> (released with Deltar, ECCV 2022)
* **Depth range used:** 0.001–10 m
* **Why it matters:** this is the dataset DEPTHOR and DEPTHOR++ report their
  headline numbers on, so it is our primary cross-check against published
  results.

Layout, unchanged from the release (this is exactly what
`third_party/depthor` expects):

```
data/raw/ZJUL5/
    data.json                 # {"train": [...], "test": [{"filename": ...}]}
    theater/<timestamp>.h5    lab1/    cafe1/    cafe2/
```

Each `.h5`: `rgb` (480,640,3 uint8), `depth` (480,640 float32 m, stereo
reference), `hist_data` (64,2 — per-zone histogram mean and variance), `fr`
(64,4 — zone pixel rectangles), `mask` (64 — per-zone validity).

Our loader reproduces `dtof_to_sparse_depth` from the DEPTHOR repo exactly:
one pixel per valid zone, at the rectangle centre, carrying the zone mean
(`dataset.footprint: center`). `footprint: zone` writes the whole rectangle
instead — useful for methods expecting patch input, but **not** comparable with
DEPTHOR's published numbers.

**Get it:** manual — the Deltar README's dataset link.

## HAMMER

* **Page:** <https://github.com/Junggy/HAMMER-dataset> ·
  paper <https://arxiv.org/abs/2205.04565>
* **Depth range used:** 0.1–5 m · **Depth encoding:** uint16 millimetres
* **Direct download (no registration, ~50 GB):**
  <http://www.campar.in.tum.de/public_datasets/2022_arxiv_jung/_dataset_processed.zip>

**This is the most important dataset for our project after our own captures.**
It is the only public dataset carrying an active-stereo stream (RealSense D435
— the same sensing principle as the Astra 2), a dToF stream (L515), an I-ToF
stream (Lucid Helios) *and* laser-accurate ground truth for the same frames.
That makes it the one place where the "dToF method applied to active stereo"
domain gap can be measured without also changing the scene. DEPTHOR++ also
reports on it.

13 scenes, `scene02`–`scene14`; scenes 2–11 train, 12–14 test. Each scene has
two trajectories split into sequences, plus "naked" trajectories covering the
same viewpoints with the objects removed.

> ⚠️ **The archive's internal folder naming is not documented upstream.** The
> loader takes the subdirectory names as config (`rgb_subdir`, `gt_subdir`,
> `input_subdir` in `configs/dataset/hammer.yaml`) and defaults to
> `rgb` / `gt` / `<sensor>`. After unpacking:
>
> ```bash
> python scripts/download_data.py --dataset hammer --inspect
> ```
>
> then correct the config. The loader raises an error listing the directories it
> actually found rather than silently returning zero samples.

## RGB-D-D

* **Page:** <https://github.com/lingzhi96/RGB-D-D-Dataset>
* **Depth range used:** 0.1–10 m · **Depth encoding:** uint16 millimetres
* **Used by:** DuCos (`test_RealRGBDD.py`), FDSR, DKN, and most DSR papers.
* **Access:** by request — fill in the form / email the authors as described in
  the repo. There is no scriptable download.

```
data/raw/RGBDD/{Train,Test}/
    RGBDD_RGB/<name>_RGB.jpg           # 512x384
    RGBDD_GT/<name>_HR_gt.png          # uint16 mm 512x384, GT
    RGBDD_LR/<name>_LR_fill_depth.png  # uint16 mm 192x144, real ToF input
```

`dataset.downsample: real` uses the genuine low-resolution ToF frame — the
honest setting, and the one DuCos's "RealRGBDD" checkpoint is for.
`dataset.downsample: sync` ignores it and bicubically downsamples the GT
instead; that is the synthetic protocol most papers tabulate, kept only so we
can reproduce their numbers.

## Mirror3D-NYU

* **Page:** <https://github.com/3dlg-hcvc/mirror3d>
* **Role:** NYUv2 frames with manually corrected depth in mirror regions.
  DEPTHOR++ reports a 37% improvement in mirror regions on it. Mirrors and
  glass are a genuine failure mode for active stereo (the projected pattern
  reflects away), so this is directly relevant to the Astra 2 — a stairwell with
  a glass balustrade is exactly this problem.
* **Status:** ⬜ no loader yet. Implement as a variant of `nyuv2` that also
  returns the mirror mask so the metrics can be computed inside and outside
  mirror regions separately.

## Hypersim

* **Page:** <https://github.com/apple/ml-hypersim>
* **Depth range used:** 0.001–20 m
* **Role:** DEPTHOR's training set, and the cleanest source of perfect GT for
  fitting the Astra 2 degradation model.

> ⚠️ Hypersim stores **distance to the camera centre**, not planar Z. Convert
> with `z = dist / sqrt(1 + ((u-cx)/f)^2 + ((v-cy)/f)^2)` or every metric is
> systematically wrong towards the image corners. The stub loader's TODO list
> says so; do not skip it.

Use the frame lists in `third_party/depthor/assets/hypersim_{train,val}.txt` so
our training split matches DEPTHOR's, and commit them into
`data/splits/hypersim/`.

## TartanAir

* **Page:** <https://theairlab.org/tartanair-dataset/>
* **Depth range used:** 0.001–80 m
* **Role:** volume and viewpoint diversity for pre-training; several
  environments contain stairwells.

> ⚠️ Depth is `inf` for sky in outdoor scenes. Replace with 0 and let the mask
> handle it. Depth is already planar Z — do **not** apply the Hypersim
> conversion.

## TOFDC / TOFDSR

* **Page:** <https://yanzq95.github.io/projectpage/TOFDC/index.html>
* **Depth encoding:** uint16 millimetres
* **Role:** DuCos's second real-world benchmark (`test_RealTOFDSR.py`).
* The exact file lists DuCos used are already vendored in the repo:
  `third_party/ducos/data/TOFDC_Filled_{Train,Test}.txt`. Parse those rather
  than globbing, so our split is bit-identical to theirs.

## RGB-D Stair Dataset

* **Status:** ⬜ URL and licence still to be confirmed — see the TODO list in
  `data/loaders/stair_dataset.py`.
* **Role:** the only *public* stair-specific RGB-D data we have found. It is the
  bridge between the generic indoor benchmarks and our own captures.

Before writing the loader, answer two questions and record the answers here:

1. **Is the depth metric?** Several stair datasets ship 8-bit colourised depth
   visualisations, which are useless as ground truth. If so, the dataset can
   only serve as a qualitative RGB set and must not appear in a metric table.
2. **Are RGB and depth registered?** Verify on five frames by hand.

## Our Astra 2 captures

* **Sensor:** Orbbec Astra 2, active stereo, ~0.6–8 m, uint16 millimetre depth.
* **Status:** ⬜ capture in progress; loader stub in
  `data/loaders/astra2_custom.py`.
* **Not public.** Raw data stays in team storage and under `data/raw/astra2/`
  (git-ignored); only the split JSON goes into the repo.

Open decisions that must be settled and documented **here** before any number
from this dataset is reported:

| Decision | Why it matters |
|---|---|
| **Reference-depth source** — multi-view reconstruction, laser scan, or temporally averaged + hole-filled Astra 2 frames | The third option makes the sensor its own ground truth, which makes RMSE optimistic and δ1 meaningless. If we use it, every table must say so. `meta["gt_source"]` records it per sample. |
| **RGB/depth alignment** | Different lenses. Use the factory extrinsics from the Orbbec SDK; store per-session intrinsics in `meta` (the degradation model needs `focal_px` and `baseline_m`). |
| **Session-level splits** | Consecutive frames of one staircase are near-duplicates. |
| **Scene inventory** | How many distinct staircases, materials (metal/wood/glass/concrete), lighting conditions, up vs down. A benchmark on five staircases in one building measures memorisation. |

---

## Checkpoints

| Key | What | Size | How |
|---|---|---|---|
| `depth_anything_v2_vits` | Depth Anything V2 ViT-S — **required by both baselines** | ~100 MB | `python scripts/download_data.py --weights depth_anything_v2_vits` (scripted) |
| `ducos` | DuCos pretrained models | ~215 MB each | HuggingFace `RaynWu2002/DuCos`, files under `DuCos/ckpts/` |
| `depthor` | DEPTHOR v1 (`Depthor-ZJU-Large` / `-Small`) | ~1 GB | Google Drive links in `third_party/depthor/README.md` — manual |
| DEPTHOR++ | — | — | **Does not exist publicly.** See [metrics_protocol.md](metrics_protocol.md#baseline-availability). |
