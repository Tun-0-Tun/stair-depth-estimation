# Datasets

Datasets are big and shared between several people, so they live in **one
external folder**, outside the repo. Everyone points the same environment
variable at their own copy:

```bash
export STAIR_DATA_ROOT=/path/to/shared/data     # put it in your shell rc
python scripts/check_data.py --load             # does my copy match the layout?
```

Configs name datasets *relatively* (`root: VOID/void_1500`), so the same
`configs/dataset/*.yaml` works on every machine. An absolute `root:` overrides
the variable if you need a one-off. Default when unset: `<repo>/../data`.

Nothing but the split files in `data/splits/` ever goes into git.

## Expected layout of the shared folder

```
$STAIR_DATA_ROOT/
├── VOID/void_1500/      # VOID, stair sequences        -> loader void_stairs
├── MinJiang-Dataset/    # dual-view RGB-D on stairs    -> loader minjiang
├── nyu_depth_v2/        # NYU Depth v2                 -> loader nyuv2
├── ZJUL5/               # ZJU-L5, real dToF            -> loader zju_l5
├── HAMMER/              # HAMMER, multi-sensor         -> loader hammer
├── Middlebury/          # depth-SR test set, 30 pairs  -> loader middlebury
└── Lu/                  # depth-SR test set, 6 pairs   -> loader lu
```

Folder names are what `scripts/check_data.py` and the configs expect. Rename
yours or change `root:` in the config — but pick one and keep it the same for
all three of us, because a split file is a list of paths.

## Overview

| Dataset | Role | Input | RGB | Depth | Depth encoding | GT source |
|---|---|---|---|---|---|---|
| [VOID](#void) | **primary stair benchmark** | real sparse (~1500 pts) | 640×480 | 640×480 | uint16 **/ 256** → m | dense reference |
| [MinJiang](#minjiang) | stair training / qualitative | none (simulate) | 640×480 | 640×480 | uint16 mm | ⚠ the sensor itself |
| [NYUv2](#nyu-depth-v2) | general benchmark | none (simulate) | 640×480 | 640×480 | float m / normalised | filled Kinect |
| [ZJU-L5](#zju-l5) | **DEPTHOR benchmark** | real 8×8 dToF | 640×480 | 8×8 zones | float m | stereo reconstruction |
| [HAMMER](#hammer) | **sensor-gap study** | real, per sensor | 1224×1024 | per sensor | uint16 mm | laser |
| [Middlebury](#lu-and-middlebury) | depth-SR comparison | none (simulate) | varies | varies | **uint8 / 255 → normalised** | structured light |
| [Lu](#lu-and-middlebury) | depth-SR comparison | none (simulate) | varies | varies | **uint8 / 255 → normalised** | ASUS Xtion Pro |
| `synthetic_stairs` | CI only | generated | any | any | — | generated |

"Input: none (simulate)" means the dataset ships only ground truth, so a
degradation model from `configs/degradation/` produces the network input.

---

## VOID

`root: VOID/void_1500` · loader `void_stairs` · [github.com/alexklwong/void-dataset](https://github.com/alexklwong/void-dataset)

The closest public analogue of our task: a **real** sparse depth input — points
tracked by visual-inertial odometry, not a simulation — with dense ground truth,
on real staircases. Four stair sequences (`stairs0`, `stairs1`, `stairs3`,
`stairs4`), ~1270 frames each.

```
void_1500/
├── data/<sequence>/
│   ├── image/<timestamp>.png          # 640x480 RGB, uint8
│   ├── sparse_depth/<timestamp>.png   # uint16, ~1500 non-zero points
│   ├── ground_truth/<timestamp>.png   # uint16, dense
│   ├── validity_map/<timestamp>.png   # uint16, 256 where sparse is valid
│   ├── absolute_pose/<timestamp>.txt
│   └── K.txt                          # 3x3 intrinsics, one row per line
├── train_image.txt  train_sparse_depth.txt  train_ground_truth.txt  ...
└── test_*.txt                         # official file lists
```

> ⚠ **Depth is `uint16 / 256.0` metres**, not millimetres. Every other dataset
> here uses 1000. Get it wrong and depth is off by 3.9×.

`void_150` and `void_500` are the same scenes with fewer sparse points — useful
for a sparsity ablation. Set `root:` accordingly.

Intrinsics from `K.txt` land in `meta["intrinsics"]` as `[fx, fy, cx, cy]`; the
Astra 2 degradation model needs them.

## MinJiang

`root: MinJiang-Dataset` · loader `minjiang` · [github.com/Zhoujiahuan1005/MinJiang-Dataset](https://github.com/Zhoujiahuan1005/MinJiang-Dataset)

Consumer RGB-D of obstacles a robot meets, from two synchronised viewpoints.
`STAIRS` has 909 frames per camera, `ELEVATOR` 284.

```
MinJiang-Dataset/
├── STAIRS/
│   ├── CAM1_RGB/<timestamp>_rgb_cam1.png          # 640x480 uint8
│   ├── CAM1_GDEP/<timestamp>_grey_depth_cam1.png  # 640x480 uint16 MILLIMETRES
│   ├── CAM1_DEP/<timestamp>_depth_cam1.png        # colourised preview, NOT depth
│   └── CAM2_RGB/ CAM2_GDEP/ CAM2_DEP/             # second viewpoint
└── ELEVATOR/                                       # same six folders
```

Two traps:

* **`_DEP` is not depth.** It is an 8-bit colourised visualisation. The real
  measurement is `_GDEP` ("grey depth"). The loader reads `_GDEP` only and
  raises if you point it at `_DEP`.
* **The depth is both the ground truth and the sensor output.** There is no
  independent reference, so a completion method is scored against the very
  sensor it is meant to fix, and holes are simply dropped from the mask. RMSE
  from this dataset is optimistic. Use it for training and qualitative stair
  results; label any number that comes from it. `meta["gt_source"]` says so on
  every sample.

Select scene and camera in the config: `scenes: [STAIRS]`, `cameras: [CAM1]`.

## NYU Depth v2

`root: nyu_depth_v2` · loader `nyuv2` · [cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

The official pack plus two incompatible preprocessed ones circulate; the
baselines each use one, so the loader reads all three. Pick with
`dataset.layout`.

```
# layout: mat  -- official labeled pack (this is what we have)   [default]
nyu_depth_v2/nyu_depth_v2_labeled.mat          # keys: images, depths, rawDepths

# layout: h5   -- completion pack (NLSPN / CompletionFormer / Deltar / DEPTHOR)
nyu_depth_v2/nyudepthv2/{train,val}/<scene>/*.h5   # keys: rgb, depth, sometimes raw

# layout: npy  -- super-resolution pack (DKN / FDSR / DuCos)
nyu_depth_v2/{test_images_v2.npy, test_depth.npy, test_minmax.npy}
```

* `mat`: MATLAB v7.3, i.e. HDF5 — read with `h5py`, not `scipy.io.loadmat`.
  All 1449 labeled samples, depth is float metres. MATLAB is column-major, so
  the loader transposes. `rawDepths` is the unfilled Kinect depth and is used
  as a real sparse input, exactly as the h5 pack's `raw` is.
  ⚠ **No official train/test split in this file** — the loader returns all
  1449 samples, which is fine to benchmark on but must never be trained on.
* `h5`: depth is float metres. The `raw` key, where present, is the *unfilled*
  Kinect depth — a genuine sparse sensor input, better than any simulation. The
  loader uses it automatically.
* `npy`: depth is **min-max normalised per image**. `test_minmax.npy` holds the
  `(max, min)` in metres and the loader denormalises with it. Without that file
  every metric is silently wrong by a per-image scale factor, so the loader
  refuses to run rather than guess.

Download links: NLSPN's README for the h5 pack, DKN's for the npy pack.

## ZJU-L5

`root: ZJUL5` · loader `zju_l5` · released with [Deltar](https://github.com/zju3dv/deltar) (ECCV 2022)

Real ST VL53L5CX 8×8 dToF plus RGB. This is where DEPTHOR reports its headline
numbers, so it is our cross-check against published results.

```
ZJUL5/
├── data.json                # {"train": [...], "test": [{"filename": ...}]}
├── theater/<timestamp>.h5
├── lab1/  cafe1/  cafe2/
```

Each `.h5`: `rgb` (480,640,3 uint8), `depth` (480,640 float32 m, stereo
reference), `hist_data` (64,2 — per-zone histogram mean and variance), `fr`
(64,4 — zone pixel rectangles), `mask` (64 — per-zone validity).

`footprint: center` (default) reproduces DEPTHOR's `dtof_to_sparse_depth`
exactly — one pixel per valid zone, at the rectangle centre. `footprint: zone`
fills the whole rectangle instead; useful, but **not** comparable with their
published numbers.

Download: scriptable with `gdown` (verified 2026-09-09) —

```bash
uv run --with gdown python -c "import gdown; gdown.download_folder(
  url='https://drive.google.com/drive/folders/1ZGUdagrmFDr90Lm6qG1FkbZR_Tgpmr64',
  output='/tmp/deltar')"
unzip /tmp/deltar/ZJUL5.zip -d $STAIR_DATA_ROOT/
```

The folder also holds `demo.zip`, `nyu.pt` and calibration data we do not use.
527 test frames across 10 scenes.

## HAMMER

`root: HAMMER` · loader `hammer` · [github.com/Junggy/HAMMER-dataset](https://github.com/Junggy/HAMMER-dataset)

The only public dataset with an **active stereo** stream (RealSense D435 — the
same sensing principle as the Astra 2), a dToF stream (L515), an I-ToF stream
and laser-accurate ground truth for the same frames. That makes it the one place
to measure the "dToF method applied to active stereo" domain gap without also
changing the scene.

13 scenes; 2–11 train, 12–14 test. Direct download, no registration (~50 GB):
`http://www.campar.in.tum.de/public_datasets/2022_arxiv_jung/_dataset_processed.zip`

The archive's naming is not documented upstream; this is what is actually in it:

```
HAMMER/<scene>_traj<n>_<m>/
├── polarization/          # the RGB camera IS the polarization camera
│   ├── rgb/<frame>.png            # what the loader reads as rgb_subdir
│   ├── _gt/<frame>.png            # laser GT, gt_subdir
│   ├── depth_d435/<frame>.png     # active stereo, input_subdir (default)
│   ├── depth_l515/  depth_tof/    # the other two streams
│   └── pol/  _instance/  _pose/
├── d435/  l515_depth/  tof/   # same streams in each sensor's own frame
└── extrinsics/
```

> ⚠ Read the streams from **`polarization/`**, not from the per-sensor folders.
> The authors ship every depth stream already warped into the RGB frame there,
> so rgb, GT and input are siblings and mutually registered. The per-sensor
> folders hold the unwarped originals — using them means doing the projection
> ourselves for no gain.

The subdirectory names are config keys (`rgb_subdir`, `gt_subdir`,
`input_subdir`), so switching sensor is a one-line override:
`dataset.input_subdir=polarization/depth_l515`.

Depth is uint16 millimetres. `input_sensor: d435` (active stereo, our analogue),
`l515` (dToF) or `tof`.

## Lu and Middlebury

`root: Middlebury` / `root: Lu` · loaders `middlebury` / `lu` · [web.cecs.pdx.edu/~fliu/project/depth-enhance](https://web.cecs.pdx.edu/~fliu/project/depth-enhance/)

The two small test sets every guided depth-SR paper reports on, DuCos and WAVE
included. Both ship in one 46 MB archive, `Depth_Enh.zip`, released with Lu et
al., "Depth Enhancement via Low-Rank Matrix Completion" (CVPR 2014):

```
Depth_Enh/
├── 01_Middlebury_Dataset/   30 pairs  -> Middlebury/
├── 02_RGBZ_Dataset/          9 pairs  -> not part of the benchmark, skip
└── 03_RGBD_Dataset/          6 pairs  -> Lu/
```

Each scene ships three variants — `clean_*`, `noisy_*`, `output_*`. Only
`output_*` is used; the other two are the input and the intermediate result of
the authors' own completion method. Lay them out flat:

```
Middlebury/<scene>_output_color.png   <scene>_output_depth.png     # 30 pairs
Lu/<scene>_output_color.png           <scene>_output_depth.png     # 6 pairs
```

> ⚠ **Depth is not metric here.** It is an 8-bit PNG divided by 255 — a
> normalised scale with no metres in it. This is the one exception to rule 4 in
> `CLAUDE.md`, because the data has no metric scale to convert to. RMSE from
> these two sets lives on `[0, 1]`; DuCos multiplies it by 100 and calls it
> "centimetres". **Never put it in the same column as a metric RMSE** from VOID,
> ZJU-L5 or NYUv2. `meta["depth_unit"]` says so on every sample.

> ⚠ **`03_RGBD_Dataset` misspells the RGB files** as `ouput_color` (sic), while
> its depth files are spelled correctly. Rename on copy:
> ```bash
> for f in 03_RGBD_Dataset/*ouput_color*; do
>   cp "$f" "Lu/$(basename "$f" | sed 's/ouput_color/output_color/')"
> done
> ```
> Do not teach the loader the typo — DuCos hard-codes `output_color` too, and
> finds zero RGB frames in the archive as shipped.

Both sets are GT-only, so they need a degradation: the published protocol is
bicubic ×4/×8/×16, i.e. `--degradation bicubic_x8`. The loader crops to a
multiple of 16 first (`mod_crop`), reproducing upstream's `modcrop`, because the
published numbers are scored on the cropped frame.

---

## What a loader must produce

Whatever the layout, `__getitem__` returns the same thing — the contract in
[`data/loaders/base.py`](../data/loaders/base.py):

| key | shape / dtype | meaning |
|---|---|---|
| `rgb` | (H, W, 3) float32 | RGB in [0, 1], aligned to `gt_depth` |
| `gt_depth` | (H, W) float32 | **metres**, 0 = no measurement |
| `sparse_depth` | (H, W) float32 | completion-style input, metres, 0 = none |
| `sparse_mask` | (H, W) bool | where `sparse_depth` has a value |
| `lr_depth` | (h, w) float32 | SR-style input, dense, metres |
| `mask` | (H, W) bool | GT validity — the mask metrics use |
| `meta` | dict | dataset, sample_id, split, input_modality, intrinsics… |

Every sample carries **both** input forms, one native to the dataset and one
derived by a documented rule, so an adapter never branches on which dataset it
is looking at. Adding a loader: implement `_build_index` and `_load_raw`,
everything else (unit checks, resizing, mask construction, deriving the missing
input form) happens once in the base class. `tests/test_loaders.py` enforces it.

## Split protocol

Splits are frozen as JSON in `data/splits/<dataset>/<split>.json` and are the
only data files in git:

```json
{"dataset": "void_stairs", "split": "test", "ids": ["stairs0/1552024072.1300", "..."]}
```

1. **Prefer the authors' split** when one exists, so our numbers stay comparable
   with published tables: ZJU-L5's `data.json`, VOID's `test_*.txt`, HAMMER's
   scene 12–14 convention.
2. **Never split sequential data by frame.** Consecutive frames of one staircase
   are near-duplicates; a random frame split leaks the test set into training
   and inflates every metric. Split by sequence / scene / capture session — this
   bites VOID, MinJiang and HAMMER.
3. Changing a split invalidates every number previously reported on that
   dataset. Say so in the commit message.

Freeze one with `BaseDepthDataset.write_split_file(path)`.

## Checkpoints

Weights are not datasets: they go in `checkpoints/` inside the repo (git-ignored).

| What | Size | Where |
|---|---|---|
| Depth Anything V2 ViT-S — **both baselines need it** | ~100 MB | `huggingface.co/depth-anything/Depth-Anything-V2-Small` |
| DuCos | ~215 MB each | `huggingface.co/RaynWu2002/DuCos`, files under `DuCos/ckpts/` |
| DEPTHOR v1 (`depthor_zju_large.pt` 148 MB, `_small.pt` 121 MB) | ~270 MB | `gdown` the Drive ids from `third_party/depthor/README.md`, verified working |
| DEPTHOR++ | — | **does not exist publicly**, see [metrics_protocol.md](metrics_protocol.md#baseline-availability) |

## Datasets we looked at and did not take

**Stair dataset with depth maps** — depth is an 8-bit *colourised* PNG and the
labels are bounding boxes. It is a stair **detection** dataset; there is no
metric depth in it, so it cannot serve as ground truth here. Usable only as an
RGB source if we ever need one.
