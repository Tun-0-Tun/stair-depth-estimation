# A' -- WAVE + validity channel + selective fusion + stair-aware losses

One output (HR depth). Stair knowledge lives in the selective fusion (network) and in two
training-only losses. RGB branch (ML-DWT + DINO) and decoder are WAVE's, untouched.

| Piece | File |
|---|---|
| model, fusion, checkpoint loader (the only WAVE patch) | `models/wave_aprime.py` |
| `L = mean_valid(W * L_rec) + beta * L_planar` | `models/aprime_losses.py`, `models/planar_loss.py` |
| edge band `E` (binary / gaussian) from StairNet masks | `data/stairnet/edges.py` |
| StairNet-DepthOut port + cache | `data/stairnet/infer.py`, `cache.py` |
| strong stair-aware degradation | `data/degradation/astra_hard.py` |
| step-geometry metric (needs K) | `metrics/stair_geometry.py` |
| train / eval / ablations | `models/aprime_runner.py`, `scripts/{train,eval}_aprime.py`, `scripts/run_ablations.py` |

## Order of work (on sinope, `STAIR_DATA_ROOT` set)

```bash
python scripts/make_splits.py                       # data/splits/ours_seed0.json  (by staircase)
python scripts/astra_defect_stats.py                # real Astra defects + what each severity does; tune SEVERITY
# StairNet weights -> $STAIR_DATA_ROOT/weights/stairnet/depth_out/best.pth
python scripts/stairnet_check_mendeley.py           # port vs authors' val labels
python scripts/stairnet_cache.py                    # masks -> $STAIR_DATA_ROOT/ours/stairnet_cache
python scripts/stairnet_viz.py                      # ~50 overlays -> outputs/stairnet_viz  (LOOK at them: tread/riser ids)
python scripts/train_aprime.py configs/aprime/a2_fusion_orig.yaml --smoke 5      # all losses finite + decreasing
python scripts/run_ablations.py --severity strong --dry-run                      # prints the commands
```

Rows needing intrinsics (`a4_planar`, step-geometry columns) switch on when `intrinsics: [fx, fy, cx, cy]`
(528x688 frame) is set in `configs/dataset/ours.yaml`; until then `use_planar_loss` is forced off with a warning.

## Protocol facts (read before comparing numbers)

* GT is the Astra 2 sensor's own depth: no independent reference. Numbers rank methods, they are not accuracy.
* Input for every method: degraded HR depth -> `masked_downsample` x8 (A', DuCos: dense filled LR) or the HR map with
  holes (WAVE/DEPTHOR adapters). delta1 = 1.25 (DuCos tables use 1.05 -- not mixed).
* Normalisation: A' and the WAVE adapter: per-frame min/max of the VALID INPUT; DuCos adapter: min/max of the upsampled
  input; DEPTHOR: fixed dataset range. None uses GT statistics. Original WAVE training normalised by GT min/max.
* Splits by staircase; list in `data/splits/ours_seed0.json`. Test corruption deterministic per frame, training
  corruption re-drawn every epoch (`stochastic`, seeded).
* Severity presets in `astra_hard.py` are starting values, not calibrated to the real Astra yet.
* DuCos: only the x4 checkpoint is on disk; DEPTHOR was trained on 8x8 dToF zones, not on dense-with-holes input.

## Third-party and attribution

* WAVE -- Nasir, Liu, Mian, *WAVE: Reversing the Guidance Hierarchy for Coarse-to-Fine Guided Depth Super-Resolution*
  (MIT; the WAVE README cites arXiv 2601.17723, our adapter 2608.25302 -- one of them is wrong). `third_party/wave`.
* StairNet-DepthOut (RGB-only, StairNetV3) and StairNet-DepthIn (RGB-D, selective module, StairNetV2), C. Wang et al.,
  <https://github.com/MrChenWang/StairNet-DepthOut>, <https://github.com/MrChenWang/StairNet-DepthIn>,
  **CC BY-NC 3.0, non-commercial use only** (permission for this project confirmed by the project lead).
  The authors warn that the code may differ from the papers. Imported, not copied.
* Wang, Pei, Qiu, Tang, *RGB-D-Based Stair Detection and Estimation Using Deep Learning*, Sensors 23(4), 2175 (2023),
  doi 10.3390/s23042175 (StairNetV2, selective module).
* Wang, Pei, Qiu, Wang, Tang, *StairNetV3: depth-aware stair modeling using deep learning*, The Visual Computer (2024),
  doi 10.1007/s00371-024-03268-8.
* RGB-D stair dataset, Mendeley Data, doi 10.17632/6kffmjt7g2.1.
* Perez-Yus, Lopez-Nicolas, Guerrero, ECCV-W 2014 and CVIU 2017 -- stairs as parallel planes, riser/tread by normal angle
  (the planar loss is our transfer of that idea).
