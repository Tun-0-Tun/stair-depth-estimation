# Design doc

Depth completion / super-resolution for the Orbbec Astra 2, specialised for
staircases. Joint SOW with Huawei.

## The problem, stated precisely

The Astra 2 is an active-stereo camera. On a staircase it fails in a specific,
structured way, and that structure is the whole project:

1. **Occlusion shadows.** Every tread edge is a depth discontinuity. A surface
   visible to one IR camera is hidden from the other, producing a band of
   invalid pixels of width `f·B·(1/Z_fg − 1/Z_bg)` next to each edge. On stairs
   these bands are *periodic and aligned with exactly the geometry a robot needs
   to measure*.
2. **Quadratic depth error.** Disparity is quantised (~1/8 px), so depth error
   grows as `Z²/(f·B)`. At 4 m the Astra 2's quantisation step alone is
   comparable to a stair riser.
3. **Low-signal dropout.** Dark or specular treads (polished stone, metal
   nosings, glass balustrades) return no pattern.

A generic depth-completion method trained on dToF data does not model any of
this. Our thesis is that (a) modelling the sensor correctly in the training
data and (b) refining in a way that respects depth discontinuities together
close most of the gap.

## Non-negotiables

| Constraint | Why |
|---|---|
| One metric implementation, no exceptions | Four methods × ten datasets; any second implementation makes the table meaningless. See [metrics_protocol.md](metrics_protocol.md). |
| Upstream code vendored as submodules, never edited | A baseline number must be attributable to the authors' code at a specific commit. Every difference lives in an adapter. |
| One experiment = one YAML file | The unit of reproducibility. If a result cannot be re-created from a committed file, it is not a result. |
| Data never enters git | Splits do. Everything else lives under git-ignored roots. |
| Degradation is seeded per sample | Two machines must produce byte-identical inputs, or "the metric moved" is uninterpretable. |

## Architecture

```
configs/experiment/*.yaml        one file = one reproducible run
        |
        v
  utils/config.py   (Hydra-layout composition, no Hydra dependency)
        |
        +---> data/loaders/*      one file per dataset -> BaseDepthDataset contract
        |         |
        |         +-> data/degradation/*   GT depth -> realistic sensor input
        |
        +---> baselines/*         thin adapters over third_party/* (never edited)
        |     models/*            our method, same interface
        |
        +---> metrics/            THE metric implementations
        |
        +---> utils/results.py    per-run JSON + one committed summary CSV
```

The one interface that matters is the **sample contract** in
[`data/loaders/base.py`](../data/loaders/base.py). Every sample carries *both*
input forms — a sparse full-resolution map (completion style) and a dense
low-resolution map (SR style) — one native to the dataset and one derived by a
documented rule. That is why `scripts/run_benchmark.py` contains no
`if model == ...` branches: a completion method and an SR method consume the
same sample.

### Why not Hydra

`configs/` uses Hydra's exact layout — groups as directories, a `defaults:` list
per experiment, `key.sub=value` overrides — but composition is 120 lines in
`utils/config.py` instead of a dependency. Reasons: the repo then runs on a bare
`python + torch + numpy` install (which matters inside the Orbbec SDK
container); Hydra's working-directory rewriting fights with relative dataset
roots; and migration is mechanical because the YAML does not have to change.
If we ever need sweeps or multirun, install `hydra-core` and delete that file.

## The Astra 2 noise model

[`data/degradation/active_stereo_astra2.py`](../data/degradation/active_stereo_astra2.py)
— **owner: Olesya.** This is the SOW deliverable "Astra 2 degradation model",
and it is the highest-leverage file in the repo: DEPTHOR's central claim is that
simulation realism, not architecture, determines real-world accuracy.

Modelled, in the physically correct domain rather than as depth-domain noise:

| Effect | Model |
|---|---|
| Disparity quantisation | round `f·B/Z` to `1/2^subpixel_bits` px |
| Matching noise | Gaussian on disparity, `sigma_disp_px` |
| Occlusion shadows | forward-warp z-buffer left→right consistency test |
| Edge fattening | local averaging in a band around discontinuities |
| Low-signal dropout | spatially correlated field, probability ∝ Z² |
| Operating range | hard clip to `[z_min, z_max]` |

### Calibration procedure (not yet done)

The shipped defaults come from the datasheet, **not from measurement**. Until
calibrated, every result using this model must carry `uncalibrated` in the
notes column — `configs/degradation/astra2_uncalibrated.yaml` is named to make
that hard to forget.

1. Capture stair scenes with the Astra 2 **and** a reference (laser scan or a
   slow high-quality reconstruction), ≥ 20 scenes across materials and
   lighting.
2. Compute the per-pixel error `Z_astra − Z_ref` and fit:
   * `sigma_disp_px` from the variance of the error mapped into disparity — it
     should be flat in disparity and quadratic in depth; if it is not, the
     model is wrong, not the parameter.
   * `dropout_base` and `dropout_z_ref` from the observed invalid-pixel rate as
     a function of depth and surface luminance.
   * `edge_invalid_prob` and `edge_fatten_px` from the invalid-band width
     measured at known depth discontinuities.
3. **Validate, do not just fit.** Apply the fitted model to held-out reference
   depth and compare the *distribution* of simulated vs real error — at minimum
   the invalid-pixel rate, the error histogram per depth bin, and the mean
   connected-component size of holes. A model that matches the mean but not the
   spatial structure will not transfer.
4. Save as `configs/degradation/astra2_calibrated.yaml` and record the fit
   quality here.

## Our method

[`models/pipeline.py`](../models/pipeline.py) — a working skeleton, not the
deliverable. Two encoders (RGB, sparse depth + validity mask) → confidence-gated
fusion → U-Net decoder → edge-aware refinement with measurements re-injected as
hard constraints. ~7.9M parameters.

Deliberate choices, each contestable and therefore each ablatable by flipping a
flag in `configs/model/ours.yaml`:

* **The validity mask is an input channel.** A zero means "no measurement", not
  "zero metres"; without the mask the first convolution cannot tell them apart.
  This is the most common bug in completion code.
* **Depth is normalised by the dataset range, not per sample.** Per-sample
  min-max (what DuCos does) leaks the ground-truth scale into the input and
  makes metric depth unrecoverable when the input is empty.
* **Refinement re-injects the measurements every iteration.** Where the sensor
  measured, we should not invent.
* **No monocular foundation backbone yet.** Both baselines lean on Depth
  Anything V2; adding it is the obvious next experiment, and
  `StairDepthNetConfig.mono_features` is the hook.

### Open research questions, in priority order

1. **Does the Astra 2 degradation model transfer?** Train on Hypersim +
   Astra 2 simulation, evaluate on real Astra 2 captures. This is the project's
   central risk: if simulated and real error distributions differ, everything
   downstream is built on sand. Measure it early, before optimising
   architecture.
2. **How much does a dToF-trained baseline lose on active stereo?** Run DEPTHOR
   on HAMMER's D435 stream vs its L515 stream. Same scenes, same GT, different
   sensor — the cleanest possible isolation of the domain gap.
3. **Does stair-specific supervision help, or does more data suffice?** Compare
   fine-tuning on stairs against training on more generic indoor data.
4. **What FPS is actually required?** The SOW lists FPS as an acceptance metric
   without a target. Get a number from the robotics side before optimising —
   DuCos's ViT backbone will not hit 30 FPS on an embedded device, and if the
   requirement is 5 FPS that changes the architecture search.

## Work split

| Area | Owner |
|---|---|
| Astra 2 degradation model + calibration captures | Olesya |
| Stair dataset capture + `astra2_custom` / `stair_dataset` loaders | Olesya |
| Remaining loaders (Hypersim, TartanAir, TOFDC, Mirror3D) | Maxim |
| Baseline runs on the CUDA box, cross-check table | Maxim |
| Metrics, harness, adapters, our method | tech lead |

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-01 | `third_party/depthor` points at **DEPTHOR v1** | DEPTHOR++ has no public code or weights; see [metrics_protocol.md](metrics_protocol.md#baseline-availability) |
| 2026-09-01 | Hydra layout, no Hydra dependency | zero-dependency runs; mechanical migration later |
| 2026-09-01 | Every sample carries both input modalities | keeps `if model == ...` out of the experiment code |
| 2026-09-01 | Reference baselines (`bicubic`, `nn_fill`) are first-class | a floor for the table, and an end-to-end pipeline check that needs no weights |
