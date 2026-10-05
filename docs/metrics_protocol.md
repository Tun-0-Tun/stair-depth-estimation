# Metrics protocol

The acceptance metrics in the SOW are **AbsRel, RMSE, δ1 and FPS**. This
document defines exactly how each is computed, so that a number in our report
means the same thing regardless of who produced it, on which dataset, for which
method.

## The rule

> **There is exactly one implementation of the depth metrics in this repository:
> [`metrics/depth_metrics.py`](../metrics/depth_metrics.py). Writing another one
> anywhere else — a notebook, a script, a training loop, an adapter — is a
> blocking review comment.**

Timing has the same rule and lives in
[`metrics/runtime_metrics.py`](../metrics/runtime_metrics.py).

This is not bureaucracy. The two baselines we must compare against do not agree
on what "RMSE" means:

| | DEPTHOR (`third_party/depthor/src/utils/utils.py`) | DuCos (`third_party/ducos/utils/metric_func.py`) |
|---|---|---|
| Units | metres | centimetres, on min-max **normalised** depth |
| Valid mask | `min_depth < gt < max_depth` | `gt >= 0`, plus a 6-pixel border crop |
| δ thresholds | 1.25, 1.25², 1.25³ | **1.05, 1.15, 1.25** |
| Aggregation | per-image average | per-image average |

Their published numbers therefore cannot be put in one table. We recompute
everything from raw metric depth in metres.

---

## Definitions

Let `p` be the prediction and `g` the ground truth over the set of valid pixels
`V` (defined below), `N = |V|`.

| Metric | Formula | Direction |
|---|---|---|
| **AbsRel** | `mean(|p − g| / g)` | ↓ |
| **RMSE** | `sqrt(mean((p − g)²))`, in **metres** | ↓ |
| RMSE(log) | `sqrt(mean((log p − log g)²))` | ↓ |
| **δ1 / δ2 / δ3** | `mean( max(p/g, g/p) < 1.25^k )`, k = 1, 2, 3 | ↑ |
| MAE | `mean(|p − g|)` | ↓ |
| SqRel | `mean((p − g)² / g)` | ↓ |
| SILog | `sqrt(Var(log p − log g)) × 100` | ↓ |

The δ comparison is **strict** (`<`, not `≤`): a pixel exactly on the threshold
does not count. This matches the reference implementations; getting it wrong
shifts δ1 by 1/N, which is invisible on a large dataset and just as wrong.
`tests/test_metrics.py::test_delta_threshold_is_strict_and_counts_the_right_pixels`
pins it.

## Valid pixels

A pixel is scored iff **all** of:

1. `gt` is finite (`NaN`/`inf` in ground truth means "no measurement"),
2. `min_depth < gt < max_depth`,
3. the loader's own `mask` is true there (e.g. unlabelled regions, mirror masks).

Note the range test is **strict on both sides**, so `gt == 0` — the universal
"no measurement" encoding — is excluded, and so is a ground-truth value pinned
at exactly `max_depth` by a clamp upstream.

`min_depth` / `max_depth` are per dataset, declared in
`configs/dataset/*.yaml`, never hard-coded:

| Dataset | min | max | Rationale |
|---|---|---|---|
| NYUv2 | 0.001 | 10.0 | the convention of every NYU table |
| ZJU-L5 | 0.001 | 10.0 | DEPTHOR's `configs/test_zju.txt` |
| ARKitScenes | 0.1 | 10.0 | indoor lidar range |
| HAMMER | 0.1 | 5.0 | tabletop capture volume |
| RGB-D-D | 0.1 | 10.0 | phone ToF range |
| Astra 2 (ours) | 0.6 | 8.0 | the sensor's datasheet operating range |

## Predictions

Predictions are **clamped** into `[min_depth, max_depth]` before scoring
(`MetricConfig.clamp_pred`, default on). `NaN` and `+inf` map to `max_depth`,
`−inf` to `min_depth`.

Deliberate choice: a non-finite prediction is a model failure and must be
**penalised, not dropped**. Excluding such pixels from the mask would let a
model improve its score by predicting garbage where it is uncertain.

## Crops

`MetricConfig.crop`, set per dataset in `configs/dataset/*.yaml`:

| Value | Effect | When |
|---|---|---|
| `none` | full frame | default; our own evaluations |
| `eigen` | rows 45:471, cols 41:601 of a 480×640 frame | comparing against NYUv2 completion tables |
| `border6` | drop a 6-pixel border | comparing against DSR tables (DuCos, FDSR, DKN) |

`eigen` raises if the frame is not 480×640, rather than cropping something
arbitrary.

## Aggregation

Default `pooling: image` — compute the metrics per image, then average over
images. This is what the depth completion and DSR literature reports, and what
makes our numbers comparable with theirs.

`pooling: pixel` pools all valid pixels and computes the metrics once. It
weights images with large valid regions more heavily. Use it for ablations if
you have a reason; **never** for headline numbers, and always say which you
used.

Images with fewer than `min_valid_pixels` valid pixels are skipped and counted
in `n_skipped`, which appears in the results row. A run with a large
`n_skipped` is not a good run — it means the mask or the depth range is wrong.

## FPS and latency

Defined in [`metrics/runtime_metrics.py`](../metrics/runtime_metrics.py):

* `batch_size = 1` — single-frame latency is what matters on the robot.
* `warmup` iterations discarded (CUDA autotuning, cuDNN algorithm selection,
  MPS graph compilation), then `iters` timed iterations.
* The device is synchronised before starting and before stopping the clock.
  Without this, CUDA timings measure queue submission, not execution, and come
  out 10–100× too fast.
* Timing covers **only** `predict` — no disk I/O, no metric computation. The
  pre/post-processing inside an adapter **is** included, because it is part of
  the method's real cost.
* Headline number is `1 / median_latency`. Mean, p95 and std are recorded too,
  so a noisy measurement is visible rather than hidden.

FPS is only meaningful together with the device and the input resolution, both
of which are recorded in the run JSON. **Never compare an FPS measured on one
machine with one measured on another.**

---

## Baseline availability

What each baseline is, where it comes from and what it takes as input:
[baselines.md](baselines.md). This section covers only whether it can be
run here, and what to do when it cannot.

### DEPTHOR++ — code and weights are not public

Status as of **2026-09-01**:

| | Code | Weights | Notes |
|---|---|---|---|
| DEPTHOR (ICCV 2025, arXiv:2504.01596) | ✅ <https://github.com/ShadowBbBb/Depthor> | ✅ Google Drive (ZJU-Large / ZJU-Small) | vendored at `third_party/depthor` |
| **DEPTHOR++ (arXiv:2509.26498)** | ❌ **none** | ❌ **none** | the paper links no repository; the authors' repo contains v1 only |

The SOW names DEPTHOR++ as a mandatory baseline. We cannot run it. What the
repo does about that:

* `third_party/depthor` is the **v1** submodule, named honestly.
* `configs/model/depthor.yaml` runs v1 and is the row we can actually produce.
* `configs/model/depthor_plus_plus.yaml` exists, is marked `available: false`,
  and its adapter raises an explanatory error. **A v1 number must never appear
  in a DEPTHOR++ row.**

Options, in the order we recommend them:

1. **Report DEPTHOR-v1 and say so.** v1 is itself the SOTA on ZJU-L5; per the
   ++ abstract the delta is ~22% RMSE / ~11% Rel on ZJU-L5. A clearly-labelled
   v1 row plus that citation is defensible in a report.
2. **Ask the authors** (contact details in the paper) for weights or an early
   release. Worth doing now — it costs an email and the answer changes the plan.
3. **Re-implement ++ from the paper.** Only with an explicit team decision, and
   it must live in `models/`, never in `baselines/` — a re-implementation is our
   work, not the authors', and mislabelling it would be misconduct.

### DEPTHOR v1 — runnable anywhere, via a shim

DEPTHOR's CSPN++ refinement does `import BpOps`, a CUDA extension from
[BP-Net](https://github.com/kakaxi314/BP-Net) that builds against **CUDA 12.1
only** — no CPU or MPS build exists.

[`baselines/bpops_shim.py`](../baselines/bpops_shim.py) supplies that one
primitive (a local, per-pixel convolution) in portable PyTorch, transcribed from
BP-Net's own `conv2d_kernel_lf`. It is checked against a literal transcription
of that CUDA loop in `tests/test_bpops_shim.py`, and the ZJU-L5 reproduction
above was produced with it.

It is still *our* code, not the authors', so:

* the adapter prints a warning when it activates,
* `bpops` appears in the run record as `shim` or `cuda`,
* on a CUDA machine install the real extension and it takes precedence:
  `git clone https://github.com/kakaxi314/BP-Net && cd BP-Net && python setup.py install`

Weights: both `Depthor-ZJU-Large` and `-Small` download with `gdown` from the
Google Drive links in `third_party/depthor/README.md` — see
[datasets.md](datasets.md#checkpoints).

### DEPTHOR v1 off ZJU-L5 — feed it the right sparsity, or the number is meaningless

DEPTHOR consumes the sparse map directly, and it was trained on exactly one
convention: **one pixel per dToF zone**, its own `dtof_to_sparse_depth`, which
`footprint: center` reproduces for real ZJU-L5 data. Give it anything denser and
it stops using the measurements and falls back on its learned depth prior. It
does not warn, and the failure looks like a bad method rather than a bad input.

Measured on 10 MinJiang frames, everything else held fixed, varying only the
zone footprint (`configs/degradation/dtof_l5*.yaml`):

| input | absrel | rmse | δ1 | silog | pred/gt |
|---|---|---|---|---|---|
| `dtof_l5` — 0.6 of the zone, ~1700 px/zone | 2.8000 | 4.5328 | 0.0176 | 42.87 | 1.88–4.57 |
| `dtof_l5_center` — 1 px/zone | **0.1560** | **0.7741** | **0.8144** | **25.11** | 0.87–0.99 |
| `nn_fill` on the same input (floor) | 0.2152 | 1.3382 | 0.6776 | 44.55 | — |

So: **use `dtof_l5_center` for any DEPTHOR-family model.** With it, DEPTHOR
beats the floor on every metric and its absolute scale sits on the measurements
(pred/gt ≈ 0.9–1.0) on a dataset it never saw. `nn_fill` cannot tell the two
apart — after nearest-neighbour densification both give the same piecewise
constant 8×8 map — so the floor does not reveal the problem.

The same reasoning limits VOID: its ~740 scattered VIO points are neither a zone
grid nor a dense LR map, and DEPTHOR over-predicts there by a constant ~1.5×
across every frame. Zone-structured input is what this family needs; a VOID
number is out-of-distribution and must be labelled as such.

#### The bin grid is not the evaluation window

DEPTHOR's head sums over `n_bins` centres spanning `[min_depth, max_depth]`, so
those two numbers belong to the *checkpoint*, not to the dataset being scored.
`configs/model/depthor.yaml` pins them at 0.001/10.0 and
[`model_depth_range`](../baselines/base.py) takes them in preference to `eval.*`;
without that, running on `void_stairs` (window starts at 0.2) re-spaced the bins
and moved δ2 by 12 points, silently. Baselines that do not state a range keep
using the evaluation window.

### DuCos — runnable

Verified end-to-end on 2026-09-01 with the official `x4.pth.tar` checkpoint,
on CPU, through `scripts/run_baseline.py`. No CUDA-only dependencies.

---

## Cross-check against published numbers

### DEPTHOR v1 on ZJU-L5 — reproduced (2026-09-09)

Full 527-frame test split, our loader, our metric code, `bpops="shim"`, CPU.
Reference: DEPTHOR paper (arXiv:2504.01596v2) **Table 2**, row *Ours-Large*,
RMSE in metres.

| Metric | Published | Ours | Δ | Within ±10%? |
|---|---|---|---|---|
| RMSE | 0.350 | 0.3501 | +0.03% | ✅ |
| Rel (AbsRel) | 0.075 | 0.0759 | +1.2% | ✅ |
| δ1 | 0.933 | 0.9321 | −0.1% | ✅ |

Context from the same run — the reference floors on the identical split:

| Method | AbsRel | RMSE | δ1 | FPS (CPU) |
|---|---|---|---|---|
| DEPTHOR-Large | 0.0759 | 0.3501 | 0.9321 | 1.19 |
| nn_fill | 0.1223 | 0.5775 | 0.8349 | 157 |
| bicubic | 0.1222 | 0.5771 | 0.8352 | 1816 |
| DuCos (x4 ckpt) | 0.1304 | 0.5526 | 0.8311 | — |

This single result validates four things at once, which is why it was worth
chasing: the DEPTHOR adapter, the ZJU-L5 loader (including the zone-centre
sparse construction), our unified metric implementation, and the BpOps shim.
Treat it as the regression anchor — if a change moves this row, the change is
wrong until proven otherwise.

DuCos scores at the bicubic floor here because ZJU-L5's input is 64 points in a
480×640 frame (0.02% dense). It is a super-resolution model being handed a
completion problem; the number is correct and the comparison is not meaningful.
Its own benchmarks are the SR ones.

### Still to do

| Method | Dataset | Status |
|---|---|---|
| DuCos | RGB-D-D (real) | dataset not obtained (access by request) |
| DuCos | NYUv2 ×4 | dataset not obtained |
| DEPTHOR | Mirror3D-NYU | no loader yet |

### Expected sources of disagreement

Before concluding "our metric code is wrong", check these — most of them shift
a number by more than 10% on their own:

1. **Unit convention.** DuCos reports RMSE in centimetres on min-max normalised
   depth; ours is in metres on absolute depth. These are not off by a constant
   factor — the normalisation is per image, so the ratio varies per sample.
2. **δ thresholds.** DuCos's "DEL_105 / DEL_115 / DEL_125" are 1.05 / 1.15 /
   1.25 — only their third column is our δ1.
3. **Crop.** `border6` vs `eigen` vs none moves RMSE by a few percent.
4. **Depth range.** A `max_depth` of 8 vs 10 changes which pixels are scored.
5. **Subset.** Our first ZJU-L5 attempt used 20 of 527 frames and gave
   RMSE 0.280 against a published 0.350 — a 20% "disagreement" that was purely
   the subset. Run the full split before comparing.
6. **Checkpoint.** DuCos ships a different checkpoint per scale and per dataset.
7. **Split.** See [datasets.md](datasets.md#split-protocol).

### Rule for recording the outcome

Write the result here **whether or not it matches**. If a number is outside
±10%, the entry must name the cause or say explicitly that the cause is
unknown. "We could not reproduce it and do not know why" is a legitimate and
useful finding; quietly dropping the row is not.

### DuCos x4 — reproduced (2026-10-05)

Reference: DuCos paper (arXiv:2503.04171) **Table 1**, ×4 RMSE. Upstream
reports Middlebury/Lu on the 0–255 scale (`midd_calc_rmse`) and NYU in cm, so
ours is `rmse × 255` and `rmse × 100`. Checkpoint `x4.pth.tar`, PIL bicubic LR,
`border6` crop; NYU = the last 449 frames of the labeled `.mat`.

| Dataset | Paper | Ours | Bicubic (ours) |
|---|---|---|---|
| Middlebury (30) | 1.45 | 1.45 | 2.30 |
| Lu (6) | 1.38 | 1.38 | 2.47 |
| NYU v2 (449), cm | 2.60 | 2.60 | 4.27 |

Before the LR map was built with PIL's anti-aliased resize (upstream's own
`Image.resize(..., BICUBIC)`), `F.interpolate` produced an aliased input and
DuCos scored 4.00 / 4.59 / 7.2 cm — worse than bicubic. A protocol mismatch can
make a SOTA model lose to its floor without any error; reproduce the paper
first.

### WAVE on NYU v2 — reproduced (2026-10-05)

Reference: WAVE paper (arXiv:2608.25302), NYU-trained rows, RMSE in cm; same
449 frames and PIL bicubic LR as the DuCos anchor.

| Scale | Paper | Ours |
|---|---|---|
| ×8 | 2.50 | 2.50 |
| ×16 | 4.60 | 4.61 |
| ×32 | 7.90 | 7.97 |

⚠ `configs/model/wave.yaml` labels the ×8 file as the HyperSim-trained model,
whose paper row is 4.61 on NYU. We get the NYU-trained row's 2.50, so the ×8
checkpoint is most likely the NYU-trained one -- check the upstream file name
before calling any ×8 row "zero-shot from HyperSim".

## Comparing methods on a GT-only dataset

Three things make MinJiang (and NYUv2, and Hypersim) easy to read wrongly. All
three were measured on 890 MinJiang frames on 2026-09-27.

### The floor is not neutral — it is the ground truth, filtered

A GT-only dataset has no sensor input, so a degradation model derives one *from
the same ground truth the metrics score against*. A no-learning baseline then
inverts that derivation:

* `nn_fill` on `dtof_l5_center` returns the GT quantised to 64 zones plus 1–3 cm
  of noise;
* `bicubic` on `bicubic_x8` returns the GT low-passed and resampled.

Neither predicts depth. Both inherit the GT's absolute scale exactly, so their
threshold metrics (δ1–δ3) measure *how well the sensor grid samples the scene*,
never *how well depth is recovered*. A learned method has to reconstruct the
structure **and** infer the scale from scratch, so "barely ahead of the floor on
δ1" does not mean "barely better than trivial". Read RMSE, SqRel and SiLog on
these datasets; treat δ against a derived floor as a sanity check only.

### Rows from different degradations are not comparable

The degradation decides how much information the method gets. On MinJiang:

| protocol | values per frame |
|---|---|
| `dtof_l5_center` | 64 |
| `bicubic_x8` | 4800 (a 60×80 grid) |

That is a 75× difference, so a WAVE row and a DEPTHOR row on MinJiang answer
different questions and must never be sorted into one ranking. Compare each
method with the floor **of its own protocol**:

| dToF protocol, 890 frames | DEPTHOR | `nn_fill` (floor) |
|---|---|---|
| absrel | **0.2111** | 0.2585 |
| rmse | **0.7822** | 0.8437 |
| sqrel | **0.5125** | 0.6365 |
| silog | **23.70** | 27.99 |
| rmse_log | 0.3770 | **0.3462** |
| δ1 / δ2 / δ3 | 0.776 / 0.859 / 0.890 | **0.789 / 0.874 / 0.918** |
| fps | 20.1 | 94.6 |

| ×8 SR protocol, 890 frames | WAVE | `bicubic` (floor) |
|---|---|---|
| absrel | **0.0502** | 0.0937 |
| rmse | **0.3318** | 0.4577 |
| rmse_log | **0.1936** | 0.2742 |
| sqrel | **0.0433** | 0.0812 |
| silog | **19.16** | 26.16 |
| δ1 / δ2 / δ3 | **0.939 / 0.965 / 0.979** | 0.858 / 0.918 / 0.953 |
| fps | 4.35 | 652.8 |

WAVE clears its floor on every metric. DEPTHOR splits: it wins everything that
penalises the *magnitude* of the error and loses everything that counts the
*fraction* of pixels inside a ratio.

### A split like DEPTHOR's is the signature of a scale bias

SiLog removes a global multiplier by construction; `rmse_log` does not. When one
improves and the other degrades, suspect a systematic factor rather than noise.
Measured directly on 5 MinJiang frames, DEPTHOR's `pred/gt` median runs
0.87–0.99 — it under-predicts by roughly 7%, which is enough to push pixels over
the δ1 threshold while barely moving SiLog. Worth re-measuring on the full split
before drawing conclusions from it.

The same test on VOID gives a *constant* 1.43–1.53 across every frame, including
at pixels where the sparse input already holds the exact depth. Constant factor
= a prior that has drifted; a factor that varies per frame (1.88–4.57, what
`dtof_l5` produced) = the model has lost its anchor entirely.

## Known-good anchors

Two properties the harness checks continuously, so a regression in the metric
code is caught by CI rather than by a wrong number in a report:

* **Identity degradation ⇒ perfect score.** With `degradation=identity` the
  input *is* the ground truth, so RMSE must be exactly 0 and δ1 exactly 1
  (`tests/test_loaders.py::test_identity_degradation_round_trips_exactly`).
* **Analytic metric values.** `tests/test_metrics.py` computes every metric on a
  2×2 frame whose expected values are derived by hand in the test docstrings —
  not copied from a run.
