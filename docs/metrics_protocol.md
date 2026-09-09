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

## Known-good anchors

Two properties the harness checks continuously, so a regression in the metric
code is caught by CI rather than by a wrong number in a report:

* **Identity degradation ⇒ perfect score.** With `degradation=identity` the
  input *is* the ground truth, so RMSE must be exactly 0 and δ1 exactly 1
  (`tests/test_loaders.py::test_identity_degradation_round_trips_exactly`).
* **Analytic metric values.** `tests/test_metrics.py` computes every metric on a
  2×2 frame whose expected values are derived by hand in the test docstrings —
  not copied from a run.
