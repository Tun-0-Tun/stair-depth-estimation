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

### DEPTHOR v1 — runnable, but only on CUDA

DEPTHOR's CSPN++ refinement does `import BpOps`, a CUDA extension from
[BP-Net](https://github.com/kakaxi314/BP-Net) that builds against **CUDA 12.1
only**. There is no CPU or MPS path, so this baseline **cannot run on a Mac**.
Everything else in the adapter is verified working on any machine (the
Depth-Anything-V2 path patch included).

On the CUDA box:

```bash
git clone https://github.com/kakaxi314/BP-Net && cd BP-Net && python setup.py install
python scripts/run_baseline.py --model depthor --dataset zju_l5 --check
```

### DuCos — runnable

Verified end-to-end on 2026-09-01 with the official `x4.pth.tar` checkpoint,
on CPU, through `scripts/run_baseline.py`. No CUDA-only dependencies.

---

## Cross-check against published numbers

**Status: not yet performed — blocked on the datasets, not on the code.**

The comparison the SOW asks for requires ZJU-L5 (manual download), RGB-D-D
(access by request) and the DEPTHOR checkpoints (Google Drive). None of them
can be fetched by a script, and none are on the machine this repo was built on.
The harness is ready; running it is a data-access task, not a code task.

### What to run

```bash
# DEPTHOR v1 on its headline benchmark (needs CUDA + BpOps)
python scripts/run_baseline.py experiment=depthor_zju_l5

# DuCos on its real-world benchmark
python scripts/run_baseline.py experiment=ducos_rgbdd_real

# DuCos on the synthetic x4 protocol
python scripts/run_baseline.py experiment=ducos_nyuv2_x4
```

### Reference numbers to compare against

Fill this table in as the runs complete. Take the published values from the
papers' own tables, and record the exact table/row you took them from.

| Method | Dataset | Metric | Published | Ours | Δ | Within ±10%? |
|---|---|---|---|---|---|---|
| DEPTHOR v1 | ZJU-L5 | RMSE | _(paper Tab. ?)_ | | | |
| DEPTHOR v1 | ZJU-L5 | AbsRel | | | | |
| DEPTHOR v1 | ZJU-L5 | δ1 | | | | |
| DuCos | RGB-D-D (real) | RMSE | | | | |
| DuCos | NYUv2 ×4 | RMSE | | | | |

### Expected sources of disagreement

Before concluding "our metric code is wrong", check these — most of them shift
a number by more than 10% on their own:

1. **Unit convention.** DuCos reports RMSE in centimetres on min-max normalised
   depth; ours is in metres on absolute depth. These are not off by a constant
   factor — the normalisation is per image, so the ratio varies per sample.
   Converting their table to ours requires their per-image min/max.
2. **δ thresholds.** DuCos's "DEL_105 / DEL_115 / DEL_125" are 1.05 / 1.15 /
   1.25 — only their third column is our δ1.
3. **Crop.** `border6` vs `eigen` vs none moves RMSE by a few percent.
4. **Depth range.** A `max_depth` of 8 vs 10 changes which pixels are scored.
5. **Checkpoint.** DuCos ships a different checkpoint per scale and per dataset;
   `x4.pth.tar` on RGB-D-D is not what their table reports.
6. **Split.** See [datasets.md](datasets.md#split-protocol).

### Rule for recording the outcome

Write the result here **whether or not it matches**. If a number is outside
±10%, the entry must name the cause or say explicitly that the cause is
unknown. "We could not reproduce it and do not know why" is a legitimate and
useful finding; quietly dropping the row is not.

---

## Known-good anchors

Two properties the harness checks continuously, so a regression in the metric
code is caught by CI rather than by a wrong number in a report:

* **Identity degradation ⇒ perfect score.** With `degradation=identity` the
  input *is* the ground truth, so RMSE must be exactly 0 and δ1 exactly 1
  (`tests/test_loaders.py::test_identity_degradation_round_trips_exactly`).
* **Analytic metric values.** `tests/test_metrics.py` computes every metric on a
  2×2 frame whose expected values are derived by hand in the test docstrings —
  not copied from a run.
