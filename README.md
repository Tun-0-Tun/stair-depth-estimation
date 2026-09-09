# stair-depth-estimation

Depth completion / super-resolution for the **Orbbec Astra 2**, specialised for
**staircases**. Joint SOW with Huawei.

Stairs break active stereo in a specific way: every tread edge is a depth
discontinuity, so there is an occlusion shadow next to it, and the holes line up
with exactly the geometry a robot needs. The repo holds our method, the two
required baselines, and a harness that scores all of them with the same code.

## Setup

```bash
git clone --recurse-submodules <repo> && cd stair-depth-estimation
uv sync --group baselines           # omit the group if you only touch our code
export STAIR_DATA_ROOT=/path/to/shared/data
uv run pre-commit install
uv run pytest
uv run python scripts/run_baseline.py experiment=smoke
```

The smoke run goes loader → degradation model → method → metrics → results
table on a generated staircase. No data, no weights. If it passes, the setup is
fine.

## What the repo does

**Unified metrics.** `metrics/depth_metrics.py` is the only implementation of
AbsRel, RMSE, RMSE(log), δ1/δ2/δ3, MAE, SqRel, SILog; `metrics/runtime_metrics.py`
the only one of FPS/latency. Needed because the baselines don't agree on what
they even report — DuCos gives RMSE in centimetres on min-max-normalised depth
and calls 1.05/1.15/1.25 "δ". A pre-commit hook greps for metric arithmetic
outside `metrics/`.

**One dataset interface.** Five real datasets with five different layouts, units
and input types, all producing the same sample dict (metres, `[0,1]` RGB,
explicit validity masks). Every sample carries both a sparse full-resolution
input and a dense low-resolution one — one native, one derived — so a completion
method and an SR method consume the same sample and the experiment code has no
`if model == ...` branches.

**Sensor degradation models.** Datasets that ship only ground truth need a
simulated input. `data/degradation/` has the Astra 2 active-stereo model
(disparity quantisation, occlusion shadows, edge fattening, depth-dependent
dropout), a zone-based dToF model, bicubic SR and random-sparse. All seeded per
sample, so two machines produce byte-identical inputs.

**Baseline adapters.** `third_party/` holds the official repos as submodules and
is never edited; `baselines/` absorbs every difference — DuCos's per-sample
normalisation, DEPTHOR's fixed 480×640 geometry, both of their hard-coded
checkpoint paths. Each adapter has an `availability()` that says up front what
is missing instead of dying inside a forward pass.

**Config-driven experiments.** Hydra layout composed by `utils/config.py`
without the dependency. One experiment = one YAML in `configs/experiment/`.

**Result recording.** Each run appends a row to
`experiments/results/benchmark.csv` (in git) and dumps config, environment, git
commit, per-sample metrics and timing to `experiments/runs/`. W&B and MLflow are
optional behind a flag.

## Layout

```
data/loaders/       one file per dataset, all speaking one sample contract
data/degradation/   clean GT depth -> realistic sensor input
data/splits/        frozen id lists — the only data in git
third_party/        DEPTHOR and DuCos as submodules, never edited
baselines/          adapters over third_party/*, one interface
models/             our method
metrics/            the metric code
configs/            dataset / model / degradation / experiment YAML
scripts/            run_baseline, run_benchmark, train, check_data
tests/              metric correctness, loader contract, adapter smoke tests
```

## Data

Datasets live in one **external** folder shared by the team, pointed at by
`STAIR_DATA_ROOT` — see [docs/datasets.md](docs/datasets.md) for the required
layout of each.

```bash
uv run python scripts/check_data.py --load     # is my copy laid out right?
```

| Dataset | Loader | Role |
|---|---|---|
| VOID (stair sequences) | `void_stairs` | primary stair benchmark — real sparse input + dense GT |
| MinJiang | `minjiang` | stair training / qualitative (depth = the sensor, not a reference) |
| NYU Depth v2 | `nyuv2` | general benchmark, both the completion and SR packs |
| ZJU-L5 | `zju_l5` | real dToF — DEPTHOR's headline benchmark |
| HAMMER | `hammer` | active stereo vs dToF on identical scenes |

## Running things

```bash
# can this even run here? checks deps, weights, CUDA extensions — loads nothing
uv run python scripts/run_baseline.py --model depthor --dataset zju_l5 --check

uv run python scripts/run_baseline.py experiment=ducos_nyuv2_x4
uv run python scripts/run_baseline.py --model ducos --dataset void_stairs --limit 10

uv run python scripts/run_benchmark.py --datasets void_stairs zju_l5 --models ducos nn_fill
uv run python scripts/train.py --dataset minjiang --degradation astra2_uncalibrated
```

A benchmark cell that couldn't run shows up as `NOT RUN: <reason>` rather than
disappearing.

Full recipes — overrides, choosing a degradation, reading the output row,
what to do when it fails: **[docs/running.md](docs/running.md)**.

## Baseline status

| | Code | Weights | Runs on |
|---|---|---|---|
| DuCos (ICCV 2025) | ✅ submodule | ✅ HuggingFace | anything — **verified end to end** |
| DEPTHOR v1 (ICCV 2025) | ✅ submodule | ✅ Google Drive (`gdown`) | anything — **paper numbers reproduced**, via `baselines/bpops_shim.py` |
| DEPTHOR++ | ❌ unreleased | ❌ unreleased | [details](docs/metrics_protocol.md#baseline-availability) |
| `bicubic`, `nn_fill` | reference floors | — | anything |

DEPTHOR v1 reproduces its published ZJU-L5 table row to within 1.2% on the full
527-frame split (RMSE 0.3501 vs 0.350, Rel 0.0759 vs 0.075, δ1 0.9321 vs 0.933)
— see [metrics_protocol.md](docs/metrics_protocol.md#cross-check-against-published-numbers).
That row is the repo's regression anchor.

DEPTHOR++ is named in the SOW but the authors published neither code nor
weights. The config exists, is marked unavailable, and refuses to run rather
than quietly returning v1 numbers under a ++ label.

`BpOps`, the CUDA-only extension DEPTHOR needs, is supplied in portable PyTorch
by `baselines/bpops_shim.py` (checked against BP-Net's kernel). Runs are tagged
`bpops=shim` or `bpops=cuda`.

## House rules

1. **One metric implementation** — a second AbsRel or RMSE anywhere, notebooks
   included, is a blocking review comment.
2. **`third_party/` is read-only.** Differences go in an adapter, so a baseline
   number stays attributable to the authors' code at a known commit.
3. **No data or weights in git.** Splits are, and they're what keeps three sets
   of numbers comparable.
4. **One experiment = one YAML.** If it can't be re-created from a committed
   file, it isn't a result.
5. **Uncalibrated means labelled.** The Astra 2 noise model currently ships
   datasheet defaults, not a fit to real captures.

## Docs

* [docs/running.md](docs/running.md) — how to run a given model on given data, overrides, reading results
* [docs/datasets.md](docs/datasets.md) — external data folder, per-dataset layout, sample contract, splits
* [docs/metrics_protocol.md](docs/metrics_protocol.md) — metric definitions, FPS protocol, baseline availability
* [docs/design_doc.md](docs/design_doc.md) — the problem, architecture, Astra 2 noise model and its calibration
* [CONTRIBUTING.md](CONTRIBUTING.md) — branches, PR template, review rules
