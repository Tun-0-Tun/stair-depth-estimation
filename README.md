# stair-depth-estimation

Depth completion / super-resolution for the **Orbbec Astra 2**, specialised for
**staircases**. Joint SOW with Huawei.

Stairs break active stereo in a very specific way: every tread edge is a depth
discontinuity, which means an occlusion shadow next to it, and the holes line
up with exactly the geometry a robot needs. The repo holds our method, the two
required baselines, and a benchmark harness that scores all of them with the
same code.

## Setup

```bash
git clone --recurse-submodules <repo> && cd stair-depth-estimation
uv sync --group baselines     # drop the group if you only touch our own code
uv run pre-commit install
uv run pytest
uv run python scripts/run_baseline.py experiment=smoke
```

The smoke run goes through the whole pipeline — loader → Astra 2 degradation
model → method → metrics → results table — on a generated staircase. No
downloads, no weights. If it passes, your setup is fine.

Prefix commands with `uv run`, or activate `.venv` once and forget about it.

## What's here

```
data/loaders/       one file per dataset, all speaking the same sample format
data/degradation/   clean GT depth -> realistic sensor input (Astra 2, dToF, bicubic)
data/splits/        frozen train/val/test id lists — the only data in git
third_party/        DEPTHOR and DuCos as submodules, never edited
baselines/          adapters over third_party/*, one interface
models/             our method
metrics/            the metric code — accuracy and FPS
configs/            dataset / model / degradation / experiment YAML
scripts/            run_baseline, run_benchmark, train, download_data
tests/              metric correctness, loader contract, adapter smoke tests
```

## Data

```bash
uv run python scripts/download_data.py --list             # what's on disk
uv run python scripts/download_data.py --dataset zju_l5   # fetch, or print the manual steps
uv run python scripts/download_data.py --verify
```

Half the datasets need a form or an email to the authors. For those the script
prints where to go and where to put the files, and exits non-zero — it never
substitutes something fake. Layouts, units and gotchas per dataset:
[docs/datasets.md](docs/datasets.md).

Weights: `--weights depth_anything_v2_vits` (both baselines need it), `--weights
ducos`, `--weights depthor`.

## Running things

```bash
# can this even run here? checks deps, weights, CUDA extensions — loads nothing
uv run python scripts/run_baseline.py --model depthor --dataset zju_l5 --check

# one experiment file = one reproducible run
uv run python scripts/run_baseline.py experiment=ducos_rgbdd_real
uv run python scripts/run_baseline.py --model ducos --dataset nyuv2 --limit 10

# the matrix
uv run python scripts/run_benchmark.py --datasets zju_l5 hammer --models depthor ducos nn_fill

# training
uv run python scripts/train.py experiment=ours_stairs train.epochs=50
```

Each run appends a row to `experiments/results/benchmark.csv` (in git) and dumps
config, environment, per-sample metrics and timing into `experiments/runs/`.
A benchmark cell that couldn't run shows up as `NOT RUN: <reason>` rather than
disappearing.

## Baseline status

| | Code | Weights | Runs on |
|---|---|---|---|
| DuCos (ICCV 2025) | ✅ submodule | ✅ HuggingFace | anything — **verified end to end** |
| DEPTHOR v1 (ICCV 2025) | ✅ submodule | ✅ Google Drive, manual | **CUDA only** — needs `BpOps` (CUDA 12.1), so not on a Mac |
| DEPTHOR++ | ❌ unreleased | ❌ unreleased | [see the details](docs/metrics_protocol.md#baseline-availability) |
| `bicubic`, `nn_fill` | reference floors | — | anything |

DEPTHOR++ is named in the SOW but the authors have published neither code nor
weights. The config exists, is marked unavailable, and refuses to run rather
than quietly returning v1 numbers under a ++ label.

## House rules

1. **One metric implementation** — `metrics/depth_metrics.py` and
   `metrics/runtime_metrics.py`. A second AbsRel or RMSE anywhere, including a
   notebook, is a blocking review comment. The reason: the two baselines don't
   even agree on the units (DuCos reports RMSE in cm on normalised depth, and
   its "δ" thresholds are 1.05/1.15/1.25).
2. **`third_party/` is read-only.** Format differences, unit conversions and
   hard-coded paths go in an adapter, so a baseline number stays attributable to
   the authors' code at a known commit.
3. **No data or weights in git.** Splits are, and they're what keeps our three
   sets of numbers comparable.
4. **One experiment = one YAML.** If it can't be re-created from a committed
   file, it isn't a result.
5. **Uncalibrated means labelled.** The Astra 2 noise model currently ships
   datasheet defaults, not a fit to real captures; anything using it carries
   `uncalibrated` in the notes.

## Docs

* [docs/datasets.md](docs/datasets.md) — every dataset, how to get it, split protocol
* [docs/metrics_protocol.md](docs/metrics_protocol.md) — exact metric definitions, FPS protocol, baseline availability, cross-check against published numbers
* [docs/design_doc.md](docs/design_doc.md) — the problem, architecture, Astra 2 noise model and how to calibrate it, open questions
* [CONTRIBUTING.md](CONTRIBUTING.md) — branches, PR template, review rules
