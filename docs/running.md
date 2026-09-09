# Running things

How to evaluate a given method on given data, and how to read what comes out.

Prerequisites: `uv sync --group baselines`, and `STAIR_DATA_ROOT` pointing at
the shared dataset folder ([datasets.md](datasets.md)). Prefix everything with
`uv run`, or activate `.venv` once.

---

## The short version

```bash
# 1. can this even run here? checks deps, weights, CUDA extensions - loads nothing
uv run python scripts/run_baseline.py --model depthor --dataset zju_l5 --check

# 2. run it
uv run python scripts/run_baseline.py --model depthor --dataset zju_l5 \
    model.checkpoint=checkpoints/depthor/depthor_zju_large.pt
```

Always start with `--check`. It tells you exactly which package, checkpoint or
CUDA extension is missing and the command that fixes it, instead of failing
twenty minutes into a run.

## Two ways to specify a run

**By experiment file** — the reproducible one. One YAML fully describes
dataset + model + degradation + seed + evaluation protocol:

```bash
uv run python scripts/run_baseline.py experiment=depthor_zju_l5
```

Use this for anything you will report. If a number cannot be re-created from a
committed file, it is not a result.

**By flags** — for poking around:

```bash
uv run python scripts/run_baseline.py --model ducos --dataset void_stairs --limit 10
```

Both forms accept the same overrides, so you can start from an experiment file
and change one thing:

```bash
uv run python scripts/run_baseline.py experiment=depthor_zju_l5 eval.max_samples=20
```

## Choosing the pieces

| Group | Where the options live | List them |
|---|---|---|
| model | `configs/model/*.yaml` | `bicubic` `nn_fill` `ducos` `depthor` `depthor_plus_plus` `ours` |
| dataset | `configs/dataset/*.yaml` | `void_stairs` `minjiang` `zju_l5` `nyuv2` `hammer` `synthetic_stairs` |
| degradation | `configs/degradation/*.yaml` | `none` `identity` `astra2_uncalibrated` `dtof_l5` `bicubic_x4` `bicubic_x8` `random_sparse_500` |
| experiment | `configs/experiment/*.yaml` | `smoke` `depthor_zju_l5` `ducos_nyuv2_x4` |

**When do I need `--degradation`?** Only for datasets that ship ground truth
but no sensor input — `minjiang`, `nyuv2`, `synthetic_stairs`. They will refuse
to run without one. `void_stairs`, `zju_l5` and `hammer` carry a real sensor
input and ignore it. `scripts/check_data.py` and the dataset table in
[datasets.md](datasets.md) say which is which.

## Overrides

Any config key, with dots:

```bash
uv run python scripts/run_baseline.py --model ducos --dataset zju_l5 \
    model.checkpoint=checkpoints/ducos/x4.pth.tar \
    dataset.footprint=zone \
    eval.max_samples=50 eval.crop=border6 \
    runtime.iters=10 \
    seed=1
```

Values are parsed as YAML, so `null`, `3`, `true`, `[1,2]` all work.

> ⚠ **A group override replaces the whole group.** `dataset=void_stairs` wipes
> every `dataset.*` you set before it. The launcher applies `--dataset/--model/
> --degradation` first so the common case is safe, but if you write both forms
> as bare overrides, put the group first:
> `dataset=void_stairs dataset.root=/elsewhere` — not the other way round.

The keys worth knowing:

| Key | Meaning |
|---|---|
| `model.checkpoint` | weights to load; `--check` verifies the path exists |
| `dataset.root` | absolute path overrides `STAIR_DATA_ROOT`, for a one-off |
| `dataset.split` | `train` / `val` / `test` |
| `eval.max_samples` | same as `--limit` |
| `eval.min_depth` / `eval.max_depth` | metric range in metres; defaults to the dataset's |
| `eval.crop` | `none` / `eigen` / `border6` — see [metrics_protocol.md](metrics_protocol.md#crops) |
| `eval.pooling` | `image` (report this) / `pixel` (ablations only) |
| `runtime.measure` | `false`, or `--no-runtime`, to skip the FPS pass |
| `device` | `auto` / `cpu` / `cuda` / `mps` |
| `seed` | seeds the degradation model too, so the input is reproducible |
| `tracking.backend` | `none` / `wandb` / `mlflow` |

## Worked examples

```bash
# The regression anchor: DEPTHOR reproducing its published ZJU-L5 row.
uv run python scripts/run_baseline.py experiment=depthor_zju_l5

# DuCos on real stair data, 20 frames, on the GPU.
uv run python scripts/run_baseline.py --model ducos --dataset void_stairs \
    --device cuda --limit 20 model.checkpoint=checkpoints/ducos/x4.pth.tar

# What does our Astra 2 simulation do to a method? Same data, two sensor models.
uv run python scripts/run_baseline.py --model nn_fill --dataset minjiang --degradation dtof_l5
uv run python scripts/run_baseline.py --model nn_fill --dataset minjiang --degradation astra2_uncalibrated

# Sanity anchor: input == ground truth, so every metric must come out perfect.
uv run python scripts/run_baseline.py --model nn_fill --dataset minjiang --degradation identity

# A dataset that lives outside the shared folder.
uv run python scripts/run_baseline.py --model depthor --dataset void_stairs \
    dataset.root=/Users/me/Downloads/void_1500
```

## Several methods at once

```bash
# matrix: every dataset x every model
uv run python scripts/run_benchmark.py --datasets void_stairs zju_l5 \
    --models depthor ducos nn_fill bicubic --limit 50

# or a list of experiment files, each with its own protocol
uv run python scripts/run_benchmark.py --experiments depthor_zju_l5 ducos_nyuv2_x4

uv run python scripts/run_benchmark.py --dry-run    # print the matrix, run nothing
uv run python scripts/run_benchmark.py --auto       # skip cells that cannot run here
```

Writes `experiments/results/table_<stamp>.{md,csv}` — dataset × method × metric.
Pick the columns with `--metrics absrel rmse delta1 fps`.

A cell that could not run appears as `NOT RUN: <reason>`, never as a blank.
"We did not run it" and "it scored badly" must not look the same in a report.

## Training

```bash
uv run python scripts/train.py --dataset minjiang --degradation astra2_uncalibrated \
    train.epochs=50 train.batch_size=8 train.lr=0.001
```

Same config machinery. Validation uses the same `metrics/` code as the
benchmark, so a training curve and a benchmark row are directly comparable.
Checkpoints go to `checkpoints/ours/{best,last}.pt`.

## What you get back

```
experiments/results/benchmark.csv         one appended row per run - IN GIT
experiments/runs/<stamp>_<experiment>/
    config.json      the full composed config + environment + git commit
    summary.json     the aggregate metrics
    per_sample.csv   per-frame metrics, for error analysis
```

`benchmark.csv` is the shared record: if your number is not in it, it did not
happen. `experiments/runs/` is **not** in git — regenerate it by re-running the
experiment file.

Reading a row:

* `n_skipped` should be near zero. A large value means frames had no valid
  ground truth — usually a wrong `eval.min_depth`/`max_depth` or a broken mask.
* `git_commit` ends in `-dirty` if the tree had uncommitted changes. Do not cite
  a dirty row; it is not reproducible.
* `fps` is only meaningful together with the device and input resolution, both
  recorded in `config.json`. Never compare FPS across machines.
* `metric_config` is the evaluation protocol as JSON. Two rows with different
  `metric_config` are not comparable.

## When something goes wrong

| Symptom | Cause |
|---|---|
| `dataset root does not exist` | `STAIR_DATA_ROOT` unset or the folder is named differently — `python scripts/check_data.py` |
| `provides ground truth only, so it needs a degradation model` | add `--degradation` |
| `[<model>] cannot run:` | run with `--check`; it lists the missing pieces and the fix |
| `checkpoint ... is missing N tensors` | wrong checkpoint for that variant/scale |
| every metric is `nan` | no valid ground-truth pixels — check the depth range |
| an override seems ignored | group-override ordering, see the warning above |
| DEPTHOR prints `bpops=shim` | expected off CUDA; see [metrics_protocol.md](metrics_protocol.md#depthor-v1--runnable-anywhere-via-a-shim) |

## Adding your own experiment

Copy the closest file in `configs/experiment/`, change what you need, commit it:

```yaml
defaults:
  - dataset: void_stairs
  - model: ducos
  - degradation: none

seed: 0
device: auto
model:
  checkpoint: checkpoints/ducos/x4.pth.tar
eval:
  crop: none
  pooling: image
  min_depth: 0.2
  max_depth: 10.0
runtime:
  measure: true
  warmup: 10
  iters: 50
notes: "what this run is for"
```

One file must fully describe the run — that is the whole point.
