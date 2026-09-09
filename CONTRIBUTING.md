# Contributing

Three people work in this repo. The rules below exist so that a number produced
by one of us means the same thing to the other two.

## Environment

```bash
git clone --recurse-submodules <repo> && cd stair-depth-estimation
uv sync --group baselines     # omit the group if you only touch our own code
uv run pre-commit install
uv run pytest
```

`uv.lock` is committed: everyone gets identical versions, and `uv sync` is the
only supported way to build the environment. If you add a dependency, edit
`pyproject.toml`, run `uv lock`, and commit both files in the same PR.

`pre-commit install` is not optional — it runs `ruff`, `black` and `isort` on
commit, so formatting never shows up in a diff you are trying to review.

DEPTHOR additionally needs the `BpOps` CUDA extension, which pip cannot
install; see [docs/metrics_protocol.md](docs/metrics_protocol.md#baseline-availability).

## Branches

* Work on `feat/<short-task>`, `fix/<short-task>`, `exp/<hypothesis>`.
  Examples: `feat/hypersim-loader`, `fix/zju-zone-centres`,
  `exp/mono-backbone-fusion`.
* Branch from `main`. Keep branches short-lived — a week is long.
* `main` is protected: changes land through a reviewed PR.
* Nobody force-pushes `main`.

## Submodules

`third_party/depthor` and `third_party/ducos` are submodules pinned to a
specific upstream commit.

* **Never edit anything inside `third_party/`.** If upstream code needs
  changing to work with our pipeline, that change belongs in
  `baselines/<name>_adapter.py`. A baseline number must be attributable to the
  authors' code at a known commit.
* Bumping a submodule pointer is its own PR, with a note saying what changed
  upstream and whether any published number moved.
* After pulling a branch that bumped one: `git submodule update --init --recursive`.

## Pull requests

Small and single-purpose. A loader, an adapter fix, and a new experiment are
three PRs.

**Every PR needs a review.** For a change that touches `metrics/`,
`data/loaders/base.py`, `baselines/base.py` or any split file, the reviewer must
be the person who did *not* write it — those files silently affect every number
in the report.

### PR description template

```markdown
## What
One or two sentences. What changed and why.

## How verified
- [ ] `pytest -q` passes
- [ ] `python scripts/run_baseline.py experiment=smoke` passes
- [ ] For a loader:   ran `pytest tests/test_loaders.py -k <dataset>` with real data on disk
- [ ] For an adapter: ran `python scripts/run_baseline.py --model <m> --dataset <d> --limit 10`
- [ ] For a metric change: added/updated an analytic test in tests/test_metrics.py

## Metrics
Paste the relevant rows. Before/after if this could move a number.

| dataset | method | AbsRel | RMSE | δ1 | FPS |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

Device + resolution for the FPS number: ...
Config used: configs/experiment/<name>.yaml (+ any overrides)

## Risk
What could this break that the tests do not cover?
```

A PR that changes any number in `experiments/results/benchmark.csv` must say so
explicitly and explain why the number moved.

## Where things go

| You are adding… | Put it in | Then |
|---|---|---|
| a dataset | `data/loaders/<name>.py`, `@register_dataset` | add `configs/dataset/<name>.yaml`, add its layout to `scripts/check_data.py` and `docs/datasets.md`, commit the split |
| a sensor model | `data/degradation/<name>.py`, `@register_degradation` | add `configs/degradation/<name>.yaml`, add a case to `test_degradation_contract` |
| a baseline | submodule in `third_party/`, adapter in `baselines/` | add `configs/model/<name>.yaml`, implement `availability()` |
| a model block | `models/modules/<name>.py` | make it switchable from `configs/model/ours.yaml` so it can be ablated |
| an experiment | `configs/experiment/<name>.yaml` | one file must fully describe the run |
| a metric | `metrics/depth_metrics.py` **only** | add an analytic test with the expected value derived by hand |
| exploration | `notebooks/` | nothing in a notebook is part of the pipeline; move it into a module before relying on it |

## Non-negotiable rules

These are blocking review comments, not preferences.

1. **One metric implementation.** `metrics/depth_metrics.py` and
   `metrics/runtime_metrics.py` are the only places AbsRel, RMSE, RMSE(log),
   δ1/δ2/δ3, MAE, SqRel, SILog, FPS and latency are computed. Not in a
   notebook, not in a training loop, not "just for a quick check". If you need
   something the module does not provide, add it there with a test.
   *Reviewer check: `grep -rn "sqrt(mean\|delta1\|abs_rel\|absrel" --include=*.py`
   outside `metrics/` and `tests/`.*
2. **`third_party/` is read-only.** See above.
3. **No data, weights or run artefacts in git.** Only `data/splits/*.json` and
   `experiments/results/*.csv|md`.
4. **Metric depth in metres, everywhere.** A loader converts on read; an adapter
   converts on the way in and back on the way out. If a tensor in
   `data/`, `metrics/` or `models/` is in millimetres or normalised units,
   that is a bug.
5. **Degradation is seeded per sample.** Never call `np.random` directly in a
   degradation model; use the `rng` passed to `apply()`.
6. **Report what you did not run.** A missing cell in a table gets a reason.
   Never delete a row because it failed.
7. **Uncalibrated models are labelled.** Anything using
   `astra2_uncalibrated.yaml` carries `uncalibrated` in the notes.

## Git hygiene

* Commit messages: `<area>: <what>` — e.g. `loaders: add hypersim, ray-distance
  to planar Z`, `metrics: clamp silog variance at zero`.
* One logical change per commit.
* Do not commit `experiments/runs/`; the summary CSV is enough.
* If you regenerate a split file, say why in the commit message — a changed
  split invalidates every previously reported number for that dataset.

## Before you claim a result

- [ ] It came from a committed `configs/experiment/*.yaml`
- [ ] `experiments/results/benchmark.csv` has the row
- [ ] The device and input resolution are recorded (FPS is meaningless without them)
- [ ] The split file is committed
- [ ] `n_skipped` is small — a large value means the mask or depth range is wrong
- [ ] If a degradation model was involved, its calibration status is stated
