# Splits

Frozen train/val/test id lists. **These are the only data files in git** — they
are a few KB each, and they are what makes three people's numbers comparable.

Format, `data/splits/<dataset>/<split>.json`:

```json
{"dataset": "zju_l5", "split": "test", "ids": ["theater/1645696174.476698.h5", "..."]}
```

The ids are whatever `BaseDepthDataset.record_id` returns for that loader —
usually a path relative to the dataset root. A loader given `split_file=` keeps
only those ids, **in that order**, and errors if any are missing from disk
rather than silently evaluating on a subset.

## Making one

```python
ds = build_dataset({"loader": "zju_l5", "root": "data/raw/ZJUL5", "split": "test"})
ds.write_split_file("data/splits/zju_l5/test.json")
```

## Rules

* Prefer the authors' own split when one exists, so our numbers stay comparable
  with the published tables — see [../../docs/datasets.md](../../docs/datasets.md#split-protocol).
* **Never split sequential data by frame.** Consecutive frames of one staircase
  are near-duplicates; a random frame split leaks test into train.
  Split by scene / video / capture session.
* Changing a split invalidates every number previously reported on that dataset.
  Say so in the commit message and re-run the affected rows.
