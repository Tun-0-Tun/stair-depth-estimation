# Notebooks

Exploration and visualisation only — error maps, qualitative comparisons,
looking at a new dataset for the first time.

Nothing here is part of the pipeline. Two consequences:

* Anything a result depends on gets moved into a module first. A number that
  only exists in a notebook is not a result.
* The metric rule still applies: `from metrics.depth_metrics import
  compute_depth_metrics`, never a quick reimplementation in a cell.

Clear outputs before committing (`pre-commit` does not touch notebooks, so this
one is on you).
