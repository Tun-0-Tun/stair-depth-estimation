"""Unified evaluation metrics.

Import accuracy metrics from :mod:`metrics.depth_metrics` and timing from
:mod:`metrics.runtime_metrics`.  Never re-implement either.
"""

from metrics.depth_metrics import (
    LOWER_IS_BETTER,
    METRIC_NAMES,
    PRIMARY_METRICS,
    DepthMetricAccumulator,
    MetricConfig,
    compute_depth_metrics,
    format_metrics,
)
from metrics.runtime_metrics import RuntimeConfig, describe_device, measure_runtime

__all__ = [
    "LOWER_IS_BETTER",
    "METRIC_NAMES",
    "PRIMARY_METRICS",
    "DepthMetricAccumulator",
    "MetricConfig",
    "RuntimeConfig",
    "compute_depth_metrics",
    "describe_device",
    "format_metrics",
    "measure_runtime",
]
