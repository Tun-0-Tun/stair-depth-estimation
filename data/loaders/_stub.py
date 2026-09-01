"""Helper for loaders that are specified but not implemented yet.

A stub is a *contract*, not a placeholder: it registers the dataset key, carries
the real ``DatasetInfo`` (units, ranges, sensor, URL) and lists exactly what the
implementer has to do.  Configs and tests can already reference it; the failure
when you try to use it names the file to edit and the checklist to follow.

To implement one: delete ``StubLoaderMixin`` from the bases and write
``_build_index`` / ``_load_raw`` following ``data/loaders/nyuv2.py``, then add
the dataset to the parametrised smoke test in ``tests/test_loaders.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["NotImplementedDatasetError", "StubLoaderMixin"]


class NotImplementedDatasetError(NotImplementedError):
    """Raised when a stub loader is instantiated."""


class StubLoaderMixin:
    """Mix in *before* ``BaseDepthDataset`` to turn a loader into a stub."""

    #: Free-form checklist shown in the error message.
    todo: tuple[str, ...] = ()

    def _build_index(self) -> Sequence[Any]:
        cls = type(self)
        module = cls.__module__.replace(".", "/") + ".py"
        steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(self.todo, 1)) or "  (no notes yet)"
        raise NotImplementedDatasetError(
            f"{cls.__name__} is a stub. Implement it in {module}.\n"
            f"Dataset: {self.info.name} <{self.info.url}>\n"
            f"TODO:\n{steps}\n"
            "Contract: data/loaders/base.py :: BaseDepthDataset. "
            "Reference implementation: data/loaders/nyuv2.py."
        )

    def _load_raw(self, record: Any) -> Mapping[str, Any]:
        raise NotImplementedDatasetError(f"{type(self).__name__} is a stub")
