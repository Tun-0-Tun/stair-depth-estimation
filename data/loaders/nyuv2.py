"""NYU Depth v2.

Three on-disk layouts are in circulation, so this loader reads all of them:

``layout="mat"`` -- the official labeled pack, ``nyu_depth_v2_labeled.mat``: one
    MATLAB v7.3 (= HDF5) file with all 1449 labeled samples.  Keys ``images``
    (uint8), ``depths`` (float metres, Levin-filled) and ``rawDepths`` (the
    unfilled Kinect depth, holes as 0).  MATLAB stores column-major, so h5py
    hands back reversed axes and we transpose on the way out.  There is no
    official train/test split in this file -- every sample is returned, which
    is fine for benchmarking but means it must never be used for training.

``layout="h5"`` -- the depth-*completion* convention (NLSPN / CompletionFormer /
    Deltar / DEPTHOR).  One HDF5 per sample with datasets ``rgb`` (3, H, W or
    H, W, 3, uint8) and ``depth`` (H, W, float32 metres).  Directory tree
    ``<root>/nyudepthv2/{train,val}/<scene>/*.h5``.

``layout="npy"`` -- the depth-*super-resolution* convention (DKN / FDSR / DuCos).
    ``test_images_v2.npy`` (N, H, W, 3 uint8) and ``test_depth.npy`` (N, H, W
    float, min-max normalised **per image**), plus ``test_minmax.npy``
    (2, N) holding ``(max, min)`` in metres.  We denormalise back to metres here
    so that the rest of the repo only ever sees metric depth -- see
    ``docs/metrics_protocol.md`` for why that matters when comparing against
    DuCos's published RMSE, which is in centimetres on the normalised scale.

Standard eval crop: the Eigen crop (``MetricConfig(crop="eigen")``) for
completion-style comparisons, the 6-pixel border crop for DSR-style ones.
Set it in ``configs/dataset/nyuv2.yaml``, never in code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["NYUv2Dataset"]


@register_dataset("nyuv2")
class NYUv2Dataset(BaseDepthDataset):
    info = DatasetInfo(
        name="nyuv2",
        modality="gt_only",
        min_depth=1e-3,
        max_depth=10.0,
        native_size=(480, 640),
        sensor="Kinect v1 (structured light), Levin-filled GT",
        url="https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html",
        notes="GT only: needs a degradation model to produce the network input.",
    )

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        layout: str = "h5",
        minmax_file: str | None = "test_minmax.npy",
        mat_file: str = "nyu_depth_v2_labeled.mat",
        **kwargs: Any,
    ) -> None:
        if layout not in ("h5", "npy", "mat"):
            raise ValueError(f"layout must be 'h5', 'npy' or 'mat', got {layout!r}")
        self.layout = layout
        self.minmax_file = minmax_file
        self.mat_file = mat_file
        self._npy_cache: dict[str, np.ndarray] = {}
        self._mat_handle: Any = None
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------

    def _build_index(self) -> Sequence[Any]:
        if self.layout == "mat":
            n = self._mat()["depths"].shape[0]
            return [{"id": f"nyu_{i:04d}", "index": i} for i in range(n)]

        if self.layout == "h5":
            subdir = {"train": "train", "val": "val", "test": "val"}.get(self.split, self.split)
            base = self.root / "nyudepthv2" / subdir
            if not base.exists():
                base = self.root / subdir
            files = sorted(base.rglob("*.h5"))
            if not files and self.strict:
                raise FileNotFoundError(
                    f"[nyuv2] no .h5 under {base}. Expected "
                    "<root>/nyudepthv2/{train,val}/<scene>/*.h5 -- see docs/datasets.md."
                )
            return [{"id": str(p.relative_to(base)), "path": p} for p in files]

        images = self._load_npy("test_images_v2.npy")
        n = 0 if images is None else images.shape[0]
        return [{"id": f"nyu_{i:04d}", "index": i} for i in range(n)]

    def _load_npy(self, name: str) -> np.ndarray | None:
        if name not in self._npy_cache:
            path = self.root / name
            if not path.exists():
                if self.strict:
                    raise FileNotFoundError(
                        f"[nyuv2] layout='npy' needs {path}. Download the DSR-format NYU pack "
                        "(see docs/datasets.md) or switch to layout='h5'."
                    )
                return None
            self._npy_cache[name] = np.load(path)
        return self._npy_cache[name]

    def _mat(self) -> Any:
        """The open ``nyu_depth_v2_labeled.mat`` handle (~2.8 GB, read lazily)."""
        # ponytail: one handle held open for the dataset's lifetime. Fine for a
        # single-process run; give each worker its own handle if we ever set
        # num_workers > 0, because an h5py file does not survive a fork.
        if self._mat_handle is None:
            try:
                import h5py
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "layout='mat' needs h5py: pip install h5py (it is in requirements.txt)"
                ) from exc

            path = self.root / self.mat_file
            if not path.exists():
                raise FileNotFoundError(
                    f"[nyuv2] layout='mat' needs {path}. Download the labeled pack "
                    "(see docs/datasets.md) or switch to layout='h5' / 'npy'."
                )
            self._mat_handle = h5py.File(path, "r")
        return self._mat_handle

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.layout == "mat":
            return self._load_mat_sample(record)
        if self.layout == "h5":
            return self._load_h5(record)
        return self._load_npy_sample(record)

    def _load_mat_sample(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        i = int(record["index"])
        f = self._mat()

        # MATLAB is column-major: h5py sees images as (N, 3, W, H) and depths
        # as (N, W, H). Transpose back to (H, W, 3) / (H, W).
        rgb = np.asarray(f["images"][i]).transpose(2, 1, 0).astype(np.float32) / 255.0
        depth = np.asarray(f["depths"][i], dtype=np.float32).T

        out: dict[str, Any] = {
            "rgb": rgb,
            "gt_depth": depth,
            "sample_id": record["id"],
            "rgb_path": str(self.root / self.mat_file),
            "depth_path": str(self.root / self.mat_file),
        }
        if "rawDepths" in f:
            # Same deal as the h5 pack's 'raw': unfilled Kinect depth, a real
            # sparse input rather than a simulated one.
            out["sparse_depth"] = np.asarray(f["rawDepths"][i], dtype=np.float32).T
            out["meta"] = {"sparse_source": "kinect_raw"}
        return out

    def _load_h5(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "layout='h5' needs h5py: pip install h5py (it is in requirements.txt)"
            ) from exc

        with h5py.File(record["path"], "r") as f:
            rgb = np.array(f["rgb"])
            depth = np.array(f["depth"], dtype=np.float32)
            raw = np.array(f["raw"], dtype=np.float32) if "raw" in f else None

        if rgb.ndim == 3 and rgb.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
            rgb = rgb.transpose(1, 2, 0)
        rgb = rgb.astype(np.float32) / 255.0

        out: dict[str, Any] = {
            "rgb": rgb,
            "gt_depth": depth,
            "sample_id": record["id"],
            "rgb_path": str(record["path"]),
            "depth_path": str(record["path"]),
        }
        if raw is not None:
            # 'raw' is the unfilled Kinect depth: a genuine sparse sensor input,
            # far more honest than any simulation. Use it when present.
            out["sparse_depth"] = raw
            out["meta"] = {"sparse_source": "kinect_raw"}
        return out

    def _load_npy_sample(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        i = int(record["index"])
        images = self._load_npy("test_images_v2.npy")
        depths = self._load_npy("test_depth.npy")
        rgb = images[i].astype(np.float32) / 255.0
        depth = np.asarray(depths[i], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]

        minmax = self._load_npy(self.minmax_file) if self.minmax_file else None
        if minmax is not None:
            dmax, dmin = float(minmax[0, i]), float(minmax[1, i])
            depth = depth * (dmax - dmin) + dmin
            unit = "metres (denormalised via test_minmax.npy)"
        else:
            unit = "UNKNOWN - test_minmax.npy missing, depth left normalised"
            if self.strict:
                raise FileNotFoundError(
                    "[nyuv2] layout='npy' without test_minmax.npy would leave depth in "
                    "normalised [0,1] units, which silently corrupts every metric. "
                    "Provide the file or set minmax_file=null and strict=false knowing why."
                )
        return {
            "rgb": rgb,
            "gt_depth": depth,
            "sample_id": record["id"],
            "meta": {"depth_unit": unit},
        }
