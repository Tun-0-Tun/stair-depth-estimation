"""Hypersim -- 461 photorealistic synthetic indoor scenes (Apple, ICCV 2021).

https://github.com/apple-aiml-research/ml-hypersim

Why we want it: it is the only source here with *perfectly dense, noise-free*
ground truth.  Every real dataset we have either has holes (VOID, MinJiang) or
is the sensor being evaluated (MinJiang again).  Hypersim gives the degradation
models in ``data/degradation/`` a clean input, which is the whole point of
simulating an Astra 2 rather than borrowing someone else's noise.

Layout, after downloading with ``contrib/99991/download.py``::

    hypersim/
    ├── metadata_camera_parameters.csv        <- copy it here, see below
    └── ai_VVV_NNN/
        ├── _detail/cam_XX/metadata_camera.csv
        └── images/
            ├── scene_cam_XX_geometry_hdf5/frame.NNNN.depth_meters.hdf5
            ├── scene_cam_XX_geometry_hdf5/frame.NNNN.normal_cam.hdf5   (optional)
            └── scene_cam_XX_final_preview/frame.NNNN.color.jpg

457 scenes survive the authors' manual exclusions, 769 camera trajectories,
74 619 frames at 1024x768.  Frames inside one trajectory are ~97 views of the
same room, so they are heavily correlated -- subsample with ``frame_stride`` and
split by *scene*, never by frame.

.. warning::
   **``depth_meters`` is not planar depth.**  Upstream stores the Euclidean
   distance from the optical centre to the surface, so a pixel at the corner of
   the frame reads several per cent larger than its true Z.  We convert here,
   once, using that scene's own projection matrix -- see ``_planar_scale``.
   The widely copied ``focal = 886.81`` one-liner assumes every scene uses the
   default field of view; ``metadata_camera_parameters.csv`` shows they do not,
   which is why this loader reads the matrix instead of hard-coding a constant.

   That CSV ships with the upstream *repository*, not with the image download::

       cp ~/ml-hypersim/contrib/mikeroberts3000/metadata_camera_parameters.csv \\
          $STAIR_DATA_ROOT/hypersim/

Normals (``normal_cam``) are indexed when present and their path is passed
through in ``meta["normal_path"]``, but they are *not* part of the sample
contract -- adding an eighth key is a decision for the whole repo, not for one
loader.  Stairs are piecewise planar, so they are worth revisiting.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["HyperSimDataset"]

_CAMERA_PARAMS_CSV = "metadata_camera_parameters.csv"


def _planar_scale(m_cam_from_uv: np.ndarray, height: int, width: int) -> np.ndarray:
    """Per-pixel factor turning ray distance into planar Z.

    ``M_cam_from_uv`` maps normalised image coordinates (u, v, 1), both in
    [-1, 1] over the frame, to a direction in camera space.  For a ray of unit
    length the planar depth is the magnitude of that direction's z component, so
    ``planar = distance * |dir_z| / ||dir||``.  The factor is 1 on the optical
    axis and shrinks towards the corners, and it depends on u and v only through
    their squares -- so it is immune to getting the axis flip wrong.
    """
    u = np.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, dtype=np.float64)
    v = np.linspace(-1.0 + 1.0 / height, 1.0 - 1.0 / height, height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    uv1 = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)

    dirs = uv1 @ np.asarray(m_cam_from_uv, dtype=np.float64).T
    norm = np.linalg.norm(dirs, axis=-1)
    return (np.abs(dirs[..., 2]) / np.maximum(norm, 1e-12)).astype(np.float32)


@register_dataset("hypersim")
class HyperSimDataset(BaseDepthDataset):
    info = DatasetInfo(
        name="hypersim",
        modality="gt_only",
        min_depth=1e-3,
        max_depth=20.0,
        native_size=(768, 1024),
        sensor="path-traced render; ground truth is exact by construction",
        url="https://github.com/apple-aiml-research/ml-hypersim",
        notes="Synthetic. GT only: needs a degradation model. Depth is ray distance, converted here.",
    )

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        camera_params: str = _CAMERA_PARAMS_CSV,
        scenes: Sequence[str] | None = None,
        frame_stride: int = 1,
        load_normals: bool = False,
        **kwargs: Any,
    ) -> None:
        self.camera_params = camera_params
        self.scenes = list(scenes) if scenes else None
        self.frame_stride = max(1, int(frame_stride))
        self.load_normals = bool(load_normals)
        self._intrinsics: dict[str, np.ndarray] | None = None
        self._scale_cache: dict[tuple[str, int, int], np.ndarray] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------ camera

    def _load_intrinsics(self) -> dict[str, np.ndarray]:
        """scene name -> the 3x3 ``M_cam_from_uv`` for that scene."""
        if self._intrinsics is not None:
            return self._intrinsics

        path = self.root / self.camera_params
        if not path.exists():
            raise FileNotFoundError(
                f"[hypersim] {path} is missing. It ships with the upstream repository, not "
                "with the image download:\n"
                "  cp ~/ml-hypersim/contrib/mikeroberts3000/metadata_camera_parameters.csv "
                f"{self.root}/\n"
                "Without it the ray-distance to planar-Z conversion has no camera matrix, and "
                "depth would be silently a few per cent too large towards the frame corners."
            )

        out: dict[str, np.ndarray] = {}
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                name = row.get("scene_name") or row.get("scene") or next(iter(row.values()))
                try:
                    m = [float(row[f"M_cam_from_uv_{i}{j}"]) for i in range(3) for j in range(3)]
                except (KeyError, TypeError, ValueError):
                    continue
                out[name] = np.asarray(m, dtype=np.float64).reshape(3, 3)

        if not out:
            raise ValueError(
                f"[hypersim] {path} has no M_cam_from_uv_* columns; wrong file? "
                "Expected the upstream metadata_camera_parameters.csv."
            )
        self._intrinsics = out
        return out

    def _scale_for(self, scene: str, height: int, width: int) -> np.ndarray:
        """Cached per-scene conversion map -- identical for every frame of a scene."""
        key = (scene, height, width)
        if key not in self._scale_cache:
            intr = self._load_intrinsics()
            if scene not in intr:
                raise KeyError(
                    f"[hypersim] no camera parameters for scene {scene!r} in "
                    f"{self.root / self.camera_params}"
                )
            self._scale_cache[key] = _planar_scale(intr[scene], height, width)
        return self._scale_cache[key]

    # ------------------------------------------------------------- index

    def _build_index(self) -> Sequence[Any]:
        records: list[dict[str, Any]] = []
        scene_dirs = sorted(p for p in self.root.glob("ai_*") if p.is_dir())
        if self.scenes:
            wanted = set(self.scenes)
            scene_dirs = [p for p in scene_dirs if p.name in wanted]

        for scene in scene_dirs:
            for geom in sorted((scene / "images").glob("scene_cam_*_geometry_hdf5")):
                cam = geom.name.removeprefix("scene_cam_").removesuffix("_geometry_hdf5")
                preview = scene / "images" / f"scene_cam_{cam}_final_preview"
                for i, depth in enumerate(sorted(geom.glob("frame.*.depth_meters.hdf5"))):
                    if i % self.frame_stride:
                        continue
                    frame = depth.name.split(".")[1]
                    rgb = preview / f"frame.{frame}.color.jpg"
                    if not rgb.exists():
                        continue  # colour pass not downloaded for this frame
                    normal = geom / f"frame.{frame}.normal_cam.hdf5"
                    records.append(
                        {
                            "id": f"{scene.name}/cam_{cam}/{frame}",
                            "scene": scene.name,
                            "depth": depth,
                            "rgb": rgb,
                            "normal": normal if normal.exists() else None,
                        }
                    )

        if not records and self.strict:
            raise FileNotFoundError(
                f"[hypersim] no frames under {self.root}. Expected "
                "ai_VVV_NNN/images/scene_cam_XX_geometry_hdf5/frame.NNNN.depth_meters.hdf5 "
                "with a matching frame.NNNN.color.jpg in scene_cam_XX_final_preview -- "
                "see docs/datasets.md."
            )
        return records

    # -------------------------------------------------------------- read

    @staticmethod
    def _read_hdf5(path: Path) -> np.ndarray:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise ImportError("hypersim needs h5py: pip install h5py") from exc

        with h5py.File(path, "r") as f:
            # Upstream writes a single dataset per file; do not hard-code its name.
            key = next(iter(f.keys()))
            return np.asarray(f[key][:], dtype=np.float32)

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32) / 255.0
        dist = self._read_hdf5(record["depth"])

        h, w = dist.shape[:2]
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"[hypersim] {record['id']}: rgb {rgb.shape[:2]} != depth {(h, w)}")

        # Ray distance -> planar Z, with this scene's own projection matrix.
        depth = dist * self._scale_for(record["scene"], h, w)
        # Pixels with no geometry behind them are NaN upstream; 0 means "no
        # measurement" in our contract, and the mask drops them either way.
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        meta: dict[str, Any] = {
            "scene": record["scene"],
            "depth_unit": "metres (planar Z, converted from ray distance)",
        }
        if record["normal"] is not None:
            # Not part of the sample contract; carried as a path so a training
            # loop can pick it up without every loader growing an eighth key.
            meta["normal_path"] = str(record["normal"])

        return {
            "rgb": rgb,
            "gt_depth": depth.astype(np.float32),
            "sample_id": record["id"],
            "rgb_path": str(record["rgb"]),
            "depth_path": str(record["depth"]),
            "meta": meta,
        }
