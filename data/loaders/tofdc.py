"""TOFDC / TOFDSR -- real smartphone dToF with high-quality reference depth.

STATUS: stub.  OWNER: Maxim.

Why it matters: DuCos's second real-world benchmark (``test_RealTOFDSR.py``), so
it is a cross-check dataset for the DuCos adapter.  The file lists DuCos used
are already in the repo we vendored:
``third_party/ducos/data/TOFDC_Filled_Train.txt`` and ``TOFDC_Filled_Test.txt``
-- read the paths from there so our split is bit-identical to theirs.

Gotchas:

* Depth PNGs are uint16 in millimetres; the "filled" variant has holes
  inpainted, the raw one does not. DuCos uses the filled one.
* Their loader min-max normalises per sample; our contract is metres, so
  normalisation belongs in ``baselines/ducos_adapter.py``, not here.
"""

from __future__ import annotations

from data.loaders._stub import StubLoaderMixin
from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["TOFDCDataset"]


@register_dataset("tofdc")
class TOFDCDataset(StubLoaderMixin, BaseDepthDataset):
    info = DatasetInfo(
        name="tofdc",
        modality="lr",
        min_depth=0.1,
        max_depth=10.0,
        depth_unit_per_metre=1000.0,
        sensor="smartphone dToF + RGB, structure-light reference",
        url="https://yanzq95.github.io/projectpage/TOFDC/index.html",
        notes="Use third_party/ducos/data/TOFDC_Filled_*.txt as the split lists.",
    )
    todo = (
        "Parse third_party/ducos/data/TOFDC_Filled_Test.txt for the (rgb, lr, gt) triples.",
        "Load uint16 mm PNGs and divide by 1000.",
        "Copy the parsed id list into data/splits/tofdc/test.json and commit it.",
    )
