"""A' at init == WAVE (max |diff| < 1e-4, fp32), and the fusion variants are 0.5/0.5 at init."""

import sys

import pytest
import torch

pytest.importorskip("pytorch_wavelets")
from models.wave_aprime import FUSIONS, SelectiveFusion, WaveAPrime, load_wave_checkpoint  # noqa: E402

Wave = sys.modules["Components.WAVE"].WAVE


@pytest.fixture(scope="module")
def wave_and_ckpt(tmp_path_factory):
    torch.manual_seed(0)
    w = Wave(use_pretrained=False, num_feats=48, patch_size=16, scale=8).eval()
    w.semantics_encoder.init_weights()
    p = tmp_path_factory.mktemp("ck") / "w.pth"
    torch.save({"model": w.state_dict()}, p)
    return w, p


@pytest.mark.parametrize("fusion", FUSIONS)
def test_equivalence_at_start(wave_and_ckpt, fusion):
    w, p = wave_and_ckpt
    a = WaveAPrime(scale=8, use_validity_channel=True, fusion=fusion).eval()
    rep = load_wave_checkpoint(a, p, log=lambda *_: None)
    assert (fusion == "none") == (not rep["new"])
    torch.manual_seed(1)
    img, d = torch.rand(1, 3, 128, 176), torch.rand(1, 1, 16, 22)
    valid = (torch.rand(1, 1, 16, 22) > 0.4).float()  # arbitrary mask: its weights are 0
    with torch.no_grad():
        diff = (w(img, d) - a(img, d, valid)).abs().max().item()
    assert diff < 1e-4, diff


@pytest.mark.parametrize("kind", FUSIONS[1:])
def test_fusion_init_is_identity(kind):
    f = SelectiveFusion(kind, 32).eval()
    x, y = torch.randn(2, 32, 8, 8), torch.randn(2, 32, 8, 8)
    a, b = f(x, y, torch.ones(2, 1, 8, 8))
    assert torch.allclose(a, x) and torch.allclose(b, y)


def test_upstream_selector_weights_match():
    """Our weights == upstream SEmoudle softmax weights when given the same parameters."""
    import importlib.util

    from utils.misc import REPO_ROOT

    src = REPO_ROOT / "third_party/StairNet-DepthIn/nets/modules.py"
    if not src.exists():
        pytest.skip("StairNet-DepthIn not cloned")
    spec = importlib.util.spec_from_file_location("sn_in_modules", src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    up = m.SEmoudle(32, 32).eval()
    torch.nn.init.normal_(up.fc2.weight, std=0.3)
    ours = SelectiveFusion("stairnetv2_orig", 32).eval()
    ours.fc1.load_state_dict(up.fc1.state_dict())
    ours.fc2.load_state_dict(up.fc2.state_dict())
    x, y = torch.randn(2, 32, 6, 6), torch.randn(2, 32, 6, 6)
    a, b = ours.weights(x, y, None)[:, 0], ours.weights(x, y, None)[:, 1]
    assert torch.allclose(up(x, y), a * x + b * y, atol=1e-6)
