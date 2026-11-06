"""A': WAVE + validity channel + selective fusion.   WAVE is imported (third_party/wave), not copied.

THE ONLY PATCH OF WAVE IS IN THIS FILE.  Why a patch at all: WAVE's ``forward`` and
``StructureBlock.forward`` have no hook where the two modalities meet, so both are re-stated
below, line for line, plus the three additions marked ``# A'``:

1. validity channel -- ``init_depth_projection``'s first conv gets 256 extra input channels
   (1 mask channel x PixelUnshuffle(16)^2), zero-initialised, so the checkpoint behaves as before;
2. selective fusion in front of ``irn_fusion``.  WAVE fuses by *concatenation*
   (``irn_fusion(cat([encoded_depth, encoded_rgb]))``), not by a sum, so the selector returns
   ``(2*a_d*F_d, 2*a_rgb*F_rgb)``: with a == 0.5/0.5 (zero-initialised logits) the block input is
   bit-identical to WAVE's;
3. the validity mask is passed to the fusion (``spatial``).

The RGB branch (ML-DWT + DINO) and the decoder are untouched.

Fusion kinds (``fusion=``):
  none, stairnetv2_orig, channel, spatial.
``stairnetv2_orig`` is the StairNet-DepthIn ``SEmoudle`` selector (Wang et al., Sensors 2023;
third_party/StairNet-DepthIn/nets/modules.py): U = x1 + x2 -> GAP -> 1x1 conv (no bias) + BN +
ReLU -> 1x1 conv (no bias) to 2C -> softmax over the modalities.  Upstream *sums* the weighted
inputs; here the weights are applied to the two inputs separately (see above).  BN needs
batch >= 2 in training.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.base import add_third_party_to_path

add_third_party_to_path("wave")
from Components.ConvOps import ConvOps  # noqa: E402
from Components.StructureBlock import StructureBlock  # noqa: E402
from Components.WAVE import WAVE  # noqa: E402

FUSIONS = ("none", "stairnetv2_orig", "channel", "spatial")
_BLOCKS = {1: "structure_block1", 2: "structure_block2", 3: "structure_block3", 4: "structure_block4"}


class SelectiveFusion(nn.Module):
    """``(F_d, F_rgb, valid) -> (2 a_d F_d, 2 a_rgb F_rgb)``; logits layer zero-initialised."""

    def __init__(self, kind: str, channels: int) -> None:
        super().__init__()
        if kind not in FUSIONS[1:]:
            raise ValueError(f"fusion kind must be one of {FUSIONS[1:]}, got {kind!r}")
        self.kind, c = kind, channels
        if kind == "stairnetv2_orig":
            d = max(c, 32)
            self.fc1 = nn.Sequential(nn.Conv2d(c, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(True))
            self.fc2 = nn.Conv2d(d, 2 * c, 1, bias=False)
            last = self.fc2
        elif kind == "channel":
            self.mlp = nn.Sequential(nn.Linear(c, max(c // 4, 16)), nn.ReLU(True), nn.Linear(max(c // 4, 16), 2 * c))
            last = self.mlp[-1]
        else:  # spatial
            self.proj = nn.Conv2d(2 * c + 1, 2, 1)
            last = self.proj
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def weights(self, f_d, f_rgb, valid):
        b, c = f_d.shape[:2]
        if self.kind == "stairnetv2_orig":
            z = self.fc1(F.adaptive_avg_pool2d(f_d + f_rgb, 1))
            a = self.fc2(z).reshape(b, 2, c, 1, 1)
        elif self.kind == "channel":
            a = self.mlp(f_d.add(f_rgb).mean((2, 3))).reshape(b, 2, c, 1, 1)
        else:
            if valid is None:
                valid = f_d.new_ones(b, 1, *f_d.shape[-2:])
            v = F.interpolate(valid, size=f_d.shape[-2:], mode="area")
            a = self.proj(torch.cat([f_rgb, f_d, v], 1)).unsqueeze(2)  # [B, 2, 1, H, W]
        return torch.softmax(a, dim=1)

    def forward(self, f_d, f_rgb, valid=None):
        a = self.weights(f_d, f_rgb, valid)
        return 2.0 * a[:, 0] * f_d, 2.0 * a[:, 1] * f_rgb


class FusedStructureBlock(StructureBlock):
    """``StructureBlock.forward`` re-stated with the A' fusion hook (class-swapped in place)."""

    fusion: nn.Module | None = None

    def forward(self, token, ll, depth_feat, valid=None):
        token = F.interpolate(
            token, size=(depth_feat.shape[-2], depth_feat.shape[-1]), mode="bilinear", align_corners=False
        )
        s_guide = self.dino_projector(token)
        encoded_rgb = self.rgb_encoder(self.init_rgb_projection(ll))
        depth_semantics = ConvOps.compute_linear_attention(depth_feat, s_guide, s_guide)
        encoded_depth = self.depth_enc(depth_feat) + self.alpha * depth_semantics
        if self.fusion is not None:  # A'
            encoded_depth, encoded_rgb = self.fusion(encoded_depth, encoded_rgb, valid)
        return self.irn_fusion(torch.cat([encoded_depth, encoded_rgb], dim=1))


class WaveAPrime(WAVE):
    def __init__(
        self,
        scale: int = 8,
        num_feats: int = 48,
        patch_size: int = 16,
        use_validity_channel: bool = True,
        fusion: str = "none",
        fusion_levels: tuple[int, ...] = (3, 4),
        **kw: Any,
    ) -> None:
        super().__init__(use_pretrained=False, num_feats=num_feats, patch_size=patch_size, scale=scale, **kw)
        self.scale, self.use_validity_channel, self.fusion_kind = scale, use_validity_channel, fusion
        self.semantics_encoder.init_weights()  # WAVE leaves DINO uninitialised (NaN) until a checkpoint loads
        if use_validity_channel:  # A' (1)
            conv = self.init_depth_projection.net[1]
            new = nn.Conv2d(2 * 16 * 16, conv.out_channels, 3, padding=1)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, : conv.in_channels] = conv.weight
                new.bias.copy_(conv.bias)
            self.init_depth_projection.net[1] = new
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
        for lvl in fusion_levels if fusion != "none" else ():  # A' (2)
            blk = getattr(self, _BLOCKS[lvl])
            c = blk.irn_fusion[1].out_channels
            blk.__class__ = FusedStructureBlock
            blk.fusion = SelectiveFusion(fusion, c)

    def forward(self, image, depth, valid=None):
        """``depth``: filled LR depth (normalised), ``valid``: LR validity [B,1,h,w] (1 = measured)."""
        H, W = image.shape[-2:]
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        depth_bicubic = self.bicubic(depth)
        valid_hr = None
        if self.use_validity_channel or self.fusion_kind == "spatial":  # A' (3)
            if valid is None:
                valid = torch.ones_like(depth)
            valid_hr = F.interpolate(valid, size=depth_bicubic.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([depth_bicubic, valid_hr], 1) if self.use_validity_channel else depth_bicubic
        depth0 = self.init_depth_projection(x)

        features = self.semantics_encoder.get_intermediate_layers(
            image, n=self.intermediate_layer_idx, return_class_token=True
        )
        LLs, LHs, HLs, HHs = self.dwt_full_pyramid(image)

        token1 = self.stora1(features[3][0], patch_h, patch_w)
        details1 = self.deatil_block1(token1, LHs[3], HLs[3], HHs[3])
        depth1 = self._structure(self.structure_block1, token1, LLs[3], depth0, valid_hr)
        depth1 = self.injectors1(depth1, details1)
        res1 = self.residual1(self.upsampler1(depth1))

        token2 = self.stora2(features[2][0], patch_h, patch_w)
        details2 = self.deatil_block2(token2, LHs[2], HLs[2], HHs[2])
        depth2 = self._structure(self.structure_block2, token2, LLs[2], res1, valid_hr)
        depth2 = self.injectors2(depth2, details2)
        res2 = self.residual2(self.upsampler2(depth2))

        token3 = self.stora3(features[1][0], patch_h, patch_w)
        details3 = self.deatil_block3(token3, LHs[1], HLs[1], HHs[1])
        depth3 = self._structure(self.structure_block3, token3, LLs[1], res2, valid_hr)
        depth3 = self.injectors3(depth3, details3)
        res3 = self.residual3(self.upsampler3(depth3))

        token4 = self.stora4(features[0][0], patch_h, patch_w)
        details4 = self.deatil_block4(token4, LHs[0], HLs[0], HHs[0])
        depth4 = self._structure(self.structure_block4, token4, LLs[0], res3, valid_hr)
        depth4 = self.injectors4(depth4, details4)
        res4 = self.final_upscale(depth4)

        last_token = self.semantics_encoder.forward_features(image)["x_norm_patchtokens"]
        B, N, C = last_token.shape
        last_token = last_token.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
        refined, _ = self.boundary_refiner(image, last_token, depth)
        res4 = self.residual4(torch.cat([res4, refined], dim=1))
        return self.final_projection(res4) + depth_bicubic

    @staticmethod
    def _structure(block, token, ll, depth_feat, valid):
        if isinstance(block, FusedStructureBlock):
            return block(token, ll, depth_feat, valid)
        return block(token, ll, depth_feat)


def load_wave_checkpoint(model: WaveAPrime, path, log=print) -> dict[str, list[str]]:
    """Load an original WAVE checkpoint into A', report every key that did not come from it.

    * ``init_depth_projection.net.1.weight`` [F,256,3,3] -> [F,512,3,3] (validity half = 0)
    * ``*.fusion.*`` are new modules (zero-initialised logits): expected to be missing
    * any other missing key is an error; unexpected keys are listed.
    """
    ck = torch.load(path, map_location="cpu")
    state = ck.get("model", ck) if isinstance(ck, dict) else ck
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    own = model.state_dict()
    k = "init_depth_projection.net.1.weight"
    if k in state and k in own and state[k].shape[1] * 2 == own[k].shape[1]:
        state[k] = torch.cat([state[k], torch.zeros_like(state[k])], dim=1)
        log(f"[ckpt] {k}: expanded {tuple(state[k].shape[:2])}+validity half, new weights = 0")
    missing, unexpected = model.load_state_dict(state, strict=False)
    new = [m for m in missing if ".fusion." in m]
    bad = [m for m in missing if ".fusion." not in m]
    log(f"[ckpt] {path}: loaded {len(state) - len(unexpected)} tensors; "
        f"{len(new)} new fusion tensors not in checkpoint (initialised to 0.5/0.5); "
        f"{len(unexpected)} unexpected: {unexpected[:5]}")
    if bad:
        raise RuntimeError(f"checkpoint lacks {len(bad)} tensors that WAVE has, e.g. {bad[:5]}")
    return {"new": new, "unexpected": list(unexpected)}
