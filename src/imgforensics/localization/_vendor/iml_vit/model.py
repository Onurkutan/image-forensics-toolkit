# Vendored from SunnyHaze/IML-ViT, file iml_vit_model.py, at commit
# 07dd2be0f4ea27a5c97c9fa5ffbe236733833eac (2025-06-29),
# https://github.com/SunnyHaze/IML-ViT .
#
# MIT License
#
# Copyright (c) 2023 Xiaochen Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---
#
# Vendored for imgforensics: inference-only. Modifications:
#
# - `forward` returns the predicted probability mask only. Upstream it takes
#   the ground-truth mask and edge mask as arguments and returns
#   `(loss, mask, edge_loss)`; the BCE losses, the `edge_lambda`
#   hyper-parameter and the `vit_pretrain_path` MAE-initialization hook are
#   training-only and were dropped.
# - The four `print` calls upstream's `forward` makes on every pass were
#   dropped.
# - Class renamed `iml_vit_model` -> `IMLViTModel` (this repository's naming),
#   and type hints added. Submodule attribute names (`encoder_net`,
#   `featurePyramid_net`, `predict_head`) are unchanged, so the released
#   checkpoint loads with `strict=True`.
"""IML-ViT assembled: windowed-attention ViT, simple feature pyramid, decoder head."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from imgforensics.localization._vendor.iml_vit.decoder import PredictHead
from imgforensics.localization._vendor.iml_vit.vit import (
    LastLevelMaxPool,
    SimpleFeaturePyramid,
    ViT,
)


class IMLViTModel(nn.Module):
    """The IML-ViT manipulation localizer, in its released inference configuration."""

    def __init__(
        self,
        # ViT backbone:
        input_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 768,
        # Simple feature pyramid network:
        fpn_channels: int = 256,
        fpn_scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        # MLP embedding:
        mlp_embeding_dim: int = 256,
        # Decoder head norm
        predict_head_norm: str = "BN",
    ) -> None:
        """Build the model.

        Args:
            input_size (int): size of the input image, default 1024.
            patch_size (int): patch size of the Vision Transformer.
            embed_dim (int): embedding dim for the ViT.
            fpn_channels (int): number of embedding channels for the simple feature pyramid.
            fpn_scale_factors (list of float): the rescale factor for each SFPN layer.
            mlp_embeding_dim (int): dim of the decoder head.
            predict_head_norm (str): the norm layer of the predict head, one of 'BN', 'IN'
                and 'LN'. The released checkpoint was trained with 'BN'.
        """
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        # window attention vit
        self.encoder_net = ViT(
            img_size=input_size,
            patch_size=16,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[
                # 2, 5, 8, 11 use global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            residual_block_indexes=[],
            use_rel_pos=True,
            out_feature="last_feat",
        )

        # simple feature pyramid network
        self.featurePyramid_net = SimpleFeaturePyramid(
            input_dim=embed_dim,
            out_channels=fpn_channels,
            scale_factors=fpn_scale_factors,
            top_block=LastLevelMaxPool(),
            norm="LN",
        )
        # MLP predict head
        self.predict_head = PredictHead(
            feature_channels=[fpn_channels for _ in range(5)],
            embed_dim=mlp_embeding_dim,
            norm=predict_head_norm,  # important! may influence the results
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the ``(B, 1, input_size, input_size)`` probability mask for ``x``."""
        features = self.encoder_net(x)
        pyramid = self.featurePyramid_net(features)
        logits = self.predict_head(list(pyramid.values()))

        # up-sample to input_size x input_size
        mask_pred = F.interpolate(
            logits, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
        )
        return torch.sigmoid(mask_pred)
