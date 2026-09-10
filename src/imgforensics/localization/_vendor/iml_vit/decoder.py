# Vendored from SunnyHaze/IML-ViT, file modules/decoderhead.py, at commit
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
# The upstream file credits SegFormer for the head's design,
# https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py .
#
# ---
#
# Vendored for imgforensics: inference-only. Modifications: the file's own
# copy of the channel-first `LayerNorm` is imported from the vendored `vit`
# module instead of being duplicated, the commented-out per-level `MLP`
# projections (and the unused `MLP` class itself) were dropped, and type hints
# were added for this repository's linter. Module and parameter names are
# unchanged, so the released checkpoint loads with `strict=True`.
"""IML-ViT's decoder head: fuse the five pyramid levels into one logit map."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from imgforensics.localization._vendor.iml_vit.vit import LayerNorm


class PredictHead(nn.Module):
    def __init__(
        self,
        feature_channels: list[int],
        embed_dim: int = 256,
        predict_channels: int = 1,
        norm: str = "BN",
    ) -> None:
        """
        Upstream note on the normalization choice, kept verbatim:

        We tested three different types of normalization in the decoder head, and they
        may yield different results due to dataset configurations and other factors.
        Some intuitive conclusions are as follows:
            - "LN" -> Layer norm : The fastest convergence, but poor generalization
              performance.
            - "BN" Batch norm : When include authentic images during training, set
              batchsize = 2 may have poor performance. But if you can train with larger
              batchsize (e.g. A40 with 48GB memory can train with batchsize = 4) It may
              performs better.
            - "IN" Instance norm : A form that can definitely converge, equivalent to a
              batchnorm with batchsize=1. When abnormal behavior is observed with
              BatchNorm, one can consider trying Instance Normalization. It's important
              to note that in this case, the settings should include setting
              track_running_stats and affine to True, rather than the default settings
              in PyTorch.
        """
        super().__init__()
        assert len(feature_channels) == 5, "feature_channels must be a list of 5 elements"

        self.linear_fuse = nn.Conv2d(
            in_channels=embed_dim * 5, out_channels=embed_dim, kernel_size=1
        )

        assert norm in ["LN", "BN", "IN"], (
            "Argument error when initialize the predict head : Norm argument should be one "
            "of the 'LN', 'BN' , 'IN', which represent Layer_norm, Batch_norm and Instance_norm"
        )

        self.norm: nn.Module
        if norm == "LN":
            self.norm = LayerNorm(embed_dim)
        elif norm == "BN":
            self.norm = nn.BatchNorm2d(embed_dim)
        else:
            self.norm = nn.InstanceNorm2d(embed_dim, track_running_stats=True, affine=True)

        self.dropout = nn.Dropout()

        self.linear_predict = nn.Conv2d(embed_dim, predict_channels, kernel_size=1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4, c5 = x  # 1/4 1/8 1/16 1/32 1/64

        _, _, h, w = c1.shape  # Target size of all the features

        resized = [
            F.interpolate(level, size=(h, w), mode="bilinear", align_corners=False)
            for level in (c1, c2, c3, c4, c5)
        ]

        fused = self.linear_fuse(torch.cat(resized, dim=1))
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return self.linear_predict(fused)
