# Vendored from mjkwon2021/CAT-Net, file lib/models/network_CAT.py (with the
# architecture hyper-parameters of experiments/CAT_full.yaml folded in), at
# commit 331b8059c3f55efec1d9075de79dd153413f2061 (2026-08-05),
# https://github.com/mjkwon2021/CAT-Net .
#
# The CAT-Net repository is licensed under the Apache License, Version 2.0
# (README.md, "License", updated 2026-08-05: the code is Apache-2.0 and the
# weights and datasets are CC-BY-4.0). You may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# The upstream file carries this header of its own, retained here:
#
#   ---------------------------------------------------------------------
#   Copyright (c) Microsoft
#   Licensed under the MIT License.
#   Written by Ke Sun (sunk@mail.ustc.edu.cn)
#   ---------------------------------------------------------------------
#   Modified by Myung-Joon Kwon
#   mjkwon2021@gmail.com
#   Aug 22, 2020
#
# ---
#
# Vendored for imgforensics: inference-only. Changes made to the vendored
# copy (Apache-2.0 section 4(b) notice):
#
# - The `yacs` config object is gone. `CAT_Net.__init__` took a `config`
#   whose `MODEL.EXTRA` came from `experiments/CAT_full.yaml`; those
#   hyper-parameters are now the `_EXTRA` literal below, read from that file
#   at the commit above, and `DATASET.NUM_CLASSES` (2) and
#   `EXTRA.FINAL_CONV_KERNEL` (1) are named constants.
# - `init_weights` and `get_seg_model` dropped: both exist to load the
#   ImageNet HRNet and the DCT-stream pretrainings before training, and the
#   released checkpoint already contains those weights.
# - `logging` dropped along with them; `os` and `numpy` were imported only
#   for `init_weights`.
# - `F.upsample` (removed from newer PyTorch) replaced by the identical
#   `F.interpolate`; both default to `align_corners=False`.
# - The HRNet building blocks moved to the sibling `hrnet` module.
# - Class renamed `CAT_Net` -> `CATNet` (this repository's naming), and type
#   hints added. The linter also wanted `strict=True` on the one `zip` (both
#   sides have four branches) and one suppression where upstream stores `None`
#   placeholders inside a `ModuleList`.
#
# Submodule and parameter names are unchanged, so `CAT_full_v2.pth.tar`
# loads with `strict=True`.
"""CAT-Net: an HRNet RGB stream and a DCT-coefficient stream fused into one mask.

See :mod:`imgforensics.localization.catnet` for how this is driven and for
the sources describing the architecture and its inputs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from imgforensics.localization._vendor.catnet.hrnet import (
    BN_MOMENTUM,
    BasicBlock,
    BatchNorm2d,
    Bottleneck,
    HighResolutionModule,
    blocks_dict,
)

#: ``MODEL.EXTRA`` of ``experiments/CAT_full.yaml``. ``STAGE1`` is folded
#: into the constants below it, since only its block type, depth and width
#: are ever read.
_EXTRA: dict[str, dict[str, Any]] = {
    "STAGE2": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 2,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": (4, 4),
        "NUM_CHANNELS": (48, 96),
        "FUSE_METHOD": "SUM",
    },
    "STAGE3": {
        "NUM_MODULES": 4,
        "NUM_BRANCHES": 3,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": (4, 4, 4),
        "NUM_CHANNELS": (48, 96, 192),
        "FUSE_METHOD": "SUM",
    },
    "STAGE4": {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 4,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": (4, 4, 4, 4),
        "NUM_CHANNELS": (48, 96, 192, 384),
        "FUSE_METHOD": "SUM",
    },
    "DC_STAGE3": {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 2,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": (4, 4),
        "NUM_CHANNELS": (96, 192),
        "FUSE_METHOD": "SUM",
    },
    "DC_STAGE4": {
        "NUM_MODULES": 2,
        "NUM_BRANCHES": 3,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": (4, 4, 4),
        "NUM_CHANNELS": (96, 192, 384),
        "FUSE_METHOD": "SUM",
    },
    "STAGE5": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 4,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": (4, 4, 4, 4),
        "NUM_CHANNELS": (24, 48, 96, 192),
        "FUSE_METHOD": "SUM",
    },
}

#: ``STAGE1``: one bottleneck stage, 4 blocks of width 64.
_STAGE1_BLOCK = "BOTTLENECK"
_STAGE1_BLOCKS = 4
_STAGE1_CHANNELS = 64

#: ``DATASET.NUM_CLASSES``: authentic and manipulated.
NUM_CLASSES = 2

#: ``EXTRA.FINAL_CONV_KERNEL``.
_FINAL_CONV_KERNEL = 1

#: Channels of the DCT volume the DCT stream consumes (a 21-bin histogram of
#: the quantized coefficient values; see the wrapper module).
DCT_VOLUME_CHANNELS = 21


class CATNet(nn.Module):
    """CAT-Net's full two-stream network, in its released inference configuration."""

    def __init__(self) -> None:
        super().__init__()

        # RGB branch
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        block = blocks_dict[_STAGE1_BLOCK]
        self.layer1 = self._make_layer(block, 64, _STAGE1_CHANNELS, _STAGE1_BLOCKS)
        stage1_out_channel = block.expansion * _STAGE1_CHANNELS

        self.stage2_cfg = _EXTRA["STAGE2"]
        num_channels = self._stage_channels(self.stage2_cfg)
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = _EXTRA["STAGE3"]
        num_channels = self._stage_channels(self.stage3_cfg)
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = _EXTRA["STAGE4"]
        num_channels = self._stage_channels(self.stage4_cfg)
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, RGB_final_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True
        )

        # DCT coefficient branch
        self.dc_layer0_dil = nn.Sequential(
            nn.Conv2d(
                in_channels=DCT_VOLUME_CHANNELS,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dilation=8,
                padding=8,
            ),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.dc_layer1_tail = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.dc_layer2 = self._make_layer(
            BasicBlock, inplanes=4 * 64 * 2, planes=96, blocks=4, stride=1
        )

        self.dc_stage3_cfg = _EXTRA["DC_STAGE3"]
        num_channels = self._stage_channels(self.dc_stage3_cfg)
        self.dc_transition2 = self._make_transition_layer([96], num_channels)
        self.dc_stage3, pre_stage_channels = self._make_stage(self.dc_stage3_cfg, num_channels)

        self.dc_stage4_cfg = _EXTRA["DC_STAGE4"]
        num_channels = self._stage_channels(self.dc_stage4_cfg)
        self.dc_transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.dc_stage4, DC_final_stage_channels = self._make_stage(
            self.dc_stage4_cfg, num_channels, multi_scale_output=True
        )

        DC_final_stage_channels.insert(0, 0)  # to match # branches

        # stage 5
        self.stage5_cfg = _EXTRA["STAGE5"]
        num_channels = self._stage_channels(self.stage5_cfg)
        self.transition4 = self._make_transition_layer(
            [i + j for (i, j) in zip(RGB_final_channels, DC_final_stage_channels, strict=True)],
            num_channels,
        )
        self.stage5, pre_stage_channels = self._make_stage(self.stage5_cfg, num_channels)

        last_inp_channels = sum(pre_stage_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=NUM_CLASSES,
                kernel_size=_FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if _FINAL_CONV_KERNEL == 3 else 0,
            ),
        )

    @staticmethod
    def _stage_channels(layer_config: dict[str, Any]) -> list[int]:
        """A stage's per-branch channel counts, scaled by its block's expansion."""
        expansion = blocks_dict[layer_config["BLOCK"]].expansion
        return [channels * expansion for channels in layer_config["NUM_CHANNELS"]]

    def _make_transition_layer(
        self, num_channels_pre_layer: list[int], num_channels_cur_layer: list[int]
    ) -> nn.ModuleList:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers: list[nn.Module | None] = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s: list[nn.Module] = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        # `None` marks "this branch needs no transition"; `forward` checks for
        # it. PyTorch stores it happily; the type stub does not admit it.
        return nn.ModuleList(transition_layers)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False
                ),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(
        self,
        layer_config: dict[str, Any],
        num_inchannels: list[int],
        multi_scale_output: bool = True,
    ) -> tuple[nn.Sequential, list[int]]:
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = list(layer_config["NUM_BLOCKS"])
        num_channels = list(layer_config["NUM_CHANNELS"])
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            reset_multi_scale_output = multi_scale_output or i != num_modules - 1
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: torch.Tensor, qtable: torch.Tensor) -> torch.Tensor:
        """Logits over :data:`NUM_CLASSES`, at one quarter of the input resolution.

        Args:
            x: ``(B, 3 + 21, H, W)``: the normalized RGB image concatenated
                with the DCT volume. ``H`` and ``W`` must be multiples of 8.
            qtable: ``(B, 1, 8, 8)`` luminance quantization table.
        """
        RGB, DCTcoef = x[:, :3, :, :], x[:, 3:, :, :]

        # RGB Stream
        x = self.conv1(RGB)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        RGB_list = self.stage4(x_list)

        # DCT Stream
        x = self.dc_layer0_dil(DCTcoef)
        x = self.dc_layer1_tail(x)
        B, C, H, W = x.shape
        # [B, 256, H/8, W/8]
        x0 = (
            x.reshape(B, C, H // 8, 8, W // 8, 8)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(B, 64 * C, H // 8, W // 8)
        )
        # [B, C, 8, 8, H/8, W/8]
        x_temp = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, C, 8, 8, H/8, W/8]
        x1 = xq_temp.reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, H/8, W/8]
        x = torch.cat([x0, x1], dim=1)
        x = self.dc_layer2(x)  # [B, 96, H/8, W/8]

        x_list = []
        for i in range(self.dc_stage3_cfg["NUM_BRANCHES"]):
            if self.dc_transition2[i] is not None:
                x_list.append(self.dc_transition2[i](x))
            else:
                x_list.append(x)
        y_list = self.dc_stage3(x_list)

        x_list = []
        for i in range(self.dc_stage4_cfg["NUM_BRANCHES"]):
            if self.dc_transition3[i] is not None:
                x_list.append(self.dc_transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        DC_list = self.dc_stage4(x_list)

        # stage 5
        merged = [
            torch.cat([RGB_list[i + 1], DC_list[i]], 1)
            for i in range(self.stage5_cfg["NUM_BRANCHES"] - 1)
        ]
        merged.insert(0, RGB_list[0])
        x_list = []
        for i in range(self.stage5_cfg["NUM_BRANCHES"]):
            if self.transition4[i] is not None:
                x_list.append(self.transition4[i](merged[i]))
            else:
                x_list.append(merged[i])
        stage5 = self.stage5(x_list)

        # Upsampling
        x0_h, x0_w = stage5[0].size(2), stage5[0].size(3)
        x1 = F.interpolate(stage5[1], size=(x0_h, x0_w), mode="bilinear")
        x2 = F.interpolate(stage5[2], size=(x0_h, x0_w), mode="bilinear")
        x3 = F.interpolate(stage5[3], size=(x0_h, x0_w), mode="bilinear")

        fused = torch.cat([stage5[0], x1, x2, x3], 1)

        return self.last_layer(fused)
