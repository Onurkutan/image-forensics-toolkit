# Vendored from mjkwon2021/CAT-Net, file lib/models/network_CAT.py, at commit
# 331b8059c3f55efec1d9075de79dd153413f2061 (2026-08-05),
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
# The upstream file carries this header of its own, which is retained here
# because the HighResolutionNet building blocks below come from HRNet:
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
# - Split off from `network_CAT.py`: this module keeps only the reusable
#   HRNet pieces (`BasicBlock`, `Bottleneck`, `HighResolutionModule`,
#   `blocks_dict`); the CAT-Net network itself is in the sibling `network`
#   module.
# - `logger`/`logging` removed; the three `_check_branches` error messages
#   now raise `ValueError` directly instead of logging and then raising.
# - `F.upsample` (removed from newer PyTorch) replaced by the identical
#   `F.interpolate`; both default to `align_corners=False`.
# - Type hints added for this repository's linter, which also wanted a local
#   alias for `self.fuse_layers` in `HighResolutionModule.forward` (indexing a
#   `ModuleList` twice loses the element type) and two suppressions where
#   upstream stores `None` placeholders inside a `ModuleList`.
#
# Class and parameter names are unchanged, so the released checkpoint loads
# with `strict=True`.
"""HRNet's residual blocks and multi-resolution fusion module, as CAT-Net uses them."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        blocks: type[BasicBlock] | type[Bottleneck],
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(
        self,
        num_branches: int,
        blocks: type[nn.Module],
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
    ) -> None:
        if num_branches != len(num_blocks):
            raise ValueError(f"NUM_BRANCHES({num_branches}) <> NUM_BLOCKS({len(num_blocks)})")

        if num_branches != len(num_channels):
            raise ValueError(f"NUM_BRANCHES({num_branches}) <> NUM_CHANNELS({len(num_channels)})")

        if num_branches != len(num_inchannels):
            raise ValueError(
                f"NUM_BRANCHES({num_branches}) <> NUM_INCHANNELS({len(num_inchannels)})"
            )

    def _make_one_branch(
        self,
        branch_index: int,
        block: type[BasicBlock] | type[Bottleneck],
        num_blocks: list[int],
        num_channels: list[int],
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = [
            block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)
        ]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: type[BasicBlock] | type[Bottleneck],
        num_blocks: list[int],
        num_channels: list[int],
    ) -> nn.ModuleList:
        branches = [
            self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)
        ]
        return nn.ModuleList(branches)

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers: list[nn.ModuleList] = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer: list[nn.Module | None] = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                            BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s: list[nn.Module] = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            # `None` marks "this branch feeds itself", and `forward` never
            # calls that entry. PyTorch stores it happily; the type stub does
            # not admit it.
            fuse_layers.append(nn.ModuleList(fuse_layer))  # type: ignore[arg-type]

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> list[int]:
        return self.num_inchannels

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        assert self.fuse_layers is not None
        # Aliased with its element type spelled out: indexing twice into a
        # bare `ModuleList` loses it, and every lookup below does exactly that.
        fuse_layers = cast("list[nn.ModuleList]", list(self.fuse_layers))
        x_fuse = []
        for i in range(len(fuse_layers)):
            y = x[0] if i == 0 else fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                    )
                else:
                    y = y + fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict: dict[str, type[BasicBlock] | type[Bottleneck]] = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck,
}
