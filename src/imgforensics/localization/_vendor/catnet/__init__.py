"""CAT-Net's model definition, vendored for inference.

Vendored from ``mjkwon2021/CAT-Net`` at commit
``331b8059c3f55efec1d9075de79dd153413f2061`` (Apache-2.0; the HRNet blocks it
builds on are MIT, Copyright (c) Microsoft). Each module keeps the upstream
license headers and lists what was changed; the short version is that the
``yacs`` configuration machinery, the dataset pipeline (which is where the
``jpegio`` and ``torch_dct`` dependencies lived) and all training code were
removed, while every submodule and parameter name was left alone so
``CAT_full_v2.pth.tar`` loads with ``strict=True``.

Importing this package requires ``torch`` (the optional ``ml`` extra);
:mod:`imgforensics.localization.catnet` imports it lazily.
"""

from imgforensics.localization._vendor.catnet.network import (
    DCT_VOLUME_CHANNELS,
    NUM_CLASSES,
    CATNet,
)

__all__ = ["DCT_VOLUME_CHANNELS", "NUM_CLASSES", "CATNet"]
