"""IML-ViT's model definition, vendored for inference.

Vendored from ``SunnyHaze/IML-ViT`` at commit
``07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`` (MIT, Copyright (c) 2023
Xiaochen Ma). Each module keeps the upstream license header and lists what
was changed; the short version is that training-only code (losses, the MAE
initialization hook, the augmentation pipeline) and the ``fvcore`` dependency
were removed, while every module and parameter name was left alone so the
released checkpoint loads with ``strict=True``.

Importing this package requires ``torch`` and ``timm`` (the optional ``ml``
extra); :mod:`imgforensics.localization.iml_vit` imports it lazily.
"""

from imgforensics.localization._vendor.iml_vit.model import IMLViTModel

__all__ = ["IMLViTModel"]
