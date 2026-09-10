"""Recursive conversion of detector ``details`` values into JSON-serialisable Python types.

Detector ``details`` dicts often carry numpy scalars/arrays (e.g. a shape
tuple's numpy ints, or a raw float32) and, occasionally, raw bytes. Neither
survives :func:`json.dumps` unchanged, so both the CLI's ``--json`` output
and the explanation report builder need the same conversion; this module is
the one place it lives.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and bytes into JSON-serialisable values.

    Dicts and lists/tuples are walked recursively; everything else (``int``,
    ``float``, ``str``, ``bool``, ``None``, and any other type already
    JSON-native) is returned unchanged.
    """
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value
