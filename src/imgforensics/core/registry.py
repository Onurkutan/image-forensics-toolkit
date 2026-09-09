"""A simple name-to-class registry for detector implementations."""

from __future__ import annotations

from imgforensics.core.base import BaseDetector

_REGISTRY: dict[str, type[BaseDetector]] = {}


def register(name: str):
    """Class decorator that registers a :class:`BaseDetector` subclass under ``name``."""

    def _decorator(cls: type[BaseDetector]) -> type[BaseDetector]:
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get(name: str) -> type[BaseDetector]:
    """Return the detector class registered under ``name``.

    Raises:
        KeyError: if no detector is registered under ``name``, listing the
            available detector names in the error message.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"No detector registered as {name!r}. Available: {known}") from exc


def available() -> list[str]:
    """Return the sorted names of all registered detectors."""
    return sorted(_REGISTRY)
