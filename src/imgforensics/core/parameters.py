"""Self-describing parameters for the tools the registry holds.

A tool's constructor arguments are the natural home for "the JPEG quality ELA
re-encodes at" or "how the localizer ensemble combines its members", and from
Python they already work. What they cannot do is describe themselves: a
workbench that draws a control for a tool, and an API that accepts one from a
client it does not trust, both need the name, the type, the range and the
default *before* the tool is built (``docs/design/01_toolbox_architecture.md``,
section 3.1).

:class:`ParameterSpec` is that description, and :func:`coerce_parameters` is
the gate in front of the constructor: it turns whatever the caller sent --
form strings included -- into the keyword arguments the tool declares, or
raises :class:`ValueError` with a message written to be shown to whoever sent
it. A spec that is inconsistent with itself (a choice with no choices, a
default outside its own range) is a mistake by the tool's author rather than
by its caller, so it is rejected when the ParameterSpec is built.

Nothing here imports a tool, or anything beyond pydantic and the standard
library: the whole module is data plus one function.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, Field, model_validator

#: The parameter types a tool may declare. ``"choice"`` is a closed set of
#: strings; the three others are what their names say.
ParameterKind = Literal["int", "float", "bool", "choice"]

#: String spellings of a boolean, matched case-insensitively after stripping.
#: Deliberately short: a query string carries ``true``/``1``, an HTML form
#: carries the same, and anything else is a caller's mistake worth reporting.
_TRUE_STRINGS = frozenset({"1", "true"})
_FALSE_STRINGS = frozenset({"0", "false"})

#: The Python types a declared value can end up as, whatever was sent.
ParameterValue = int | float | bool | str


class ParameterSpec(BaseModel):
    """One user-facing parameter of a tool.

    Attributes:
        name: The keyword argument the tool's constructor takes.
        kind: One of :data:`ParameterKind`.
        default: The value used when the caller says nothing. It must match
            ``kind``, and for a number it must lie inside ``minimum`` /
            ``maximum``, so "the default is always valid" holds by
            construction.
        minimum: Smallest accepted value, inclusive (numbers only).
        maximum: Largest accepted value, inclusive (numbers only).
        step: The granularity a control should offer (numbers only). It is a
            presentation hint and is *not* enforced: rejecting a value between
            two steps would refuse a perfectly runnable configuration for the
            sake of a slider.
        choices: The accepted strings, for ``kind="choice"``.
        description: One line saying what the parameter does, shown next to
            the control.
    """

    name: str
    kind: ParameterKind
    default: ParameterValue
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: list[str] = Field(default_factory=list)
    description: str = ""

    @model_validator(mode="after")
    def _check_default_fits_the_kind(self) -> ParameterSpec:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"parameter {self.name!r}: minimum is above maximum")
        if self.kind == "choice":
            if not self.choices:
                raise ValueError(f"parameter {self.name!r}: kind 'choice' needs choices")
            if self.default not in self.choices:
                raise ValueError(
                    f"parameter {self.name!r}: default {self.default!r} is not one of its choices"
                )
            return self
        if self.kind == "bool":
            if not isinstance(self.default, bool):
                raise ValueError(f"parameter {self.name!r}: kind 'bool' needs a boolean default")
            return self
        if isinstance(self.default, bool) or not isinstance(self.default, int | float):
            raise ValueError(f"parameter {self.name!r}: kind {self.kind!r} needs a numeric default")
        if self.kind == "int" and not isinstance(self.default, int):
            raise ValueError(f"parameter {self.name!r}: kind 'int' needs an integer default")
        if self.minimum is not None and self.default < self.minimum:
            raise ValueError(f"parameter {self.name!r}: default is below minimum")
        if self.maximum is not None and self.default > self.maximum:
            raise ValueError(f"parameter {self.name!r}: default is above maximum")
        return self


def coerce_parameters(
    specs: Iterable[ParameterSpec], values: Mapping[str, object]
) -> dict[str, object]:
    """Validate ``values`` against ``specs`` and return the tool's keyword arguments.

    Every declared parameter appears in the result: one the caller sent is
    converted to the declared type, one it left out takes its ParameterSpec's default.
    Strings are accepted for every kind, because the values reach this
    function from query strings and form fields as often as from Python.

    Args:
        specs: The tool's declared parameters (``BaseDetector.parameters()``).
        values: What the caller asked for, keyed by parameter name.

    Returns:
        ``{name: value}`` covering every spec, ready to splat into the tool's
        constructor.

    Raises:
        ValueError: if ``values`` names a parameter the tool does not declare,
            or carries a value of the wrong type, outside the declared range,
            or outside the declared choices. The message names the parameter
            and what was expected, and is safe to show to the caller.
    """
    by_name = {spec.name: spec for spec in specs}
    unknown = sorted(name for name in values if name not in by_name)
    if unknown:
        known = ", ".join(by_name) or "<none>"
        raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}. Expected one of: {known}")
    return {
        name: _coerce(spec, values[name]) if name in values else spec.default
        for name, spec in by_name.items()
    }


def _coerce(spec: ParameterSpec, value: object) -> ParameterValue:
    """One value converted to ``spec``'s kind and checked against its range."""
    if spec.kind == "bool":
        return _as_bool(spec, value)
    if spec.kind == "choice":
        return _as_choice(spec, value)
    number = _as_int(spec, value) if spec.kind == "int" else _as_float(spec, value)
    return _in_range(spec, number)


def _as_bool(spec: ParameterSpec, value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    raise ValueError(f"Parameter {spec.name!r} expects a boolean (true/false, 1/0), got {value!r}")


def _as_choice(spec: ParameterSpec, value: object) -> str:
    if isinstance(value, str) and value in spec.choices:
        return value
    raise ValueError(
        f"Parameter {spec.name!r} expects one of: {', '.join(spec.choices)}; got {value!r}"
    )


def _as_int(spec: ParameterSpec, value: object) -> int:
    # A bool is an int in Python, but "quality=True" is a caller's mistake
    # rather than "quality=1", so it is refused instead of silently accepted.
    if isinstance(value, bool):
        raise ValueError(f"Parameter {spec.name!r} expects an integer, got {value!r}")
    if isinstance(value, int):
        return value
    # A JSON client that round-trips 1 through a float lands here with 1.0,
    # which is the same number; 1.5 is not and says so.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ValueError(f"Parameter {spec.name!r} expects an integer, got {value!r}")


def _as_float(spec: ParameterSpec, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Parameter {spec.name!r} expects a number, got {value!r}")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise ValueError(f"Parameter {spec.name!r} expects a number, got {value!r}")


def _in_range(spec: ParameterSpec, value: int | float) -> int | float:
    if spec.minimum is not None and value < spec.minimum:
        bound = _bound_text(spec, spec.minimum)
        raise ValueError(f"Parameter {spec.name!r} must be at least {bound}, got {value}")
    if spec.maximum is not None and value > spec.maximum:
        bound = _bound_text(spec, spec.maximum)
        raise ValueError(f"Parameter {spec.name!r} must be at most {bound}, got {value}")
    return value


def _bound_text(spec: ParameterSpec, bound: float) -> str:
    """A range bound as the caller would write it (``50``, not ``50.0``, for an int)."""
    if spec.kind == "int" and float(bound).is_integer():
        return str(int(bound))
    return str(bound)
