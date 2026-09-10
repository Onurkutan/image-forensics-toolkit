"""Tests for imgforensics.core.parameters: spec validation and value coercion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from imgforensics.core.parameters import ParameterSpec, coerce_parameters

QUALITY = ParameterSpec(name="quality", kind="int", default=95, minimum=50, maximum=100, step=1)
STRENGTH = ParameterSpec(name="strength", kind="float", default=0.5, minimum=0.0, maximum=1.0)
ATTRIBUTION = ParameterSpec(name="attribution", kind="bool", default=True)
MODE = ParameterSpec(name="mode", kind="choice", default="mean", choices=["mean", "max"])

ALL_SPECS = [QUALITY, STRENGTH, ATTRIBUTION, MODE]


def test_declared_defaults_keep_their_python_types() -> None:
    assert isinstance(QUALITY.default, int)
    assert isinstance(STRENGTH.default, float)
    assert isinstance(ATTRIBUTION.default, bool)
    assert isinstance(MODE.default, str)


def test_missing_values_take_the_declared_defaults() -> None:
    assert coerce_parameters(ALL_SPECS, {}) == {
        "quality": 95,
        "strength": 0.5,
        "attribution": True,
        "mode": "mean",
    }


def test_no_specs_accepts_nothing_and_returns_nothing() -> None:
    assert coerce_parameters([], {}) == {}


def test_unknown_parameter_names_are_listed_with_what_was_expected() -> None:
    with pytest.raises(ValueError) as excinfo:
        coerce_parameters([QUALITY], {"qualtiy": 90, "colour": "red"})

    message = str(excinfo.value)
    assert "colour" in message
    assert "qualtiy" in message
    assert "quality" in message


@pytest.mark.parametrize("value", [80, 80.0, "80", " 80 "])
def test_int_accepts_numbers_and_numeric_strings(value: object) -> None:
    coerced = coerce_parameters([QUALITY], {"quality": value})["quality"]
    assert coerced == 80
    assert isinstance(coerced, int)


@pytest.mark.parametrize("value", [80.5, "80.5", "high", True, None, [80]])
def test_int_rejects_anything_that_is_not_a_whole_number(value: object) -> None:
    with pytest.raises(ValueError, match="expects an integer"):
        coerce_parameters([QUALITY], {"quality": value})


@pytest.mark.parametrize(("value", "expected"), [(1, 1.0), ("0.25", 0.25), (0.25, 0.25)])
def test_float_accepts_ints_floats_and_numeric_strings(value: object, expected: float) -> None:
    coerced = coerce_parameters([STRENGTH], {"strength": value})["strength"]
    assert coerced == pytest.approx(expected)
    assert isinstance(coerced, float)


@pytest.mark.parametrize("value", ["a lot", True, None])
def test_float_rejects_non_numbers(value: object) -> None:
    with pytest.raises(ValueError, match="expects a number"):
        coerce_parameters([STRENGTH], {"strength": value})


@pytest.mark.parametrize("value", [49, 101, "49", "101"])
def test_numbers_outside_the_declared_range_are_refused(value: object) -> None:
    with pytest.raises(ValueError, match="quality"):
        coerce_parameters([QUALITY], {"quality": value})


def test_range_bounds_are_inclusive() -> None:
    assert coerce_parameters([QUALITY], {"quality": 50})["quality"] == 50
    assert coerce_parameters([QUALITY], {"quality": 100})["quality"] == 100


def test_range_message_spells_an_int_bound_as_an_int() -> None:
    with pytest.raises(ValueError, match=r"at most 100, got 120"):
        coerce_parameters([QUALITY], {"quality": 120})


def test_step_is_a_hint_and_does_not_reject_a_value_between_two_steps() -> None:
    fine = ParameterSpec(name="k", kind="float", default=1.0, minimum=0.0, maximum=2.0, step=0.5)
    assert coerce_parameters([fine], {"k": 1.3})["k"] == pytest.approx(1.3)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("1", True),
        ("0", False),
        ("true", True),
        ("false", False),
        ("TRUE", True),
        ("False", False),
        (" true ", True),
    ],
)
def test_bool_accepts_booleans_and_their_string_spellings(value: object, expected: bool) -> None:
    assert coerce_parameters([ATTRIBUTION], {"attribution": value})["attribution"] is expected


@pytest.mark.parametrize("value", [1, 0, "yes", "off", None])
def test_bool_rejects_everything_else(value: object) -> None:
    with pytest.raises(ValueError, match="expects a boolean"):
        coerce_parameters([ATTRIBUTION], {"attribution": value})


def test_choice_accepts_a_declared_choice() -> None:
    assert coerce_parameters([MODE], {"mode": "max"})["mode"] == "max"


@pytest.mark.parametrize("value", ["rank_mean", "MEAN", 0, None])
def test_choice_rejects_anything_not_declared_and_names_the_options(value: object) -> None:
    with pytest.raises(ValueError) as excinfo:
        coerce_parameters([MODE], {"mode": value})

    assert "mean, max" in str(excinfo.value)


def test_values_of_several_parameters_are_coerced_independently() -> None:
    coerced = coerce_parameters(ALL_SPECS, {"quality": "70", "attribution": "false"})

    assert coerced == {"quality": 70, "strength": 0.5, "attribution": False, "mode": "mean"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "mode", "kind": "choice", "default": "mean"},
        {"name": "mode", "kind": "choice", "default": "nope", "choices": ["mean"]},
        {"name": "flag", "kind": "bool", "default": "true"},
        {"name": "quality", "kind": "int", "default": 95.5},
        {"name": "quality", "kind": "int", "default": True},
        {"name": "quality", "kind": "int", "default": 20, "minimum": 50},
        {"name": "quality", "kind": "int", "default": 200, "maximum": 100},
        {"name": "quality", "kind": "int", "default": 60, "minimum": 90, "maximum": 50},
    ],
)
def test_a_spec_that_contradicts_itself_is_refused_when_it_is_built(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ParameterSpec(**kwargs)
