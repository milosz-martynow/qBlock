"""Unit tests for compute.io.coordinates module.

Tests cover:
- CartesianCoordinates: initialization, representation, tuple conversion
- Coordinates: abstract base class interface
- from_sequence: factory method for creating coordinates from sequences

All tests use pytest with parametrize, no test classes.
"""

from typing import List, Sequence, Tuple

import pytest

from compute.environment.io.coordinates import CartesianCoordinates, Coordinates

# ======================================================================
# CartesianCoordinates Initialization and Properties Tests
# ======================================================================


@pytest.mark.parametrize(
    "x, y, z",
    [
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0),
        (-1.5, 2.5, 3.5),
        (100.0, -50.0, 0.001),
    ],
    ids=["positive", "origin", "mixed_signs", "large_values"],
)
def test_cartesian_coordinates_init_and_repr(x: float, y: float, z: float) -> None:
    """Verify CartesianCoordinates initialization and string representation.

    :param x: X coordinate value.
    :type x: float
    :param y: Y coordinate value.
    :type y: float
    :param z: Z coordinate value.
    :type z: float
    """
    c: CartesianCoordinates = CartesianCoordinates(x, y, z)

    assert c.x == pytest.approx(x)
    assert c.y == pytest.approx(y)
    assert c.z == pytest.approx(z)
    assert c.as_tuple() == (x, y, z)
    assert repr(c) == f"CartesianCoordinates(x={x:.3f}, y={y:.3f}, z={z:.3f})"


@pytest.mark.parametrize(
    "x, y, z",
    [
        (1.5, 2.5, 3.5),
        (0.0, 0.0, 0.0),
        (-2.0, 4.0, 8.0),
    ],
    ids=["positive", "origin", "mixed"],
)
def test_as_tuple(x: float, y: float, z: float) -> None:
    """Verify as_tuple() returns correct tuple for CartesianCoordinates.

    :param x: X coordinate value.
    :type x: float
    :param y: Y coordinate value.
    :type y: float
    :param z: Z coordinate value.
    :type z: float
    """
    c: CartesianCoordinates = CartesianCoordinates(x, y, z)
    assert c.as_tuple() == (x, y, z)
    assert c.x == x
    assert c.y == y
    assert c.z == z


@pytest.mark.parametrize(
    "x, y, z",
    [
        (1.0, 2.0, 3.0),
        (-5.0, -10.0, -15.0),
    ],
    ids=["positive", "negative"],
)
def test_to_list(x: float, y: float, z: float) -> None:
    """Verify to_list() returns correct list for CartesianCoordinates.

    :param x: X coordinate value.
    :type x: float
    :param y: Y coordinate value.
    :type y: float
    :param z: Z coordinate value.
    :type z: float
    """
    c: CartesianCoordinates = CartesianCoordinates(x, y, z)
    assert c.to_list() == [x, y, z]


# ======================================================================
# CartesianCoordinates Inheritance Tests
# ======================================================================


@pytest.mark.parametrize(
    "x, y, z",
    [
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0),
    ],
    ids=["nonzero", "origin"],
)
def test_inheritance(x: float, y: float, z: float) -> None:
    """Verify CartesianCoordinates is a subclass of Coordinates.

    :param x: X coordinate value.
    :type x: float
    :param y: Y coordinate value.
    :type y: float
    :param z: Z coordinate value.
    :type z: float
    """
    c: CartesianCoordinates = CartesianCoordinates(x, y, z)
    assert isinstance(c, Coordinates)


def test_coordinates_as_tuple_uses_abstract_methods() -> None:
    """Verify Coordinates.as_tuple() relies on _coordinate_[1-3] methods.

    Creates a dummy subclass to track method calls and verify the
    abstract interface is correctly used.
    """

    class DummyCoordinates(Coordinates):
        def __init__(self) -> None:
            self.calls: List[int] = []

        def _coordinate_1(self) -> float:
            self.calls.append(1)
            return 1.0

        def _coordinate_2(self) -> float:
            self.calls.append(2)
            return 2.0

        def _coordinate_3(self) -> float:
            self.calls.append(3)
            return 3.0

    dummy: DummyCoordinates = DummyCoordinates()
    tup: Tuple[float, float, float] = dummy.as_tuple()

    assert tup == (1.0, 2.0, 3.0)
    assert dummy.calls == [1, 2, 3]


# ======================================================================
# CartesianCoordinates Constructor Validation Tests
# ======================================================================


@pytest.mark.parametrize(
    "args",
    [
        (1.0, 2.0),
        (1.0,),
        (),
    ],
    ids=["two_args", "one_arg", "no_args"],
)
def test_cartesian_coordinates_requires_all_args(args: Tuple) -> None:
    """Verify CartesianCoordinates raises TypeError for missing arguments.

    :param args: Incomplete argument tuple to pass to constructor.
    :type args: Tuple
    """
    with pytest.raises(TypeError):
        CartesianCoordinates(*args)


# ======================================================================
# CartesianCoordinates.from_sequence Tests
# ======================================================================


@pytest.mark.parametrize(
    "seq, expected",
    [
        ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
        (["1.0", "2.0", "3.0"], (1.0, 2.0, 3.0)),
        ([0, 0, 0], (0.0, 0.0, 0.0)),
        ((1, 2, 3), (1.0, 2.0, 3.0)),
    ],
    ids=["float_tuple", "string_list", "int_zeros", "int_tuple"],
)
def test_cartesian_from_sequence_valid(
    seq: Sequence, expected: Tuple[float, float, float]
) -> None:
    """Verify from_sequence accepts numeric and string values correctly.

    :param seq: Input sequence to convert to coordinates.
    :type seq: Sequence
    :param expected: Expected coordinate values after conversion.
    :type expected: Tuple[float, float, float]
    """
    c: CartesianCoordinates = CartesianCoordinates.from_sequence(seq)

    assert isinstance(c, CartesianCoordinates)
    assert c.as_tuple() == expected
    assert c.to_list() == list(expected)


@pytest.mark.parametrize(
    "bad_seq, reason",
    [
        ((1.0, 2.0), "wrong length (2 instead of 3)"),
        (("a", "b", "c"), "non-numeric strings"),
        ("abc", "string of length 3 but not iterable of numbers"),
    ],
    ids=["too_short", "non_numeric", "string_chars"],
)
def test_cartesian_from_sequence_invalid(bad_seq: Sequence, reason: str) -> None:
    """Verify from_sequence raises ValueError for invalid input.

    :param bad_seq: Invalid input sequence.
    :type bad_seq: Sequence
    :param reason: Description of why the input is invalid.
    :type reason: str
    """
    with pytest.raises(ValueError):
        CartesianCoordinates.from_sequence(bad_seq)
