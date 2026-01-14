"""
Unit tests for the coordinates module using pytest with fixtures and parametrize.

All docstrings follow Sphinx format.
"""

import pytest

from q_block.io.coordinates import CartesianCoordinates, Coordinates


@pytest.fixture(
    params=[
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0),
        (-1.5, 2.5, 3.5),
    ]
)
def cartesian_coords(request):
    """
    Fixture providing CartesianCoordinates for parameterized tests.

    :param request: Pytest fixture request object.
    :type request: _pytest.fixtures.SubRequest
    :return: Instance with test values.
    :rtype: CartesianCoordinates
    """
    x, y, z = request.param
    return CartesianCoordinates(x, y, z)


def test_cartesian_coordinates_init_and_repr(cartesian_coords):
    """
    Test initialization and string representation of CartesianCoordinates.

    :param cartesian_coords: Instance provided by fixture.
    :type cartesian_coords: CartesianCoordinates
    """
    c = cartesian_coords
    assert c.x == pytest.approx(c.as_tuple()[0])
    assert c.y == pytest.approx(c.as_tuple()[1])
    assert c.z == pytest.approx(c.as_tuple()[2])
    assert c.as_tuple() == (c.x, c.y, c.z)
    assert (
        repr(c)
        == f"CartesianCoordinates(x={c.x:.3f}, y={c.y:.3f}, z={c.z:.3f})"
    )


@pytest.mark.parametrize(
    "x, y, z",
    [
        (1.5, 2.5, 3.5),
        (0.0, 0.0, 0.0),
        (-2.0, 4.0, 8.0),
    ],
)
def test_as_tuple(x, y, z):
    """
    Test as_tuple() returns correct tuple for CartesianCoordinates.

    :param x: X coordinate.
    :type x: float
    :param y: Y coordinate.
    :type y: float
    :param z: Z coordinate.
    :type z: float
    """
    c = CartesianCoordinates(x, y, z)
    assert c.as_tuple() == (x, y, z)
    assert c.x == x
    assert c.y == y
    assert c.z == z


def test_inheritance(cartesian_coords):
    """
    Test that CartesianCoordinates is a subclass of Coordinates.

    :param cartesian_coords: Instance provided by fixture.
    :type cartesian_coords: CartesianCoordinates
    """
    c = cartesian_coords
    assert isinstance(c, Coordinates)


def test_coordinates_as_tuple_uses_abstract_methods():
    """
    Test that Coordinates.as_tuple() relies on _coordinate_[1-3].
    """

    class DummyCoordinates(Coordinates):
        def __init__(self):
            self.calls = []

        def _coordinate_1(self) -> float:  # type: ignore[override]
            self.calls.append(1)
            return 1.0

        def _coordinate_2(self) -> float:  # type: ignore[override]
            self.calls.append(2)
            return 2.0

        def _coordinate_3(self) -> float:  # type: ignore[override]
            self.calls.append(3)
            return 3.0

    dummy = DummyCoordinates()
    tup = dummy.as_tuple()
    assert tup == (1.0, 2.0, 3.0)
    assert dummy.calls == [1, 2, 3]


@pytest.mark.parametrize(
    "args",
    [
        (1.0, 2.0),
        (1.0,),
        tuple(),
    ],
)
def test_cartesian_coordinates_requires_all_args(args):
    """
    Test that CartesianCoordinates requires all three arguments.

    :param args: Arguments to pass to CartesianCoordinates constructor.
    :type args: tuple
    """
    with pytest.raises(TypeError):
        CartesianCoordinates(*args)


@pytest.mark.parametrize(
    "seq",
    [
        (1.0, 2.0, 3.0),
        ["1.0", "2.0", "3.0"],
    ],
)
def test_cartesian_from_sequence_valid(seq):
    """CartesianCoordinates.from_sequence should accept numeric and string values."""
    c = CartesianCoordinates.from_sequence(seq)
    assert isinstance(c, CartesianCoordinates)
    assert c.as_tuple() == (1.0, 2.0, 3.0)
    assert c.to_list() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "bad_seq",
    [
        (1.0, 2.0),
        ("a", "b", "c"),
        "abc",  # len 3 but not numeric
    ],
)
def test_cartesian_from_sequence_invalid(bad_seq):
    """CartesianCoordinates.from_sequence must raise ValueError for invalid input."""
    with pytest.raises(ValueError):
        CartesianCoordinates.from_sequence(bad_seq)
