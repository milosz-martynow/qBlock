"""Unit tests for q_block.utilities.mathematics module.

Tests cover:
- double_factorial: double factorial computation including special cases
- normalization_constant: Gaussian primitive normalization for various orbitals
- get_cartesian_components: angular momentum component generation

All tests use pytest with parametrize, no test classes.
"""

import math
from typing import List, Tuple

import pytest

from q_block.utilities.mathematics import (
    double_factorial,
    get_cartesian_components,
    normalization_constant,
)

# ======================================================================
# double_factorial Tests
# ======================================================================


@pytest.mark.parametrize(
    "n, expected",
    [
        (-1, 1),  # Quantum chemistry convention: (-1)!! = 1
        (0, 1),  # 0!! = 1
        (1, 1),  # 1!! = 1
        (2, 2),  # 2!! = 2
        (3, 3),  # 3!! = 3 * 1 = 3
        (4, 8),  # 4!! = 4 * 2 = 8
        (5, 15),  # 5!! = 5 * 3 * 1 = 15
        (6, 48),  # 6!! = 6 * 4 * 2 = 48
        (7, 105),  # 7!! = 7 * 5 * 3 * 1 = 105
    ],
    ids=["n=-1", "n=0", "n=1", "n=2", "n=3", "n=4", "n=5", "n=6", "n=7"],
)
def test_double_factorial(n: int, expected: int) -> None:
    """Verify double_factorial computes correct values.

    :param n: Input value for double factorial.
    :type n: int
    :param expected: Expected result of n!!.
    :type expected: int
    """
    result: int = double_factorial(n)
    assert result == expected
    assert isinstance(result, int)


# ======================================================================
# normalization_constant Tests
# ======================================================================


@pytest.mark.parametrize(
    "alpha, lx, ly, lz, description",
    [
        (1.0, 0, 0, 0, "s-orbital, alpha=1.0"),
        (2.5, 0, 0, 0, "s-orbital, alpha=2.5"),
        (1.0, 1, 0, 0, "px-orbital"),
        (1.0, 0, 1, 0, "py-orbital"),
        (1.0, 0, 0, 1, "pz-orbital"),
    ],
    ids=["s_a1", "s_a2.5", "px", "py", "pz"],
)
def test_normalization_constant_positive(
    alpha: float, lx: int, ly: int, lz: int, description: str
) -> None:
    """Verify normalization constant is always positive.

    :param alpha: Gaussian exponent.
    :type alpha: float
    :param lx: Angular momentum in x direction.
    :type lx: int
    :param ly: Angular momentum in y direction.
    :type ly: int
    :param lz: Angular momentum in z direction.
    :type lz: int
    :param description: Description of the test case.
    :type description: str
    """
    norm: float = normalization_constant(alpha, lx, ly, lz)
    assert norm > 0


def test_normalization_constant_s_orbital_formula() -> None:
    """Verify s-orbital normalization matches analytical formula.

    For s-orbital (l=0): N = (2*alpha/pi)^(3/4)
    """
    alpha: float = 1.0
    norm: float = normalization_constant(alpha, 0, 0, 0)
    expected: float = (2.0 * alpha / math.pi) ** 0.75
    assert abs(norm - expected) < 1e-10


@pytest.mark.parametrize(
    "alpha",
    [0.5, 1.0, 2.0, 3.0],
    ids=["alpha=0.5", "alpha=1.0", "alpha=2.0", "alpha=3.0"],
)
def test_normalization_constant_s_orbital_various_exponents(alpha: float) -> None:
    """Verify s-orbital normalization for various exponents.

    :param alpha: Gaussian exponent.
    :type alpha: float
    """
    norm: float = normalization_constant(alpha, 0, 0, 0)
    expected: float = (2.0 * alpha / math.pi) ** 0.75
    assert abs(norm - expected) < 1e-10


def test_normalization_constant_p_orbital_formula() -> None:
    """Verify px-orbital normalization matches analytical formula.

    For px-orbital (lx=1, ly=0, lz=0):
    N = (2*alpha/pi)^(3/4) * sqrt((4*alpha)^1 / 1!!)
    """
    alpha: float = 1.0
    norm: float = normalization_constant(alpha, 1, 0, 0)
    expected: float = (2.0 * alpha / math.pi) ** 0.75 * math.sqrt(4.0 * alpha)
    assert abs(norm - expected) < 1e-10


def test_normalization_constant_symmetry_in_angular_momentum() -> None:
    """Verify swapping angular momentum components gives same result.

    All p-orbital combinations (1,1,0), (1,0,1), (0,1,1) should have
    equal normalization constants due to symmetry.
    """
    alpha: float = 1.5
    norm_xy: float = normalization_constant(alpha, 1, 1, 0)
    norm_xz: float = normalization_constant(alpha, 1, 0, 1)
    norm_yz: float = normalization_constant(alpha, 0, 1, 1)

    assert abs(norm_xy - norm_xz) < 1e-10
    assert abs(norm_xy - norm_yz) < 1e-10


def test_normalization_constant_higher_exponent_larger() -> None:
    """Verify higher exponent gives larger normalization constant."""
    norm_1: float = normalization_constant(1.0, 0, 0, 0)
    norm_2: float = normalization_constant(2.0, 0, 0, 0)
    norm_3: float = normalization_constant(3.0, 0, 0, 0)

    assert norm_1 < norm_2 < norm_3


def test_normalization_constant_d_orbital_formula() -> None:
    """Verify d_xy-orbital normalization matches analytical formula.

    For d_xy (lx=1, ly=1, lz=0):
    N = (2*alpha/pi)^(3/4) * sqrt((4*alpha)^2 / (1!! * 1!! * (-1)!!))
    """
    alpha: float = 1.0
    norm: float = normalization_constant(alpha, 1, 1, 0)
    expected: float = (2.0 * alpha / math.pi) ** 0.75 * math.sqrt(16.0 * alpha**2)
    assert abs(norm - expected) < 1e-10


@pytest.mark.parametrize(
    "lx, ly, lz",
    [
        (lx, ly, lz)
        for lx in range(4)
        for ly in range(4 - lx)
        for lz in range(4 - lx - ly)
    ],
)
def test_normalization_constant_always_positive(lx: int, ly: int, lz: int) -> None:
    """Verify normalization constant is positive for all angular momentum.

    :param lx: Angular momentum in x direction.
    :type lx: int
    :param ly: Angular momentum in y direction.
    :type ly: int
    :param lz: Angular momentum in z direction.
    :type lz: int
    """
    for alpha in [0.5, 1.0, 2.0]:
        norm: float = normalization_constant(alpha, lx, ly, lz)
        assert norm > 0


# ======================================================================
# get_cartesian_components Tests
# ======================================================================


def _generate_cartesian_components(l: int) -> List[Tuple[int, int, int]]:
    """Generate all (lx, ly, lz) tuples where lx + ly + lz = l.

    Expected expansions for low angular momentum:
    - s (l=0): [(0, 0, 0)]
    - p (l=1): [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    - d (l=2): [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]

    :param l: Total angular momentum quantum number (l >= 0).
    :type l: int
    :returns: List of all valid (lx, ly, lz) component tuples.
    :rtype: List[Tuple[int, int, int]]
    """
    return [
        (lx, ly, l - lx - ly) for lx in range(l + 1) for ly in range(l + 1 - lx)
    ]


@pytest.mark.parametrize(
    "l, expected_components",
    [
        (0, _generate_cartesian_components(0)),  # s
        (1, _generate_cartesian_components(1)),  # p
        (2, _generate_cartesian_components(2)),  # d
        (3, _generate_cartesian_components(3)),  # f
        (4, _generate_cartesian_components(4)),  # g
        (5, _generate_cartesian_components(5)),  # h
        (6, _generate_cartesian_components(6)),  # i
        (7, _generate_cartesian_components(7)),  # k
        (8, _generate_cartesian_components(8)),  # l
    ],
    ids=[
        "s (l=0)",
        "p (l=1)",
        "d (l=2)",
        "f (l=3)",
        "g (l=4)",
        "h (l=5)",
        "i (l=6)",
        "k (l=7)",
        "l (l=8)",
    ],
)
def test_get_cartesian_components(
    l: int, expected_components: List[Tuple[int, int, int]]
) -> None:
    """Verify get_cartesian_components returns correct components for orbitals l=0..8.

    Expected expansions for low angular momentum:
    - s (l=0): [(0, 0, 0)]
    - p (l=1): [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    - d (l=2): [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]

    :param l: Total angular momentum quantum number.
    :type l: int
    :param expected_components: Expected list of (lx, ly, lz) tuples.
    :type expected_components: List[Tuple[int, int, int]]
    """
    components: List[Tuple[int, int, int]] = get_cartesian_components(l)
    expected_count: int = (l + 1) * (l + 2) // 2
    assert len(components) == expected_count
    assert len(components) == len(expected_components)
    for comp in expected_components:
        assert comp in components
    for lx, ly, lz in components:
        assert lx + ly + lz == l


@pytest.mark.parametrize(
    "l",
    [0, 1, 2, 3, 4, 5],
    ids=["l=0", "l=1", "l=2", "l=3", "l=4", "l=5"],
)
def test_get_cartesian_components_sum_equals_l(l: int) -> None:
    """Verify all components have lx + ly + lz = l.

    :param l: Total angular momentum quantum number.
    :type l: int
    """
    components: List[Tuple[int, int, int]] = get_cartesian_components(l)
    for lx, ly, lz in components:
        assert lx + ly + lz == l


@pytest.mark.parametrize(
    "l",
    [0, 1, 2, 3, 4, 5],
    ids=["l=0", "l=1", "l=2", "l=3", "l=4", "l=5"],
)
def test_get_cartesian_components_count(l: int) -> None:
    """Verify number of components equals (l+1)(l+2)/2.

    :param l: Total angular momentum quantum number.
    :type l: int
    """
    components: List[Tuple[int, int, int]] = get_cartesian_components(l)
    expected_count: int = (l + 1) * (l + 2) // 2
    assert len(components) == expected_count
