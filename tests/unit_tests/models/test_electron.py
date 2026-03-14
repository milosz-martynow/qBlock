"""Unit tests for q_block.models.electron module.

Tests cover:
- SpinOrbital: initialization, validation, representation
- Orbital: spin pairing, quantum number matching
- SubShell: orbital count, capacity, validation
- Shell: initialization, subshell generation

All tests use pytest with parametrize, no test classes.
"""

from typing import Tuple

import pytest

from q_block.models.electron import Orbital, Shell, SpinOrbital, SubShell


# ======================================================================
# SpinOrbital Tests
# ======================================================================


def test_spinorbital_valid_initialization() -> None:
    """Verify SpinOrbital initializes correctly with valid quantum numbers.

    Creates a SpinOrbital with (n=2, l=1, m=0, s=0.5) and validates
    all attributes and representation string.
    """
    so: SpinOrbital = SpinOrbital(n=2, l=1, m=0, s=0.5)
    assert so.n == 2
    assert so.l == 1
    assert so.m == 0
    assert so.s == 0.5
    assert so.occupied is False
    r: str = repr(so)
    assert "So(n=2" in r and "l=1" in r


@pytest.mark.parametrize(
    "n, l, m, s, description",
    [
        (0, 0, 0, 0.5, "n must be >= 1"),
        (2, 2, 0, 0.5, "l must be < n"),
        (2, 1, 2, 0.5, "|m| must be <= l"),
        (1, 0, 0, 0.0, "s must be +0.5 or -0.5"),
    ],
    ids=["invalid_n", "invalid_l", "invalid_m", "invalid_s"],
)
def test_spinorbital_invalid_parameters(
    n: int, l: int, m: int, s: float, description: str
) -> None:
    """Verify SpinOrbital raises ValueError for invalid quantum numbers.

    :param n: Principal quantum number.
    :type n: int
    :param l: Orbital angular momentum quantum number.
    :type l: int
    :param m: Magnetic quantum number.
    :type m: int
    :param s: Spin projection.
    :type s: float
    :param description: Explanation of why this parameter set is invalid.
    :type description: str
    """
    with pytest.raises(ValueError):
        SpinOrbital(n=n, l=l, m=m, s=s)


# ======================================================================
# Orbital Tests
# ======================================================================


def test_orbital_pairing_valid() -> None:
    """Verify Orbital accepts spin-up/down SpinOrbitals with matching (n, l, m).

    Creates an Orbital from two SpinOrbitals with (n=3, l=1, m=-1)
    and validates pairing and representation.
    """
    up: SpinOrbital = SpinOrbital(n=3, l=1, m=-1, s=0.5)
    down: SpinOrbital = SpinOrbital(n=3, l=1, m=-1, s=-0.5)
    orb: Orbital = Orbital(spin_up=up, spin_down=down)

    assert orb.n == 3
    assert orb.l == 1
    assert orb.m == -1
    assert orb.spin_up is up
    assert orb.spin_down is down
    assert "Orb(n=3" in repr(orb)


def test_orbital_spin_mismatch_raises() -> None:
    """Verify Orbital raises ValueError when both spins are +0.5.

    Both SpinOrbitals have s=0.5, which violates the requirement
    for opposite spins.
    """
    up: SpinOrbital = SpinOrbital(n=2, l=0, m=0, s=0.5)
    wrong_spin: SpinOrbital = SpinOrbital(n=2, l=0, m=0, s=0.5)
    with pytest.raises(ValueError):
        Orbital(spin_up=up, spin_down=wrong_spin)


def test_orbital_quantum_mismatch_raises() -> None:
    """Verify Orbital raises ValueError when (n, l, m) differ between spins.

    SpinOrbitals have different principal quantum numbers (n=2 vs n=3).
    """
    up: SpinOrbital = SpinOrbital(n=2, l=1, m=0, s=0.5)
    down: SpinOrbital = SpinOrbital(n=3, l=1, m=0, s=-0.5)
    with pytest.raises(ValueError):
        Orbital(spin_up=up, spin_down=down)


# ======================================================================
# SubShell Tests
# ======================================================================


@pytest.mark.parametrize(
    "n, l, expected_orbitals, expected_capacity",
    [
        (1, 0, 1, 2),    # 1s: 1 orbital, 2 electrons
        (2, 1, 3, 6),    # 2p: 3 orbitals, 6 electrons
        (3, 2, 5, 10),   # 3d: 5 orbitals, 10 electrons
        (4, 3, 7, 14),   # 4f: 7 orbitals, 14 electrons
    ],
    ids=["s_subshell", "p_subshell", "d_subshell", "f_subshell"],
)
def test_subshell_orbital_count_and_capacity(
    n: int, l: int, expected_orbitals: int, expected_capacity: int
) -> None:
    """Verify SubShell creates (2l+1) orbitals with capacity 2*(2l+1).

    :param n: Principal quantum number.
    :type n: int
    :param l: Orbital angular momentum quantum number.
    :type l: int
    :param expected_orbitals: Expected number of spatial orbitals (2l+1).
    :type expected_orbitals: int
    :param expected_capacity: Expected electron capacity 2*(2l+1).
    :type expected_capacity: int
    """
    sub: SubShell = SubShell(n=n, l=l)

    assert len(sub.orbitals) == expected_orbitals
    assert sub._capacity() == expected_capacity

    ms: list = list(range(-l, l + 1))
    observed_ms: list = [orb.m for orb in sub.orbitals]
    assert observed_ms == ms


@pytest.mark.parametrize(
    "n, l, reason",
    [
        (0, 0, "n must be >= 1"),
        (2, 2, "l must be < n"),
        (-1, 0, "n must be positive"),
    ],
    ids=["n_zero", "l_equals_n", "n_negative"],
)
def test_subshell_invalid_parameters(n: int, l: int, reason: str) -> None:
    """Verify SubShell raises ValueError for invalid (n, l) specifications.

    :param n: Principal quantum number.
    :type n: int
    :param l: Orbital angular momentum quantum number.
    :type l: int
    :param reason: Explanation of why the parameter set is invalid.
    :type reason: str
    """
    with pytest.raises(ValueError):
        SubShell(n=n, l=l)


# ======================================================================
# Shell Tests
# ======================================================================


@pytest.mark.parametrize(
    "n, expected_subshells",
    [
        (1, 1),   # 1s only
        (2, 2),   # 2s, 2p
        (3, 3),   # 3s, 3p, 3d
        (4, 4),   # 4s, 4p, 4d, 4f
        (5, 5),   # 5s, 5p, 5d, 5f, 5g
    ],
    ids=["n=1", "n=2", "n=3", "n=4", "n=5"],
)
def test_shell_initialization_and_subshells(
    n: int, expected_subshells: int
) -> None:
    """Verify Shell creates n subshells with correct quantum numbers.

    :param n: Principal quantum number.
    :type n: int
    :param expected_subshells: Expected number of subshells (equals n).
    :type expected_subshells: int
    """
    shell: Shell = Shell(n=n)

    assert shell.n == n
    assert len(shell.subshells) == expected_subshells

    for l, subshell in enumerate(shell.subshells):
        assert isinstance(subshell, SubShell)
        assert subshell.n == n
        assert subshell.l == l


@pytest.mark.parametrize(
    "n",
    [0, -1, -5],
    ids=["n=0", "n=-1", "n=-5"],
)
def test_shell_invalid_n(n: int) -> None:
    """Verify Shell raises ValueError when initialized with n < 1.

    :param n: Invalid principal quantum number (< 1).
    :type n: int
    """
    with pytest.raises(ValueError):
        Shell(n=n)
