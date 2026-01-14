"""
Tests for electron classes moved to `q_block.models.electron`.
"""

import pytest

from q_block.models.electron import Orbital, Shell, SpinOrbital, SubShell


def test_spinorbital_valid_initialization() -> None:
    """
    Test that SpinOrbital initializes correctly with valid quantum numbers.

    This test verifies that valid (n, l, m, s) combinations create an object
    without raising errors and that the repr contains the expected fields.

    :returns: None
    """
    so = SpinOrbital(n=2, l=1, m=0, s=0.5)
    assert so.n == 2
    assert so.l == 1
    assert so.m == 0
    assert so.s == 0.5
    assert so.occupied is False
    r = repr(so)
    assert "So(n=2" in r and "l=1" in r


@pytest.mark.parametrize(
    "n,l,m,s",
    [
        (0, 0, 0, 0.5),  # invalid n
        (2, 2, 0, 0.5),  # invalid l (l >= n)
        (2, 1, 2, 0.5),  # invalid m (|m| > l)
        (1, 0, 0, 0.0),  # invalid s
    ],
)
def test_spinorbital_invalid_parameters(
    n: int, l: int, m: int, s: float
) -> None:
    """
    Test that SpinOrbital raises ValueError for invalid quantum numbers.

    Each parameter set should be rejected by the constructor.

    :param int n: principal quantum number
    :param int l: orbital angular momentum quantum number
    :param int m: magnetic quantum number
    :param float s: spin projection
    :returns: None
    """
    with pytest.raises(ValueError):
        SpinOrbital(n=n, l=l, m=m, s=s)


def test_orbital_pairing_valid() -> None:
    """
    Test that Orbital accepts spin-up and spin-down SpinOrbitals with matching
    (n, l, m) and that repr contains expected fields.

    :returns: None
    """
    up = SpinOrbital(n=3, l=1, m=-1, s=0.5)
    down = SpinOrbital(n=3, l=1, m=-1, s=-0.5)
    orb = Orbital(spin_up=up, spin_down=down)
    assert orb.n == 3
    assert orb.l == 1
    assert orb.m == -1
    assert orb.spin_up is up
    assert orb.spin_down is down
    assert "Orb(n=3" in repr(orb)


def test_orbital_spin_mismatch_raises() -> None:
    """
    Test that providing incorrect spins raises a ValueError.

    :returns: None
    """
    up = SpinOrbital(n=2, l=0, m=0, s=0.5)
    wrong_spin = SpinOrbital(n=2, l=0, m=0, s=0.5)  # also +0.5, should be -0.5
    with pytest.raises(ValueError):
        Orbital(spin_up=up, spin_down=wrong_spin)


def test_orbital_quantum_mismatch_raises() -> None:
    """
    Test that providing SpinOrbitals with different (n, l, m) raises.

    :returns: None
    """
    up = SpinOrbital(n=2, l=1, m=0, s=0.5)
    down = SpinOrbital(n=3, l=1, m=0, s=-0.5)
    with pytest.raises(ValueError):
        Orbital(spin_up=up, spin_down=down)


def test_subshell_orbital_count_and_capacity() -> None:
    """
    Verify that SubShell creates (2l + 1) spatial orbitals and capacity equals
    2 * (2l + 1).

    The test checks orbitals length and the capacity() convenience method.

    :returns: None
    """
    n = 4
    l = 2  # d-subshell -> m = -2,-1,0,1,2 => 5 spatial orbitals
    sub = SubShell(n=n, l=l)
    assert len(sub.orbitals) == (2 * l + 1)
    assert sub._capacity() == 2 * (2 * l + 1)

    ms = list(range(-l, l + 1))
    observed_ms = [orb.m for orb in sub.orbitals]
    assert observed_ms == ms


@pytest.mark.parametrize(
    "invalid",
    [
        (0, 0),  # n < 1
        (2, 2),  # l >= n
        (-1, 0),  # negative n
    ],
)
def test_subshell_invalid_params(invalid: tuple) -> None:
    """
    SubShell should raise ValueError for invalid (n, l) specifications.

    :param tuple invalid: (n, l)
    :returns: None
    """
    n, l = invalid
    with pytest.raises(ValueError):
        SubShell(n=n, l=l)


def test_shell_initialization_and_subshells():
    n = 5
    shell = Shell(n=n)
    assert shell.n == n
    assert len(shell.subshells) == n
    for l, subshell in enumerate(shell.subshells):
        assert isinstance(subshell, SubShell)
        assert subshell.n == n
        assert subshell.l == l


def test_shell_invalid_n() -> None:
    """
    Shell should raise ValueError when initialized with n < 1.

    :returns: None
    """
    with pytest.raises(ValueError):
        Shell(n=0)
