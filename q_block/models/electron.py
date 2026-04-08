"""
Quantum-structure electron-state classes extracted from `atom.py`.

These classes are copied verbatim from `q_block.atom` to preserve existing
behaviour and documentation. The module exposes:

    - SpinOrbital
    - Orbital
    - SubShell
    - Shell
"""

from typing import Any, List, Optional


class SpinOrbital:
    """
    Representation of a single-electron spin–orbital defined by the four
    quantum numbers (n, l, m, s).

    A spin–orbital is the most fundamental one-electron quantum state in the
    central-field approximation. It corresponds to the direct product:

        spatial orbital  ×  spin eigenstate

    and is fully specified by:

        * n — principal quantum number
        * l — orbital angular momentum quantum number
        * m — magnetic quantum number (projection of L on z)
        * s — spin quantum number (+1/2 or −1/2)

    Each SpinOrbital object is unique and corresponds to a specific location
    in the complete electron state basis used by the atom model.

    :param n: Principal quantum number. Must satisfy ``n ≥ 1``.
    :type n: int

    :param l: Orbital angular momentum quantum number. Must satisfy
        ``0 ≤ l < n``.
    :type l: int

    :param m: Magnetic quantum number (``−l ≤ m ≤ l``).
    :type m: int

    :param s: Spin projection quantum number (``+0.5`` or ``-0.5``).
    :type s: float

    :param occupied: Whether this spin–orbital currently hosts an electron.
    :type occupied: bool

    :param data: Optional storage for user-defined numerical metadata.
    :type data: Optional[Any]

    :returns: None
    :rtype: None
    """

    def __init__(self, n: int, l: int, m: int, s: float) -> None:
        if n < 1:
            raise ValueError("n must be ≥ 1")
        if not (0 <= l < n):
            raise ValueError("0 ≤ l < n must hold")
        if not (-l <= m <= l):
            raise ValueError("−l ≤ m ≤ l must hold")
        if s not in (0.5, -0.5):
            raise ValueError("s must be ±0.5")

        self.n = n  # principal quantum number
        self.l = l  # angular momentum quantum number
        self.m = m  # orientation quantum number
        self.s = s  # spin projection

        self.occupied: bool = False  # True if the spin–orbital is filled
        self.data: Optional[Any] = None  # optional numerical data storage

    def __repr__(self) -> str:
        return f"So(n={self.n},l={self.l},m={self.m},s={self.s},occ={self.occupied})"


class Orbital:
    """
    A spatial orbital consisting of exactly two spin–orbitals:
    one with spin ``+1/2`` and one with spin ``−1/2``.

    An Orbital groups the two SpinOrbital objects that share the same spatial
    quantum numbers (n, l, m) but differ in spin. This matches the central-field
    model in which every spatial wavefunction admits two allowed spin states.

    :param spin_up: The spin–orbital with ``s = +0.5``. Must have the same
        ``n, l, m`` quantum numbers as ``spin_down``.
    :type spin_up: SpinOrbital

    :param spin_down: The spin–orbital with ``s = -0.5``. Must have the same
        ``n, l, m`` quantum numbers as ``spin_up``.
    :type spin_down: SpinOrbital
    """

    def __init__(self, spin_up: SpinOrbital, spin_down: SpinOrbital) -> None:
        if spin_up.s != 0.5 or spin_down.s != -0.5:
            raise ValueError("Orbital requires +0.5 and -0.5 spin states")
        if not (
            spin_up.n == spin_down.n
            and spin_up.l == spin_down.l
            and spin_up.m == spin_down.m
        ):
            raise ValueError("Both electrons must share n, l, m")

        self.n = spin_up.n  # principal quantum number
        self.l = spin_up.l  # angular momentum
        self.m = spin_up.m  # magnetic quantum number

        self.spin_up = spin_up
        self.spin_down = spin_down

    def __repr__(self) -> str:
        return f"Orb(n={self.n},l={self.l},m={self.m})"


class SubShell:
    """
    A subshell defined by quantum numbers (n, l), containing all orbitals
    with magnetic quantum numbers ``m = −l, …, +l``.

    In the central-field model, a subshell represents all orbitals that share
    the same radial and angular parts of the wavefunction. Each allowed m-value
    corresponds to one spatial orbital, each of which supports two spin states.

    Thus, a subshell contains:

        * (2l + 1) spatial orbitals
        * 2 × (2l + 1) possible spin–orbitals

    :param n: Principal quantum number. Must be at least 1.
    :type n: int

    :param l: Orbital angular momentum quantum number. Must satisfy
        ``0 ≤ l < n``. Determines the subshell type (s, p, d, f ...).
    :type l: int
    """

    def __init__(self, n: int, l: int) -> None:
        if n < 1 or not (0 <= l < n):
            raise ValueError("Invalid subshell specification")

        self.n = n  # principal quantum number
        self.l = l  # angular momentum
        self.orbitals: List[Orbital] = []  # list of orbitals for each m

        for m in range(-l, l + 1):
            up = SpinOrbital(n=n, l=l, m=m, s=+0.5)
            down = SpinOrbital(n=n, l=l, m=m, s=-0.5)
            self.orbitals.append(Orbital(spin_up=up, spin_down=down))

    def _capacity(self) -> int:
        """Return the maximum number of electrons that can occupy this subshell.

        Explanation
        -----------
        A subshell with angular momentum quantum number ``l`` contains:

            * ``2l + 1`` spatial orbitals, one for each allowed magnetic
              quantum number ``m ∈ {−l, …, +l}``
            * each spatial orbital supports **two** spin states
              (``s = +1/2`` and ``s = −1/2``)

        Therefore, the theoretical electron capacity of a subshell is:

            ``capacity = 2 × (2l + 1)``

        In this implementation, the same value is computed as:

            ``2 × len(self.orbitals)``

        because ``self.orbitals`` contains exactly one ``Orbital`` instance
        per allowed ``m`` value. Thus:

            ``len(self.orbitals) == (2l + 1)``

        making the two expressions mathematically identical. Using
        ``2 * len(self.orbitals)`` ensures that the capacity always matches
        the *actual constructed structure* and remains robust even if future
        versions introduce modifications (e.g., relativistic splitting or
        constrained orbital sets).

        Returns
        -------
        int
            Maximum number of electrons allowed in this subshell.
        """
        return 2 * len(self.orbitals)

    def __repr__(self) -> str:
        return f"Sub(n={self.n},l={self.l})"


class Shell:
    """
    A shell representing all subshells with the same principal quantum
    number ``n``. Each subshell contains its full complement of orbitals
    and spin–orbitals. A Shell therefore holds a complete set of quantum
    states for a given energy level.

    Parameters
    ----------
    :param n: Principal quantum number of the shell. Must be at least 1.
    :type n: int
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("Shell n must be ≥ 1")

        self.n = n  # principal quantum number
        self.subshells: List[SubShell] = [SubShell(n=n, l=l) for l in range(n)]

    def __repr__(self) -> str:
        return f"Shell(n={self.n})"
