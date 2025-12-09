"""
Quantum-structure model of atoms using nested:
Shell → SubShell → Orbital → SpinOrbital.

Empirical reference:
NIST Atomic Spectra Database (ASD)
https://physics.nist.gov/PhysRefData/ASD/
"""

from typing import Any, Dict, List, Optional, Tuple

from q_block.aufbau_exceptions import EMPIRICAL_EXCEPTIONS


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

    Parameters
    ----------
    n : int
        Principal quantum number. Must satisfy ``n ≥ 1``. Determines the
        energy level and radial extent of the orbital.

    l : int
        Orbital angular momentum quantum number. Must satisfy
        ``0 ≤ l < n``. Determines the shape of the spatial wavefunction.

    m : int
        Magnetic quantum number, representing the z-projection of angular
        momentum. Must satisfy ``−l ≤ m ≤ l``.

    s : float
        Spin projection quantum number. Must be either ``+0.5`` or ``−0.5``.
        Corresponds to eigenstates of ``S_z``.

    occupied : bool
        Whether this spin–orbital currently hosts an electron.

    data : Any or None
        Optional storage for user-defined numerical metadata. It may store
        additional numerical information (e.g., basis-set coefficients,
        orbital energies, SCF results).
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

    Parameters
    ----------
    spin_up : SpinOrbital
        The spin–orbital with ``s = +0.5``. Must have the same ``n, l, m``
        quantum numbers as ``spin_down``.

    spin_down : SpinOrbital
        The spin–orbital with ``s = −0.5``. Must have the same ``n, l, m``
        quantum numbers as ``spin_up``.

    Internal Variables
    ------------------
    n : int
        Principal quantum number. Must satisfy ``n ≥ 1``. Determines the
        energy level and radial extent of the orbital.

    l : int
        Orbital angular momentum quantum number. Must satisfy
        ``0 ≤ l < n``. Determines the shape of the spatial wavefunction.

    m : int
        Magnetic quantum number, representing the z-projection of angular
        momentum. Must satisfy ``−l ≤ m ≤ l``.

    spin_up : SpinOrbital
        Electron state with spin +1/2.

    spin_down : SpinOrbital
        Electron state with spin −1/2.
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

    Parameters
    ----------
    n : int
        Principal quantum number. Must be at least 1.

    l : int
        Orbital angular momentum quantum number. Must satisfy
        ``0 ≤ l < n``. Determines the subshell type (s, p, d, f ...).

    Notes
    -----
    - The ``capacity()`` method returns the maximum number of electrons that
      the subshell can hold, equal to ``2 × (2l + 1)``.

    Internal Variables
    ------------------
    n : int
        Principal quantum number. Must satisfy ``n ≥ 1``. Determines the
        energy level and radial extent of the orbital.

    l : int
        Orbital angular momentum quantum number. Must satisfy
        ``0 ≤ l < n``. Determines the shape of the spatial wavefunction.

    orbitals : list[Orbital]
        List of Orbitals corresponding to each allowed m-value.
        Length is ``2l + 1``.
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

    def capacity(self) -> int:
        """Maximum electrons = 2 × (2l + 1)."""
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
    n : int
        Principal quantum number of the shell. Must be at least 1.

    Internal Variables
    ------------------
    n : int
        Principal quantum number. Must satisfy ``n ≥ 1``. Determines the
        energy level and radial extent of the orbital.

    subshells : list[SubShell]
        All subshells for this shell, with angular momentum quantum numbers
        l = 0 … n−1.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("Shell n must be ≥ 1")

        self.n = n  # principal quantum number
        self.subshells: List[SubShell] = [SubShell(n=n, l=l) for l in range(n)]

    def __repr__(self) -> str:
        return f"Shell(n={self.n})"


class Atom:
    """
    Represents an atom composed of shells, subshells, orbitals, and spin–orbitals.
    """

    def __init__(
        self,
        Z: int,
        n_max: int = 7,
        use_empirical_exceptions: bool = True,
        empirical_exceptions: Dict[int, List[Dict[str, int]]] = EMPIRICAL_EXCEPTIONS,
    ) -> None:
        """
        Initialize an Atom object containing nested shells, subshells, orbitals,
        and spin–orbitals up to a chosen maximum principal quantum number.

        This constructor builds the complete quantum-mechanical structure of an
        atom in the nonrelativistic, central-field approximation, where each
        electron is represented by a SpinOrbital object characterized by the
        quantum numbers (n, l, m, s).

        The constructed structure includes:
            - Shell(n):       all subshells for a given principal number n
            - SubShell(n,l):  all orbitals with angular momentum quantum number l
            - Orbital(n,l,m): a pair of opposite-spin spin-orbitals
            - SpinOrbital:    a single electron state (n,l,m,s)

        No electrons are assigned at initialization; occupation occurs only in
        `fill()`, which applies either the Aufbau+Hund rules or empirical
        exceptions based on real experimental ground-state configurations.

        Parameters
        ----------
        Z : int
            Atomic number (number of electrons in the neutral ground state).
            Must be ≥ 0. If Z=0, the atom contains empty shells.

        n_max : int, optional
            Maximum principal quantum number to construct. Determines how many
            shells are created. Default is 7, which is sufficient for all known
            chemical elements.

        use_empirical_exceptions : bool, optional
            If True, electron filling will use high-accuracy experimental
            ground-state configurations for elements where the pure Aufbau
            principle fails (e.g., Cr, Cu, Ag, Au, etc.).
            If False, filling strictly follows the theoretical Aufbau ordering
            without incorporating these corrections. Default is True.

        empirical_exceptions : Dict[int, List[Dict[str, int]]], optional
            Dictionary containing experimentally verified ground-state electron
            configurations for elements whose true electron distributions deviate from
            the Aufbau principle.

        Notes
        -----
        - All quantum objects are pre-constructed so that `fill()` only marks
          specific spin-orbitals as occupied.
        - Internal state is deterministic: calling `fill()` always resets all
          occupancy flags before reassigning electrons.
        - The atom is neutral; ions can be modeled by modifying Z prior to calling
          `fill()`.
        """

        if Z < 0:
            raise ValueError("Z must be ≥ 0")

        self.Z = Z  # electron count
        self.n_max = n_max  # max principal number
        self.use_empirical = use_empirical_exceptions
        self.empirical_exceptions = empirical_exceptions

        self.shells: Dict[int, Shell] = {
            n: Shell(n=n) for n in range(1, n_max + 1)
        }  # shells indexed by n

        self._all_subshells: List[SubShell] = []  # flattened subshell list
        for shell in self.shells.values():
            self._all_subshells.extend(shell.subshells)

    @staticmethod
    def _aufbau_key(sub: SubShell) -> Tuple[int, int]:
        """Sorting key (n+l, n) for Aufbau filling."""
        return (sub.n + sub.l, sub.n)

    def _reset(self) -> None:
        """
        Reset all spin–orbitals to unoccupied state.

        Why this is necessary
        ----------------------
        The Atom object is *stateful*: once `fill()` is called, each
        `SpinOrbital` stores whether it is occupied. If the user calls
        `fill()` again—possibly after changing Z or switching exceptions—
        the previous occupation must NOT persist.

        Without reset():
        - electrons would accumulate with each call
        - configurations would become invalid
        - repeated `.fill()` would produce different results for the same input

        This method guarantees that every call to `fill()` begins from a
        completely clean state, ensuring deterministic and physically
        correct behavior.
        """
        for sub in self._all_subshells:
            for orb in sub.orbitals:
                orb.spin_up.occupied = False
                orb.spin_down.occupied = False

    def _apply_exception(self, instructions: List[Dict[str, int]]) -> None:
        """
        Apply empirical NIST-based electron configuration exceptions.

        Parameters
        ----------
        instructions : list of dict
            Each dictionary has keys {"n", "l", "electron_count"}.
        """
        for entry in instructions:
            n = entry["n"]
            l = entry["l"]
            count = entry["electron_count"]

            # locate subshell
            subshell = None
            for ss in self._all_subshells:
                if ss.n == n and ss.l == l:
                    subshell = ss
                    break
            if subshell is None:
                raise ValueError(f"Missing subshell n={n}, l={l}")

            orbitals = subshell.orbitals
            num_orb = len(orbitals)

            first = min(num_orb, count)
            for i in range(first):
                orbitals[i].spin_up.occupied = True

            second = min(num_orb, count - first)
            for i in range(second):
                orbitals[i].spin_down.occupied = True

    def fill(self) -> Dict[int, Shell]:
        """
        Fill electrons according to Aufbau and Hund rules.

        Returns
        -------
        dict[int, Shell]
            Mapping from principal quantum number to Shell instance.
        """

        self._reset()

        # --- EMPIRICAL OVERRIDE PATH (NIST EXCEPTIONS) ---
        if self.use_empirical and self.Z in self.empirical_exceptions:
            self._apply_exception(instructions=self.empirical_exceptions[self.Z])
            return self.shells

        remaining = self.Z

        # ==================================================================
        # A U F B A U   P R I N C I P L E
        # ------------------------------------------------------------------
        # The sorted(...) call below *is* the Aufbau principle:
        #
        #     Fill orbitals in order of increasing (n + l),
        #     and for equal (n + l), fill lower n first.
        #
        # This ordering corresponds to the typical Aufbau sequence:
        # 1s → 2s → 2p → 3s → 3p → 4s → 3d → 4p → 5s → …
        #
        # Note: exceptions of Aufbau principle are implemented above
        #
        # ==================================================================
        for subshell in sorted(
            self._all_subshells, key=lambda ss: Atom._aufbau_key(sub=ss)
        ):
            if remaining <= 0:
                break

            orbitals = subshell.orbitals
            num_orb = len(orbitals)
            subshell_capacity = subshell.capacity()

            electrons = min(subshell_capacity, remaining)

            # ==================================================================
            # H U N D ’ S   R U L E   (FIRST STAGE)
            # ------------------------------------------------------------------
            # Assign all electrons with spin +1/2 first, one per orbital.
            #
            # Hund’s rule:
            #     "Electrons singly occupy degenerate orbitals with parallel
            #      spins before any pairing occurs."
            #
            # This block ONLY fills +1/2 orbitals (single-occupancy stage).
            # ==================================================================
            first = min(num_orb, electrons)
            for i in range(first):
                orbitals[i].spin_up.occupied = True
            remaining -= first

            # ==================================================================
            # H U N D ’ S   R U L E   (SECOND STAGE)
            # ------------------------------------------------------------------
            # After all orbitals in the subshell have one electron,
            # the remaining electrons pair with spin –1/2.
            #
            # This corresponds to the “pairing stage” of Hund’s rule.
            # ==================================================================
            second = min(num_orb, electrons - first)
            for i in range(second):
                orbitals[i].spin_down.occupied = True
            remaining -= second

        return self.shells
