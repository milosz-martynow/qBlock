"""
Quantum-structure model of atoms using nested:
Shell → SubShell → Orbital → SpinOrbital.

Empirical reference:
NIST Atomic Spectra Database (ASD)
https://physics.nist.gov/PhysRefData/ASD/
"""

from typing import Any, Dict, List, Optional, Tuple

from q_block.atoms_data import EMPIRICAL_EXCEPTIONS


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
    :type data: Any | None

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


class Atom:
    """
    Represents an atom composed of shells, subshells, orbitals, and spin–orbitals.
    """

    def __init__(
        self,
        Z: int,
        n_max: int = 7,
        use_empirical_exceptions: bool = True,
        empirical_exceptions: Dict[
            int, List[Dict[str, int]]
        ] = EMPIRICAL_EXCEPTIONS,
        basis_set: Optional[Dict[str, Any]] = None,
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
        `fill_occupancy()`, which applies either the Aufbau+Hund rules or empirical
        exceptions based on real experimental ground-state configurations.

        :param Z: Atomic number (number of electrons in the neutral ground state).
        :type Z: int
        :raises ValueError: If ``Z < 0``.

        :param n_max: Maximum principal quantum number to construct. Determines
                how many shells are created. Default is 7.
        :type n_max: int

        :param use_empirical_exceptions: If True, electron filling will use
                empirical exceptions for certain elements; otherwise pure Aufbau is used.
        :type use_empirical_exceptions: bool

        :param empirical_exceptions: Dictionary with empirical configurations.
        :type empirical_exceptions: Dict[int, List[Dict[str, int]]]

        :param basis_set: Optional basis-set dictionary to attach to the Atom.
        :type basis_set: Optional[Dict[str, Any]]

        Notes
        -----
        - All quantum objects are pre-constructed so that ``fill_occupancy()`` only marks
            specific spin-orbitals as occupied.
        - Internal state is deterministic: calling ``fill_occupancy()`` always resets all
            occupancy flags before reassigning electrons.
        - The atom is neutral; ions can be modeled by modifying Z prior to calling
            ``fill_occupancy()``.
        """

        if Z < 0:
            raise ValueError("Z must be ≥ 0")

        self.Z = Z  # electron count
        self.n_max = n_max  # max principal number
        self.use_empirical = use_empirical_exceptions
        self.empirical_exceptions = empirical_exceptions

        # Helper variables
        self.shells: Dict[int, Shell] = {
            n: Shell(n=n) for n in range(1, n_max + 1)
        }  # shells indexed by n

        self.basis_set = basis_set

    @property
    def _all_subshells(self) -> List[SubShell]:
        """Return a flattened list of subshells in shell order (n ascending)."""
        return [ss for shell in self.shells.values() for ss in shell.subshells]

    @staticmethod
    def _aufbau_key(sub: SubShell) -> Tuple[int, int]:
        """Compute the Aufbau sorting key for a subshell.

        Parameters
        ----------
        sub : SubShell
            Subshell for which the (n + l, n) key is computed.

        Returns
        -------
        Tuple[int, int]
            Sorting key `(n + l, n)` used to implement the Aufbau ordering.
        """
        return (sub.n + sub.l, sub.n)

    def _reset(self) -> None:
        """Reset all spin–orbitals to the unoccupied state.

        This method mutates all contained :class:`SpinOrbital` objects by
        setting their ``occupied`` attribute to ``False``.

        :returns: None
        :rtype: None
        """
        for sub in self._all_subshells:
            for orb in sub.orbitals:
                orb.spin_up.occupied = False
                orb.spin_down.occupied = False

    def _apply_exception(self, instructions: List[Dict[str, int]]) -> None:
        """Apply an empirical configuration specified by ``instructions``.

        :param instructions: List of subshell assignment dictionaries with
            keys ``"n"``, ``"l"``, and ``"electron_count"`` describing
            how many electrons to place in the given subshell.
        :type instructions: List[Dict[str, int]]

        :raises ValueError: If a referenced subshell does not exist in the
            prebuilt atom.

        :returns: None
        :rtype: None
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

    def fill_occupancy(self) -> None:
        """Assign electrons to spin-orbitals following Aufbau and Hund rules.

        This method mutates the internal occupancy flags of
        :class:`SpinOrbital` objects stored in this :class:`Atom`.

        When empirical exceptions are enabled and an entry exists for
        ``self.Z``, the empirical mapping is applied instead of the pure
        Aufbau filling.

        :returns: None
        :rtype: None
        """

        self._reset()

        # ==================================================================
        # E M P I R I C A L   O V E R R I D E   P A T H
        # ------------------------------------------------------------------
        # Empirical instructions mutate existing spin-orbitals in-place.
        # When applicable, apply the empirical mapping and skip Aufbau.
        skip_aufbau = (
            self.use_empirical and self.Z in self.empirical_exceptions
        )
        if skip_aufbau:
            self._apply_exception(
                instructions=self.empirical_exceptions[self.Z]
            )

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
        if not skip_aufbau:
            remaining = self.Z

            for subshell in sorted(
                self._all_subshells, key=lambda ss: Atom._aufbau_key(sub=ss)
            ):
                if remaining <= 0:
                    break

                orbitals = subshell.orbitals
                num_orb = len(orbitals)
                subshell_capacity = subshell._capacity()

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

    def populate_spinorbitals_with_gto(self) -> None:
        """Attach Gaussian-type orbital (GTO) basis data to occupied spin-orbitals.

        The method mutates ``SpinOrbital.data`` for occupied spin-orbitals.

        :returns: None
        :rtype: None

        Notes:
            - Requires ``self.basis_set`` to be a regions-structured dictionary
              (see :mod:`q_block.basis_set_pople`). If ``basis_set`` is
              ``None`` this function returns without changes.
            - Calls :meth:`fill_occupancy` internally to ensure occupations
              exist.
            - Sets ``SpinOrbital.data`` for each occupied spin-orbital to a
              dictionary with keys ``"exponents"`` and ``"contractions"`
              (both lists of floats).

        SpinOrbital.data schema::

            {
                "exponents": List[float],
                "contractions": List[float],
            }
        """

        if self.basis_set is None:
            return

        # Ensure occupations exist
        self.fill_occupancy()

        # ------------------------------------------------------------
        # Group occupied subshells by angular momentum
        # ------------------------------------------------------------
        occupied_by_l: Dict[int, List[SubShell]] = {}

        for subshell in self._all_subshells:
            if any(
                so.occupied
                for orb in subshell.orbitals
                for so in (orb.spin_up, orb.spin_down)
            ):
                occupied_by_l.setdefault(subshell.l, []).append(subshell)

        for subshells in occupied_by_l.values():
            subshells.sort(key=lambda ss: (ss.n + ss.l, ss.n))

        # ------------------------------------------------------------
        # Assign basis shells
        # ------------------------------------------------------------
        for l, subshells in occupied_by_l.items():

            # Collect basis shells for this l
            basis_shells: List[Dict[str, List[float]]] = []
            for region in ("core", "valence_inner", "valence_outer"):
                basis_shells.extend(self.basis_set.get(region, {}).get(l, []))

            if len(basis_shells) < len(subshells):
                raise ValueError(
                    f"Insufficient basis shells for l={l}: "
                    f"{len(basis_shells)} < {len(subshells)}"
                )

            # --------------------------------------------------------
            # Split-valence aware assignment
            # --------------------------------------------------------
            assignments: Dict[SubShell, List[Dict[str, List[float]]]] = {
                ss: [] for ss in subshells
            }

            # One shell for each inner subshell
            for ss, basis in zip(subshells[:-1], basis_shells):
                assignments[ss].append(basis)

            # Remaining shells go to outermost subshell
            for basis in basis_shells[len(subshells) - 1 :]:
                assignments[subshells[-1]].append(basis)

            # --------------------------------------------------------
            # Populate SpinOrbitals
            # --------------------------------------------------------
            for subshell, shells in assignments.items():

                exponents: List[float] = []
                contractions: List[float] = []

                for sh in shells:
                    exponents.extend(sh["exponents"])
                    contractions.extend(sh["coefficients"])

                for orbital in subshell.orbitals:
                    for so in (orbital.spin_up, orbital.spin_down):
                        if so.occupied:
                            so.data = {
                                "exponents": exponents,
                                "contractions": contractions,
                            }
