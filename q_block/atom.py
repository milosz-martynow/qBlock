"""
Quantum-structure model of atoms using nested:
Shell → SubShell → Orbital → SpinOrbital.

Empirical reference:
NIST Atomic Spectra Database (ASD)
https://physics.nist.gov/PhysRefData/ASD/
"""

from typing import Any, Dict, List, Optional, Tuple

from q_block.atoms_data import EMPIRICAL_EXCEPTIONS

# Electron state classes have been moved to `electron.py` to reflect
# the singular module name. Keep imports stable for the rest of the package.
from .electron import Orbital, Shell, SpinOrbital, SubShell


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
