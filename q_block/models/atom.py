"""Quantum-structure model of atoms using nested:
Shell  SubShell  Orbital  SpinOrbital.

Empirical reference:
NIST Atomic Spectra Database (ASD)
https://physics.nist.gov/PhysRefData/ASD/
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from q_block.constants.atoms_data import (
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
    EMPIRICAL_EXCEPTIONS,
)
from q_block.io.coordinates import CartesianCoordinates
from q_block.models.electron import Orbital, Shell, SpinOrbital, SubShell


class Atom:
    """Represents an atom composed of shells, subshells, orbitals, and spin-orbitals."""

    def __init__(
        self,
        atomic_number: int,
        maximal_principal_quantum_number: int = 7,
        empirical_exceptions: Optional[Dict[int, List[Dict[str, int]]]] = EMPIRICAL_EXCEPTIONS,
        basis_set: Optional[Dict[str, Any]] = None,
        coordinates: Optional[CartesianCoordinates] = None,
    ) -> None:
        """Initialize an Atom object containing nested shells, subshells, orbitals,
        and spin-orbitals up to a chosen maximum principal quantum number.

        This constructor builds the complete quantum-mechanical structure of an
        atom in the nonrelativistic, central-field approximation, where each
        electron is represented by a SpinOrbital object characterized by the
        quantum numbers (n, l, m, s).

        The constructed structure includes:
            - Shell(n):       all subshells for a given principal number n
            - SubShell(n,l):  all orbitals with angular momentum quantum number l
            - Orbital(n,l,m): a pair of opposite-spin spin-orbitals
            - SpinOrbital:    a single electron state (n,l,m,s)

        After constructing the orbital manifold, the method immediately calls
        :meth:`fill_occupancy` to assign electrons according to either the
        Aufbau+Hund rules or empirical exceptions based on real experimental
        ground-state configurations.

        :param atomic_number: Atomic number (number of electrons in the neutral ground state).
        :type atomic_number: int
        :raises ValueError: If ``atomic_number < 0``.

        :param maximal_principal_quantum_number: Maximum principal quantum number to construct. Determines
                how many shells are created. Default is 7.
        :type maximal_principal_quantum_number: int

        :param empirical_exceptions: If not None, electron filling will use
                empirical exceptions for certain elements; otherwise pure Aufbau is used.
        :type empirical_exceptions: Optional[Dict[int, List[Dict[str, int]]]]

        :param basis_set: Optional basis-set dictionary to attach to the Atom.
        :type basis_set: Optional[Dict[str, Any]]

        :param coordinates: Optional CartesianCoordinates associated with the atom.
        :type coordinates: Optional[CartesianCoordinates]

        Notes
        -----
        - All quantum objects are pre-constructed so that ``fill_occupancy()`` only marks
          specific spin-orbitals as occupied.
        - Internal state is deterministic: calling ``fill_occupancy()`` always resets all
          occupancy flags before reassigning electrons.
        - The atom is neutral; ions can be modeled by modifying Z prior to calling
          ``fill_occupancy()``.
        """

        if atomic_number < 0:
            raise ValueError("atomic_number must be >= 0")

        self.atomic_number = atomic_number
        self.maximal_principal_quantum_number = maximal_principal_quantum_number
        self._empirical_exceptions = empirical_exceptions

        # Coordinates (CartesianCoordinates instance or None)
        self.coordinates: Optional[CartesianCoordinates] = coordinates

        # Helper variables
        self.shells: Dict[int, Shell] = {
            n: Shell(n=n) for n in range(1, maximal_principal_quantum_number + 1)
        }
        self.basis_set = basis_set
        
        # Open-shell flag: True if atom has unpaired electrons, False if closed-shell
        # This is set automatically by fill_occupancy() based on electron configuration
        self.open_shell: Optional[bool] = None

        # Build the ground-state electronic configuration immediately.
        self.fill_occupancy()

    @property
    def _all_subshells(self) -> List[SubShell]:
        """Return a flattened list of subshells in shell order (n ascending).

        :returns: List of all subshells belonging to this atom, ordered by
            increasing principal quantum number ``n`` and the insertion
            order within each :class:`Shell`.
        :rtype: List[SubShell]
        """
        return [ss for shell in self.shells.values() for ss in shell.subshells]

    @staticmethod
    def _aufbau_key(sub: SubShell) -> Tuple[int, int]:
        """Compute the Aufbau sorting key for a subshell.

        The key implements the standard Aufbau ordering by increasing
        ``(n + l)`` and, for equal ``(n + l)``, by increasing ``n``.

        :param sub: Subshell for which the ``(n + l, n)`` key is computed.
        :type sub: SubShell

        :returns: Sorting key ``(n + l, n)`` used to implement the Aufbau
            ordering.
        :rtype: Tuple[int, int]
        """
        return (sub.n + sub.l, sub.n)

    def _reset(self) -> None:
        """Reset all spin-orbitals to the unoccupied state.

        This method mutates all contained :class:`SpinOrbital` objects by
        setting their ``occupied`` attribute to ``False``.
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
        """
        for entry in instructions:
            n = entry["n"]
            l = entry["l"]
            count = entry["electron_count"]

            # locate subshell
            subshell: Optional[SubShell] = None
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
        ``self.atomic_number``, the empirical mapping is applied instead of the pure
        Aufbau filling.
        """
        self._reset()

        # ==================================================================
        # E M P I R I C A L   O V E R R I D E   P A T H
        # ------------------------------------------------------------------
        # Empirical instructions mutate existing spin-orbitals in-place.
        # When applicable, apply the empirical mapping and skip Aufbau.
        # ==================================================================
        skip_aufbau = (
            self._empirical_exceptions is not None
            and self.atomic_number in self._empirical_exceptions
        )
        if skip_aufbau:
            self._apply_exception(
                instructions=self._empirical_exceptions[self.atomic_number]
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
        # 1s  2s  2p  3s  3p  4s  3d  4p  5s  ...
        #
        # Note: exceptions of Aufbau principle are implemented above
        # ==================================================================
        if not skip_aufbau:
            remaining = self.atomic_number

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
                # H U N D ' S   R U L E   (FIRST STAGE)
                # ------------------------------------------------------------------
                # Assign all electrons with spin +1/2 first, one per orbital.
                #
                # Hund's rule:
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
                # H U N D ' S   R U L E   (SECOND STAGE)
                # ------------------------------------------------------------------
                # After all orbitals in the subshell have one electron,
                # the remaining electrons pair with spin -1/2.
                #
                # This corresponds to the "pairing stage" of Hund's rule.
                # ==================================================================
                second = min(num_orb, electrons - first)
                for i in range(second):
                    orbitals[i].spin_down.occupied = True
                remaining -= second

        # ==================================================================
        # D E T E R M I N E   O P E N - S H E L L   S T A T U S
        # ------------------------------------------------------------------
        # Count unpaired electrons (orbitals with only one spin occupied).
        # If any unpaired electrons exist, the atom is open-shell.
        # ==================================================================
        n_unpaired = 0
        for subshell in self._all_subshells:
            for orb in subshell.orbitals:
                up_occupied = orb.spin_up.occupied
                down_occupied = orb.spin_down.occupied
                if up_occupied and not down_occupied:
                    n_unpaired = n_unpaired + 1
                elif down_occupied and not up_occupied:
                    n_unpaired = n_unpaired + 1
        
        self.open_shell = (n_unpaired > 0)

    def __repr__(self) -> str:
        if (
            self.coordinates is not None
            and self.coordinates.x is not None
            and self.coordinates.y is not None
            and self.coordinates.z is not None
        ):
            coord_str = (
                f" coords=({self.coordinates.x:.3f},{self.coordinates.y:.3f},{self.coordinates.z:.3f})"
            )
        else:
            coord_str = ""
        return (
            f"Atom(atomic_number={self.atomic_number}, "
            f"symbol={ATOMS_SYMBOLS_Z_TO_SYMBOL[self.atomic_number]}{coord_str})"
        )
