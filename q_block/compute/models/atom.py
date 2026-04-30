"""Quantum-structure model of atoms using nested:
Shell  SubShell  Orbital  SpinOrbital.

Empirical reference:
NIST Atomic Spectra Database (ASD)
https://physics.nist.gov/PhysRefData/ASD/
"""

from typing import Dict, List, Optional, Tuple

from q_block.compute.environment.constants.natural.atoms_data import (
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
    EMPIRICAL_EXCEPTIONS,
)
from q_block.compute.environment.io.basis_set import BasisSet
from q_block.compute.environment.io.coordinates import CartesianCoordinates
from q_block.compute.models.basis_functions import ContractedGaussianTypeOrbital
from q_block.compute.models.electron import Shell


class Atom:
    """Represents an atom composed of shells, subshells, orbitals, and spin-orbitals."""

    def __init__(
        self,
        atomic_number: int,
        maximal_principal_quantum_number: int = 7,
        empirical_exceptions: Optional[
            Dict[int, List[Dict[str, int]]]
        ] = EMPIRICAL_EXCEPTIONS,
        basis_set: Optional[BasisSet] = None,
        coordinates: Optional[CartesianCoordinates] = None,
        charge: int = 0,
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

        :param basis_set: Optional :class:`~compute.io.basis_set.BasisSet`
            instance (e.g. :class:`~compute.io.basis_set.Pople`).  Per-element
            data can be retrieved via ``basis_set[atomic_number]``.
        :type basis_set: Optional[BasisSet]

        :param coordinates: Optional CartesianCoordinates associated with the atom.
        :type coordinates: Optional[CartesianCoordinates]

        :param charge: Formal charge on this atom.  Default ``0``
            (neutral atom).  The total system charge is the sum of all
            per-atom charges.
        :type charge: int

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

        # Formal charge on this atom (default 0 = neutral)
        self.charge: int = charge

        # Coordinates (CartesianCoordinates instance or None)
        self.coordinates: Optional[CartesianCoordinates] = coordinates

        # Helper variables
        self.shells: Dict[int, Shell] = {
            n: Shell(n=n) for n in range(1, maximal_principal_quantum_number + 1)
        }
        self.basis_set: Optional[BasisSet] = basis_set

        # Open-shell flag: True if atom has unpaired electrons, False if closed-shell
        # This is set automatically by fill_occupancy() based on electron configuration
        self.open_shell: Optional[bool] = None

        # CGTO basis functions — populated by make_contracted_gaussian_type_orbital()
        self.contracted_gaussian_type_orbitals: Optional[
            List[ContractedGaussianTypeOrbital]
        ] = None

        # Build the ground-state electronic configuration immediately.
        self.fill_occupancy()

    @property
    def _all_subshells(self) -> list:
        """Return a flattened list of subshells in shell order (n ascending).

        :returns: List of all subshells belonging to this atom, ordered by
            increasing principal quantum number ``n`` and the insertion
            order within each :class:`Shell`.
        :rtype: list
        """
        return [ss for shell in self.shells.values() for ss in shell.subshells]

    @property
    def n_electrons(self) -> int:
        """Number of electrons on this atom accounting for formal charge.

        For a neutral atom (:attr:`charge` ``= 0``) this equals the
        atomic number :math:`Z`.  A negative charge (anion) adds
        electrons and a positive charge (cation) is stored as a
        positive value but the sign convention of :attr:`charge`
        already encodes the direction:

        .. math:: N = Z + q

        :returns: Number of electrons.
        :rtype: int
        """
        return self.atomic_number + self.charge

    @staticmethod
    def _aufbau_key(sub: object) -> Tuple[int, int]:
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

        This method delegates to :meth:`Shell.reset_occupancy` for each
        shell owned by this atom.
        """
        for shell in self.shells.values():
            shell.reset_occupancy()

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

            # locate subshell via Shell
            shell = self.shells.get(n)
            if shell is None:
                raise ValueError(f"Missing shell n={n}")
            subshell = shell.find_subshell(l)
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
        n_unpaired = sum(shell.count_unpaired() for shell in self.shells.values())

        self.open_shell = n_unpaired > 0

    def make_contracted_gaussian_type_orbital(
        self,
        atom_index: Optional[int] = None,
    ) -> None:
        """Build CGTO basis functions from this atom's basis set and coordinates.

        Walks through the hierarchical basis-set structure
        (``core → valence_inner → valence_outer``) and creates one
        :class:`~compute.models.basis_functions.ContractedGaussianTypeOrbital`
        per contracted shell.  The results are stored in
        :attr:`contracted_gaussian_type_orbitals`.

        Each CGTO carries the atom's coordinates as its centre.  If the
        atom has no coordinates or no basis set, the method is a no-op
        (an empty list is stored).

        **Ordering:** regions (core → valence_inner → valence_outer),
        ascending :math:`l`, shell order within each :math:`l`.

        :param atom_index: Optional molecule-level atom index to stamp
            onto every CGTO.  When called from
            :meth:`Molecule.make_contracted_gaussian_type_orbital` this
            is the position of the atom in :attr:`Molecule.atoms`.
        :type atom_index: Optional[int]

        """
        if (
            self.basis_set is None
            or self.atomic_number not in self.basis_set
            or self.coordinates is None
        ):
            self.contracted_gaussian_type_orbitals = []
            return

        center = CartesianCoordinates(
            x=self.coordinates.x,
            y=self.coordinates.y,
            z=self.coordinates.z,
        )
        regions = self.basis_set[self.atomic_number]

        cgtos: List[ContractedGaussianTypeOrbital] = []

        for region_name in ("core", "valence_inner", "valence_outer"):
            region = regions.get(region_name, {})

            for l_val in sorted(region.keys()):
                shells = region[l_val]

                for shell in shells:
                    cgtos.append(
                        ContractedGaussianTypeOrbital(
                            center=center,
                            l=l_val,
                            exponents=shell["exponents"],
                            contractions=shell["coefficients"],
                            atom_index=atom_index,
                        )
                    )

        self.contracted_gaussian_type_orbitals = cgtos

    def to_bohr(self) -> None:
        """Convert this atom's coordinates from Ångström to Bohr in place.

        Uses :meth:`CartesianCoordinates.to_bohr` to create a new
        :class:`CartesianCoordinates` instance in Bohr and replaces
        ``self.coordinates`` with it.

        If the atom has no coordinates (``None``), the call is a no-op.

        :raises TypeError: If the coordinates are not of type
            :class:`CartesianCoordinates`.
        """
        if self.coordinates is None:
            return
        if not isinstance(self.coordinates, CartesianCoordinates):
            raise TypeError(
                f"Only CartesianCoordinates can be converted to Bohr; "
                f"got {type(self.coordinates).__name__}."
            )
        self.coordinates = self.coordinates.to_bohr()

    def __repr__(self) -> str:
        if (
            self.coordinates is not None
            and self.coordinates.x is not None
            and self.coordinates.y is not None
            and self.coordinates.z is not None
        ):
            coord_str = f" coords=({self.coordinates.x:.3f},{self.coordinates.y:.3f},{self.coordinates.z:.3f})"
        else:
            coord_str = ""
        return (
            f"Atom(atomic_number={self.atomic_number}, "
            f"symbol={ATOMS_SYMBOLS_Z_TO_SYMBOL[self.atomic_number]}{coord_str})"
        )
