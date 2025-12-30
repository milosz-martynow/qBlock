"""Molecular atomic system container.

This module defines the :class:`Molecule` class, a minimal placeholder
for molecular systems that extends :class:`q_block.atomic_system.AtomicSystem`.
"""

from __future__ import annotations

from typing import Optional, Dict, List

from .atomic_system import AtomicSystem
from .input_data import InputData
from .electron import SubShell
from .atom import Atom


class Molecule(AtomicSystem):
    """Container for atoms forming a single molecule.

    This class is a specialisation of
    :class:`q_block.atomic_system.AtomicSystem` intended to represent
    molecular systems. Upon initialisation, it extracts all
    :class:`Atom` instances from the provided :class:`InputData` and
    immediately populates their spin-orbitals with GTO data if a
    ``basis_set`` is attached to a given atom.

    :param input_data: Optional :class:`InputData` instance describing
        the atoms in the molecular system.
    :type input_data: Optional[InputData]
    """

    def __init__(self, input_data: Optional[InputData] = None) -> None:
        super().__init__(input_data=input_data)

        # Cache atoms list for convenience
        self.atoms: List[Atom] = list(self.input_data.atoms["atom"])
        # Automatically populate GTO data for all atoms that have a basis set
        self.populate_spinorbitals_with_gto()

    def populate_spinorbitals_with_gto(self) -> None:
        """Attach GTO basis data to occupied spin-orbitals for all atoms.

        This method iterates over all :class:`Atom` objects stored in the
        :attr:`atoms` list and, for each atom that has a non-``None``
        ``basis_set``, attaches Gaussian-type orbital (GTO) data to
        occupied spin-orbitals using the same algorithm previously
        implemented on :class:`Atom`.

        Atoms without an associated ``basis_set`` are left unchanged.
        """

        for atom in self.atoms:
            if atom.basis_set is None:
                continue

            # ------------------------------------------------------------
            # Group occupied subshells by angular momentum
            # ------------------------------------------------------------
            occupied_by_l: Dict[int, List[SubShell]] = {}

            for subshell in atom._all_subshells:
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
                    basis_shells.extend(
                        atom.basis_set.get(region, {}).get(l, [])  # type: ignore[union-attr]
                    )

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
