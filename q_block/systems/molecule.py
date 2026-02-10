"""Molecular atomic system container.

This module defines the :class:`Molecule` class, a minimal placeholder
for molecular systems that extends :class:`q_block.atomic_system.AtomicSystem`.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Any, Union

import pandas as pd

from q_block.systems.atomic_system import AtomicSystem
from q_block.io.input_data import InputData
from q_block.io.coordinates import CartesianCoordinates
from q_block.io.basis_set import BasisSet
from q_block.models.electron import SubShell
from q_block.models.atom import Atom


class Molecule(AtomicSystem):
    """Container for atoms forming a single molecule.

    This class is a specialisation of
    :class:`q_block.atomic_system.AtomicSystem` intended to represent
    molecular systems. Upon initialisation, it extracts all
    :class:`Atom` instances from the provided :class:`InputData` and
    immediately populates their spin-orbitals with GTO data if a
    ``basis_set`` is attached to a given atom.

    A :class:`~q_block.io.basis_set.BasisSet` instance (e.g.
    :class:`~q_block.io.basis_set.Pople`) can be passed to the
    constructor via the ``basis_set`` parameter to automatically assign
    the correct per-element regions to every atom in the molecule before
    GTO population.

    :param input_data: Optional :class:`InputData` instance describing
        the atoms in the molecular system.
    :type input_data: Optional[InputData]
    :param basis_set: Optional :class:`BasisSet` instance.  When
        provided, the per-element basis set data is automatically
        assigned to each atom based on its symbol before GTO
        population.
    :type basis_set: Optional[BasisSet]
    :param multiplicity: Spin multiplicity :math:`2S+1`.  Default ``1``
        (singlet).
    :type multiplicity: int
    """

    def __init__(
        self,
        input_data: Optional[InputData] = None,
        basis_set: Optional[BasisSet] = None,
        multiplicity: int = 1,
    ) -> None:
        super().__init__(input_data=input_data, multiplicity=multiplicity)

        # Cache atoms list for convenience
        self.atoms: List[Atom] = list(self.input_data.atoms["atom"])

        # If a BasisSet object was provided, assign it to every atom
        if basis_set is not None:
            for atom in self.atoms:
                atom.basis_set = basis_set

        # Automatically populate GTO data for all atoms that have a basis set
        self.populate_spinorbitals_with_gto()

        # Validate basis sets (only when one was provided), electron count,
        # and multiplicity consistency
        if basis_set is not None:
            self._validate_basis_sets()
        self._validate_n_electrons()
        self._validate_multiplicity()

    @property
    def total_atomic_number(self) -> int:
        """Total number of protons (sum of atomic numbers) in the molecule.

        For a neutral molecule this equals the total number of electrons.
        To obtain the electron count for a charged system, subtract the
        charge from this value.

        :returns: Sum of atomic numbers over all atoms.
        :rtype: int
        """
        return sum(atom.atomic_number for atom in self.atoms)

    @property
    def n_electrons(self) -> int:
        """Total number of electrons in the molecule.

        Each :class:`~q_block.models.atom.Atom` already accounts for
        its own formal charge via :attr:`Atom.n_electrons`
        (:math:`Z_i - q_i`), so the molecule-level count is simply the
        sum over all atoms:

        .. math:: N = \\sum_i (Z_i - q_i)

        :returns: Total electron count.
        :rtype: int
        """
        return sum(atom.n_electrons for atom in self.atoms)

    def _validate_n_electrons(self) -> None:
        """Verify that the electron count is non-negative.

        :raises ValueError: If :attr:`n_electrons` is negative, meaning
            the charge exceeds the total neutral electron count.
        """
        if self.n_electrons < 0:
            raise ValueError(
                f"Negative electron count ({self.n_electrons}): charge "
                f"({self.charge}) exceeds total nuclear charge "
                f"({self.total_atomic_number})."
            )

    def _validate_multiplicity(self) -> None:
        """Validate that the multiplicity is consistent with the electron count.

        The number of unpaired electrons implied by the multiplicity is
        :math:`n_{unpaired} = 2S = M - 1` where :math:`M` is the
        multiplicity.  :math:`n_{unpaired}` must have the same parity as
        :math:`N_{elec}` and must not exceed :math:`N_{elec}`.

        :raises ValueError: On parity mismatch or impossible multiplicity.
        """
        if self.multiplicity < 1:
            raise ValueError(
                f"Multiplicity must be >= 1; got {self.multiplicity}."
            )
        n_unpaired = self.multiplicity - 1
        if (self.n_electrons - n_unpaired) % 2 != 0:
            raise ValueError(
                f"Multiplicity {self.multiplicity} is incompatible with "
                f"{self.n_electrons} electrons (parity mismatch)."
            )
        if n_unpaired > self.n_electrons:
            raise ValueError(
                f"Multiplicity {self.multiplicity} requires {n_unpaired} "
                f"unpaired electrons, but only {self.n_electrons} electrons "
                f"are present."
            )

    def to_bohr(self) -> None:
        """Convert all atom coordinates from Ångström to Bohr in place.

        Iterates over every :class:`Atom` in :attr:`atoms` and calls
        :meth:`Atom.to_bohr`, which replaces each atom's
        :class:`CartesianCoordinates` with a Bohr-scaled copy.

        :raises TypeError: If any atom's coordinates are not of type
            :class:`~q_block.io.coordinates.CartesianCoordinates`.
        """

        for atom in self.atoms:
            if atom.coordinates is not None and not isinstance(
                atom.coordinates, CartesianCoordinates
            ):
                raise TypeError(
                    f"Only CartesianCoordinates can be converted to Bohr; "
                    f"atom {atom!r} has {type(atom.coordinates).__name__}."
                )
            atom.to_bohr()

    def _validate_basis_sets(self) -> None:
        """Verify that every atom has a valid basis set attached.

        :raises ValueError: If any atom is missing basis-set data or
            its element is absent from the attached basis set.
        """
        for atom in self.atoms:
            if atom.basis_set is None:
                raise ValueError(
                    f"Atom {atom!r} has no BasisSet attached. "
                    f"Assign a BasisSet to each atom before passing "
                    f"the molecule to HartreeFock."
                )
            if atom.atomic_number not in atom.basis_set:
                raise ValueError(
                    f"Atom {atom!r} has a BasisSet attached but its "
                    f"element is not present in the basis set."
                )

    @property
    def n_basis(self) -> int:
        """Total number of contracted basis functions in the molecule.

        A "basis function" here is one contracted Gaussian shell for a
        given angular momentum :math:`l`, contributing :math:`2l+1`
        Cartesian/spherical functions.

        :returns: Total number of basis functions across all atoms.
        :rtype: int
        """
        n_basis = 0
        for atom in self.atoms:
            if atom.basis_set is None or atom.atomic_number not in atom.basis_set:
                continue
            regions = atom.basis_set[atom.atomic_number]
            for region_name in ("core", "valence_inner", "valence_outer"):
                region = regions.get(region_name, {})
                for l_val, shells in region.items():
                    # Each shell contributes (2l+1) basis functions
                    n_basis += len(shells) * (2 * l_val + 1)
        return n_basis

    def populate_spinorbitals_with_gto(self) -> None:
        """Attach GTO basis data to occupied spin-orbitals for all atoms.

        This method iterates over all :class:`Atom` objects stored in the
        :attr:`atoms` list and, for each atom that has a non-``None``
        :attr:`~Atom.basis_set`, attaches Gaussian-type orbital (GTO)
        data to occupied spin-orbitals.

        Atoms without an associated :class:`~q_block.io.basis_set.BasisSet`
        or whose element is absent from the basis set are left unchanged.
        """

        for atom in self.atoms:
            if atom.basis_set is None or atom.atomic_number not in atom.basis_set:
                continue
            regions = atom.basis_set[atom.atomic_number]

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
                        regions.get(region, {}).get(l, [])
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

    def to_dataframe(self) -> pd.DataFrame:
        """Build a multi-index :class:`pandas.DataFrame` for occupied spin-orbitals.

        This method assumes that :meth:`populate_spinorbitals_with_gto` has
        already been called (which happens automatically in ``__init__``).
        It walks over all atoms in the molecule and collects information
        about *occupied* spin-orbitals only, together with the GTO basis
        parameters attached in ``SpinOrbital.data``.

        The returned DataFrame uses a hierarchical index to organise the
        information per atom and per spin-orbital::

            index = ["atom_id", "symbol", "n", "l", "m", "s"]

        and a *column* MultiIndex with the following logical groups:

        * ``("geometry", "cartesian", "x" | "y" | "z")`` – Cartesian
          coordinates of the atom
        * ``("quantum_numbers", "n", "")``, ``("quantum_numbers", "l", "")``,
          ``("quantum_numbers", "m", "")``, ``("quantum_numbers", "s", "")`` –
          redundant storage of quantum numbers as columns for convenience
        * ``("basis", "contractions", "c0".."cM")`` – contraction
          coefficients
        * ``("basis", "exponents", "e0".."eM")`` – primitive exponents

        where ``M`` is the maximum primitive length across all occupied
        spin-orbitals of the molecule. Shorter basis sets are padded with
        ``None`` values so that every row has the same number of
        ``c``/``e`` columns.

        :returns: Multi-index DataFrame describing occupied spin-orbitals
                  for the whole molecule.
        :rtype: pandas.DataFrame
        """

        rows: list[dict[str, Any]] = []
        max_len: int = 0

        # First pass: collect raw lists and track the maximum basis length
        for atom, atom_row in zip(self.atoms, self.input_data.atoms.itertuples()):
            atom_id = getattr(atom_row, "atom_id")
            symbol = getattr(atom_row, "symbol")
            x = float(getattr(atom_row, "x"))
            y = float(getattr(atom_row, "y"))
            z = float(getattr(atom_row, "z"))

            for subshell in atom._all_subshells:
                for orbital in subshell.orbitals:
                    for so, s in ((orbital.spin_up, +0.5), (orbital.spin_down, -0.5)):
                        if not so.occupied or not so.data:
                            continue

                        exponents = list(so.data.get("exponents", []))
                        contractions = list(so.data.get("contractions", []))
                        length = max(len(exponents), len(contractions))
                        if length > max_len:
                            max_len = length

                        rows.append(
                            {
                                "atom_id": atom_id,
                                "symbol": symbol,
                                "n": subshell.n,
                                "l": subshell.l,
                                "m": orbital.m,
                                "s": s,
                                "x": x,
                                "y": y,
                                "z": z,
                                "exponents": exponents,
                                "contractions": contractions,
                            }
                        )

        # No occupied spin-orbitals -> return an empty, but well-formed, DataFrame
        if not rows or max_len == 0:
            index = pd.MultiIndex.from_tuples(
                [], names=["atom_id", "symbol", "n", "l", "m", "s"]
            )
            geom_cols = [
                ("geometry", "cartesian", "x"),
                ("geometry", "cartesian", "y"),
                ("geometry", "cartesian", "z"),
            ]
            qn_cols = [
                ("quantum_numbers", "n", ""),
                ("quantum_numbers", "l", ""),
                ("quantum_numbers", "m", ""),
                ("quantum_numbers", "s", ""),
            ]
            basis_cols: list[tuple[str, str, str]] = []
            columns = pd.MultiIndex.from_tuples(geom_cols + qn_cols + basis_cols)
            return pd.DataFrame(index=index, columns=columns)

        # Second pass: flatten lists into c0..cM and e0..eM columns
        flattened_rows: list[dict[str | tuple[str, str, str], Any]] = []
        for row in rows:
            exponents = list(row["exponents"])
            contractions = list(row["contractions"])
            length = max(len(exponents), len(contractions))

            # First, equalise lengths within this orbital
            if len(exponents) < length:
                exponents += [None] * (length - len(exponents))
            if len(contractions) < length:
                contractions += [None] * (length - len(contractions))

            # Then pad both to the global maximum
            if length < max_len:
                exponents += [None] * (max_len - length)
                contractions += [None] * (max_len - length)

            flat: dict[str | tuple[str, str, str], Any] = {
                "atom_id": row["atom_id"],
                "symbol": row["symbol"],
                "n": row["n"],
                "l": row["l"],
                "m": row["m"],
                "s": row["s"],
                ("geometry", "cartesian", "x"): row["x"],
                ("geometry", "cartesian", "y"): row["y"],
                ("geometry", "cartesian", "z"): row["z"],
                ("quantum_numbers", "n", ""): row["n"],
                ("quantum_numbers", "l", ""): row["l"],
                ("quantum_numbers", "m", ""): row["m"],
                ("quantum_numbers", "s", ""): row["s"],
            }

            for i in range(max_len):
                flat[("basis", "contractions", f"c{i}")] = contractions[i]
                flat[("basis", "exponents", f"e{i}")] = exponents[i]

            flattened_rows.append(flat)

        df = pd.DataFrame(flattened_rows)
        df.set_index(["atom_id", "symbol", "n", "l", "m", "s"], inplace=True)
        df.columns = pd.MultiIndex.from_tuples(list(df.columns))

        return df
