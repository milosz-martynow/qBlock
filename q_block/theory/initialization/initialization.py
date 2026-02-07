"""Initialization classes for quantum-chemistry calculations.

This module implements a two-level hierarchy:

1. :class:`Initialization` – generic parent class that every quantum-chemistry
   calculation type needs.  It handles:

   * Loading molecular geometry (from an XYZ file *or* a Python structure)
   * Reading charge and spin multiplicity
   * Building a :class:`~q_block.systems.molecule.Molecule`
   * Computing the total electron count (accounting for charge)
   * Converting atomic coordinates from Ångström to Bohr (via
     :meth:`Molecule.to_bohr`)

2. :class:`HartreeFock` – inherits all of the :class:`Initialization` and adds
   Hartree-Fock-specific logic

   * Reading / validating the HF method (``RHF``, ``UHF``, ``ROHF``)
   * Validating that every atom in the molecule carries a
     :class:`~q_block.io.basis_set.BasisSet`; the actual basis-set
     data must already be attached to each
     :class:`~q_block.models.atom.Atom` before the molecule is passed in
   * Computing method-specific electron counts:
     - **RHF** : :math:`N_{occ} = N_{elec} / 2`  (must be even)
     - **UHF** : :math:`N_{\\alpha}`, :math:`N_{\\beta}`
     - **ROHF**: :math:`N_{closed}`, :math:`N_{open}`, multiplicity check
"""

from typing import List, Literal, Optional

from q_block.models.atom import Atom
from q_block.systems.molecule import Molecule


# ──────────────────────────────────────────────────────────────────────
# Allowed HF method literals
# ──────────────────────────────────────────────────────────────────────
HFMethod = Literal["RHF", "UHF", "ROHF"]

_VALID_HF_METHODS = {"RHF", "UHF", "ROHF"}


# ======================================================================
# G E N E R I C   I N I T I A L I Z A T I O N   (parent)
# ======================================================================
class Initialization:
    """Generic initialization common to every quantum-chemistry calculation.

    This class encapsulates the first, method-agnostic, steps that are
    always required:

    1. Load molecular geometry into a :class:`Molecule`.
    2. Accept charge and multiplicity.
    3. Compute the total number of electrons.
    4. Convert atomic coordinates from Ångström to Bohr.

    :param molecule: A :class:`Molecule` instance with geometry already
        loaded (atoms, coordinates, basis set assignments, etc.).
    :type molecule: Molecule
    :param charge: Net electric charge of the system.  Default ``0``
        (neutral molecule).
    :type charge: int
    :param multiplicity: Spin multiplicity :math:`2S+1`.  Default ``1``
        (singlet).
    :type multiplicity: int

    Attributes
    ----------
    molecule : Molecule
        The molecular system.
    charge : int
        System charge.
    multiplicity : int
        Spin multiplicity :math:`2S+1`.
    n_electrons : int
        Total electron count, :math:`\\sum Z_i - \\text{charge}`.
    """

    def __init__(
        self,
        molecule: Molecule,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> None:
        # ── 1. Store molecule, charge, multiplicity ──────────────────
        self.molecule: Molecule = molecule
        self.charge: int = charge
        self.multiplicity: int = multiplicity

        # ── 2. Compute total electron count ──────────────────────────
        self.n_electrons: int = self.molecule.n_electrons - self.charge
        self._validate_n_electrons()

        # ── 3. Validate multiplicity vs. electron count ──────────────
        self._validate_multiplicity()

        # ── 4. Convert coordinates to Bohr ───────────────────────────
        self.molecule.to_bohr()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_n_electrons(self) -> None:
        """Verify that the electron count is non-negative.

        :raises ValueError: If :attr:`n_electrons` is negative, meaning
            the charge exceeds the total neutral electron count.
        """
        if self.n_electrons < 0:
            raise ValueError(
                f"Negative electron count ({self.n_electrons}): charge "
                f"({self.charge}) exceeds total neutral electron count "
                f"({self.molecule.n_electrons})."
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

    def __repr__(self) -> str:
        return (
            f"Initialization(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons})"
        )


# ======================================================================
# H A R T R E E – F O C K   I N I T I A L I Z A T I O N
# ======================================================================
class HartreeFock(Initialization):
    """Hartree-Fock-specific initialisation.

    Inherits all generic setup from :class:`Initialization` and
    additionally:

    * reads the HF method (``RHF`` / ``UHF`` / ``ROHF``);
    * validates that every :class:`~q_block.models.atom.Atom` in the
      molecule has a :class:`~q_block.io.basis_set.BasisSet` attached;
    * computes method-dependent electron numbers:

      - **RHF**:  :math:`N_{occ} = N_{elec}/2` (must be even)
      - **UHF**:  :math:`N_{\\alpha}`, :math:`N_{\\beta}`
      - **ROHF**: :math:`N_{closed}`, :math:`N_{open}`

    The :class:`~q_block.io.basis_set.BasisSet` data is carried by the
    :class:`~q_block.systems.molecule.Molecule` — each atom stores its
    own basis set.  ``HartreeFock`` does **not** accept or manage basis
    set objects itself.

    :param molecule: Molecular system with basis-set data already
        attached to each atom.
    :type molecule: Molecule
    :param charge: System charge (default ``0``).
    :type charge: int
    :param multiplicity: Spin multiplicity (default ``1``).
    :type multiplicity: int
    :param hf_method: Hartree-Fock variant – one of ``"RHF"``,
        ``"UHF"``, ``"ROHF"``.
    :type hf_method: str

    Attributes
    ----------
    hf_method : HFMethod
        Validated HF method string.
    n_occ : Optional[int]
        Number of doubly-occupied orbitals (**RHF only**).
    n_alpha : Optional[int]
        Number of alpha electrons (**UHF / ROHF**).
    n_beta : Optional[int]
        Number of beta electrons (**UHF / ROHF**).
    n_closed : Optional[int]
        Number of doubly-occupied (closed-shell) orbitals (**ROHF only**).
    n_open : Optional[int]
        Number of singly-occupied (open-shell) orbitals (**ROHF only**).
    n_basis : int
        Total number of basis functions (contracted Gaussians) in the
        molecule.
    """

    def __init__(
        self,
        molecule: "Molecule",
        charge: int = 0,
        multiplicity: int = 1,
        hf_method: str = "RHF",
    ) -> None:
        # ── 1. Generic initialisation (geometry, charge, mult, Bohr) ─
        super().__init__(
            molecule=molecule,
            charge=charge,
            multiplicity=multiplicity,
        )

        # ── 2. HF method ─────────────────────────────────────────────
        self.hf_method: HFMethod = self._validate_hf_method(hf_method)

        # ── 3. Validate that every atom carries a basis set ──────────
        self._validate_basis_sets()

        # ── 4. Compute basis-set size ────────────────────────────────
        self.n_basis: int = self._count_basis_functions()

        # ── 5. Method-specific electron counts ───────────────────────
        self.n_occ: Optional[int] = None
        self.n_alpha: Optional[int] = None
        self.n_beta: Optional[int] = None
        self.n_closed: Optional[int] = None
        self.n_open: Optional[int] = None

        self._compute_hf_electron_counts()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_hf_method(method: str) -> HFMethod:
        """Validate and normalise the HF method string.

        :param method: User-supplied method string (case-insensitive).
        :type method: str
        :returns: Uppercased, validated method literal.
        :rtype: HFMethod
        :raises ValueError: If the method is not one of RHF, UHF, ROHF.
        """
        method_upper = method.strip().upper()
        if method_upper not in _VALID_HF_METHODS:
            raise ValueError(
                f"Unknown HF method {method!r}. "
                f"Choose from: {', '.join(sorted(_VALID_HF_METHODS))}."
            )
        return method_upper  # type: ignore[return-value]

    def _validate_basis_sets(self) -> None:
        """Verify that every atom has a valid basis set attached.

        :raises ValueError: If any atom is missing basis-set data or
            its element is absent from the attached basis set.
        """
        for atom in self.molecule.atoms:
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

    # ------------------------------------------------------------------
    # Basis-set helpers
    # ------------------------------------------------------------------
    def _count_basis_functions(self) -> int:
        """Count the total number of contracted basis functions.

        A "basis function" here is one contracted Gaussian shell for a
        given angular momentum :math:`l`, contributing :math:`2l+1`
        Cartesian/spherical functions.

        :returns: Total number of basis functions across all atoms.
        :rtype: int
        """
        n_basis = 0
        for row in self.molecule.input_data.atoms.itertuples():
            atom: Atom = row.atom
            if atom.basis_set is None or atom.atomic_number not in atom.basis_set:
                continue
            regions = atom.basis_set[atom.atomic_number]
            for region_name in ("core", "valence_inner", "valence_outer"):
                region = regions.get(region_name, {})
                for l_val, shells in region.items():
                    # Each shell contributes (2l+1) basis functions
                    n_basis += len(shells) * (2 * l_val + 1)
        return n_basis

    # ------------------------------------------------------------------
    # HF-specific electron counts
    # ------------------------------------------------------------------
    def _compute_hf_electron_counts(self) -> None:
        """Dispatch electron-count computation based on the HF method.

        Populates the method-specific attributes (``n_occ``, ``n_alpha``,
        ``n_beta``, ``n_closed``, ``n_open``) as described in
        ``initialization.puml``.

        :raises ValueError: On inconsistencies (e.g. odd electron count
            for RHF).
        """
        if self.hf_method == "RHF":
            self._compute_rhf()
        elif self.hf_method == "UHF":
            self._compute_uhf()
        elif self.hf_method == "ROHF":
            self._compute_rohf()

    def _compute_rhf(self) -> None:
        """RHF: Restricted Closed-Shell.

        Requires an even number of electrons.  Sets :attr:`n_occ`.
        """
        if self.n_electrons % 2 != 0:
            raise ValueError(
                f"RHF requires an even number of electrons; "
                f"got {self.n_electrons} (charge={self.charge}, "
                f"multiplicity={self.multiplicity})."
            )
        if self.multiplicity != 1:
            raise ValueError(
                f"RHF requires singlet multiplicity (1); got "
                f"{self.multiplicity}."
            )
        self.n_occ = self.n_electrons // 2
        self.n_alpha = self.n_occ
        self.n_beta = self.n_occ

    def _compute_uhf(self) -> None:
        """UHF: Unrestricted, different spatial orbitals for α and β.

        Sets :attr:`n_alpha` and :attr:`n_beta`.
        """
        n_unpaired = self.multiplicity - 1
        self.n_beta = (self.n_electrons - n_unpaired) // 2
        self.n_alpha = self.n_beta + n_unpaired

    def _compute_rohf(self) -> None:
        """ROHF: Restricted Open-Shell.

        Sets :attr:`n_closed`, :attr:`n_open`, :attr:`n_alpha`,
        :attr:`n_beta`, and verifies the multiplicity relation
        :math:`M = N_{open} + 1`.
        """
        n_unpaired = self.multiplicity - 1  # N_open
        n_paired_electrons = self.n_electrons - n_unpaired
        if n_paired_electrons % 2 != 0:
            raise ValueError(
                f"ROHF: after removing {n_unpaired} unpaired electrons, "
                f"the remaining {n_paired_electrons} electrons are not even."
            )
        self.n_closed = n_paired_electrons // 2
        self.n_open = n_unpaired

        self.n_beta = self.n_closed
        self.n_alpha = self.n_closed + self.n_open

        # Verify multiplicity = N_open + 1
        expected_multiplicity = self.n_open + 1
        if self.multiplicity != expected_multiplicity:
            raise ValueError(
                f"ROHF multiplicity inconsistency: multiplicity="
                f"{self.multiplicity} but N_open={self.n_open} implies "
                f"M={expected_multiplicity}."
            )

    def __repr__(self) -> str:
        method_info = f"hf_method={self.hf_method}"
        if self.hf_method == "RHF":
            method_info += f", n_occ={self.n_occ}"
        elif self.hf_method == "UHF":
            method_info += f", n_alpha={self.n_alpha}, n_beta={self.n_beta}"
        elif self.hf_method == "ROHF":
            method_info += (
                f", n_closed={self.n_closed}, n_open={self.n_open}"
            )
        return (
            f"HartreeFock(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, {method_info}, "
            f"n_basis={self.n_basis})"
        )
