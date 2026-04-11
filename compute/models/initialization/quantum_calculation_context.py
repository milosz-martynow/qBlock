"""Quantum calculation context classes for quantum-chemistry calculations.

This module implements a three-level hierarchy:

1. :class:`QuantumCalculationContext` – generic parent class that every quantum-chemistry
   calculation type needs.  It handles:

   * Loading molecular geometry (from an XYZ file *or* a Python structure)
   * Reading charge and spin multiplicity
   * Building a :class:`~compute.models.molecule.Molecule`
   * Computing the total electron count (accounting for charge)
   * Converting atomic coordinates from Ångström to Bohr (via
     :meth:`Molecule.to_bohr`)

2. :class:`HartreeFock` – inherits from :class:`QuantumCalculationContext` and adds
   common Hartree-Fock logic shared by all HF variants:

   * Validating that every atom in the molecule carries a
     :class:`~compute.io.basis_set.BasisSet`
   * Computing the total number of basis functions
   * Declaring common electron-count attributes (:attr:`n_alpha`,
     :attr:`n_beta`)

3. Method-specific subclasses of :class:`HartreeFock`:

   * :class:`RHF` – Restricted Closed-Shell:
     :math:`N_{occ} = N_{elec} / 2`  (singlet only, even electrons)
   * :class:`UHF` – Unrestricted:
     :math:`N_{\\alpha}`, :math:`N_{\\beta}` for any multiplicity
   * :class:`ROHF` – Restricted Open-Shell:
     :math:`N_{closed}`, :math:`N_{open}`, multiplicity check
"""

from typing import List, Literal, Optional

from compute.models.basis_functions import ContractedGaussianTypeOrbital
from compute.models.molecule import Molecule

# ──────────────────────────────────────────────────────────────────────
# Allowed HF method literals
# ──────────────────────────────────────────────────────────────────────
HFMethod = Literal["RHF", "UHF", "ROHF"]


# ======================================================================
# Q U A N T U M   C A L C U L A T I O N   C O N T E X T   (parent)
# ======================================================================
class QuantumCalculationContext:
    """Generic context common to every quantum-chemistry calculation.

    This class encapsulates the first, method-agnostic, steps that are
    always required:

    1. Load molecular geometry into a :class:`Molecule`.
    2. Read charge (sum of per-atom charges) and multiplicity from the
       molecule.
    3. Compute the total number of electrons.
    4. Convert atomic coordinates from Ångström to Bohr.

    Validation of the electron count and multiplicity consistency is
    performed automatically inside :class:`Molecule` at construction
    time.

    The ``charge`` is derived automatically from per-atom
    :attr:`~compute.models.atom.Atom.charge` values, and
    ``multiplicity`` is set on the
    :class:`~compute.models.molecule.Molecule` itself.

    :param molecule: A :class:`Molecule` instance with geometry already
        loaded (atoms, coordinates, basis set assignments, per-atom
        charges, multiplicity, etc.).
    :type molecule: Molecule

    Attributes
    ----------
    molecule : Molecule
        The molecular system.
    charge : int
        System charge (sum of per-atom charges, read from the molecule).
    multiplicity : int
        Spin multiplicity :math:`2S+1` (read from the molecule).
    n_electrons : int
        Total electron count, :math:`\\sum Z_i + \\text{charge}`.
    """

    def __init__(
        self,
        molecule: Molecule,
    ) -> None:
        # ── 1. Store molecule ────────────────────────────────────────
        self.molecule: Molecule = molecule

        # ── 2. Read charge and multiplicity from the molecule ────────
        self.charge: int = self.molecule.charge
        self.multiplicity: int = self.molecule.multiplicity

        # ── 3. Compute total electron count ──────────────────────────
        self.n_electrons: int = self.molecule.n_electrons

        # ── 4. Convert coordinates to Bohr ───────────────────────────
        self.molecule.to_bohr()

    def __repr__(self) -> str:
        return (
            f"QuantumCalculationContext(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons})"
        )


# ======================================================================
# H A R T R E E – F O C K   B A S E   C L A S S
# ======================================================================
class HartreeFock(QuantumCalculationContext):
    """Hartree-Fock base class with logic common to all HF variants.

    Inherits all generic setup from :class:`QuantumCalculationContext` and
    additionally:

    * computes the total number of basis functions;
    * declares the common electron-count attributes :attr:`n_alpha` and
      :attr:`n_beta` (populated by the concrete subclass).

    Basis-set validation is performed automatically inside
    :class:`~compute.models.molecule.Molecule` at construction time.

    Concrete subclasses (:class:`RHF`, :class:`UHF`, :class:`ROHF`)
    implement method-specific electron-count logic and validation.

    The :class:`~compute.io.basis_set.BasisSet` data is carried by the
    :class:`~compute.models.molecule.Molecule` — each atom stores its
    own basis set.  ``HartreeFock`` does **not** accept or manage basis
    set objects itself.

    :param molecule: Molecular system with basis-set data and per-atom
        charges already attached to each atom, and ``multiplicity``
        set on the molecule.
    :type molecule: Molecule
    :param hf_method: Hartree-Fock variant label (``"RHF"``, ``"UHF"``,
        or ``"ROHF"``).  Typically set by the concrete subclass.
    :type hf_method: HFMethod

    Attributes
    ----------
    hf_method : HFMethod
        Validated HF method string.
    n_alpha : Optional[int]
        Number of alpha electrons (set by subclass).
    n_beta : Optional[int]
        Number of beta electrons (set by subclass).
    n_basis : int
        Total number of basis functions (contracted Gaussians) in the
        molecule.
    """

    def __init__(
        self,
        molecule: Molecule,
        hf_method: HFMethod,
    ) -> None:
        # ── 1. Generic initialisation (geometry, charge, mult, Bohr) ─
        super().__init__(molecule=molecule)

        # ── 2. HF method label ───────────────────────────────────────
        self.hf_method: HFMethod = hf_method

        # ── 3. Compute basis-set size ────────────────────────────
        self.n_basis: int = self.molecule.n_basis

        # ── 4. Lazy CGTO list (built on first access) ───────────────
        self._cgto: Optional[List[ContractedGaussianTypeOrbital]] = None

        # ── 5. Common electron-count attributes (set by subclass) ────
        self.n_alpha: Optional[int] = None
        self.n_beta: Optional[int] = None

    @property
    def cgto(self) -> List[ContractedGaussianTypeOrbital]:
        """Contracted Gaussian-Type Orbital basis for integral computation.

        Built lazily on first access by calling
        :meth:`Molecule.make_contracted_gaussian_type_orbital`.
        Subsequent accesses return the cached list.

        :returns: Flat list of
            :class:`~compute.models.basis_functions.ContractedGaussianTypeOrbital`
            shells.
        :rtype: list
        """
        if self._cgto is None:
            self.molecule.make_contracted_gaussian_type_orbital()
            self._cgto = self.molecule.contracted_gaussian_type_orbitals or []
        return self._cgto

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, hf_method={self.hf_method}, "
            f"n_alpha={self.n_alpha}, n_beta={self.n_beta}, "
            f"n_basis={self.n_basis})"
        )


# ======================================================================
# R H F   –   R E S T R I C T E D   C L O S E D - S H E L L
# ======================================================================
class RHF(HartreeFock):
    """Restricted Closed-Shell Hartree-Fock.

    Requires an **even** number of electrons and **singlet**
    multiplicity (:math:`M = 1`).  All electrons are paired:

    .. math:: N_{occ} = N_{elec} / 2

    :param molecule: Molecular system (see :class:`HartreeFock`).
    :type molecule: Molecule

    Attributes
    ----------
    n_occ : int
        Number of doubly-occupied spatial orbitals.
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__(molecule=molecule, hf_method="RHF")

        # ── Validate constraints ─────────────────────────────────────
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

        # ── Electron counts ──────────────────────────────────────────
        self.n_occ: int = self.n_electrons // 2
        self.n_alpha = self.n_occ
        self.n_beta = self.n_occ

    def __repr__(self) -> str:
        return (
            f"RHF(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, n_occ={self.n_occ}, "
            f"n_basis={self.n_basis})"
        )


# ======================================================================
# U H F   –   U N R E S T R I C T E D
# ======================================================================
class UHF(HartreeFock):
    """Unrestricted Hartree-Fock.

    Uses separate spatial orbitals for alpha and beta electrons,
    supporting any spin multiplicity:

    .. math::

        N_{\\beta}  = (N_{elec} - n_{unpaired}) / 2

        N_{\\alpha} = N_{\\beta} + n_{unpaired}

    where :math:`n_{unpaired} = M - 1`.

    :param molecule: Molecular system (see :class:`HartreeFock`).
    :type molecule: Molecule
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__(molecule=molecule, hf_method="UHF")

        # ── Electron counts ──────────────────────────────────────────
        n_unpaired = self.multiplicity - 1
        self.n_beta = (self.n_electrons - n_unpaired) // 2
        self.n_alpha = self.n_beta + n_unpaired

    def __repr__(self) -> str:
        return (
            f"UHF(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, "
            f"n_alpha={self.n_alpha}, n_beta={self.n_beta}, "
            f"n_basis={self.n_basis})"
        )


# ======================================================================
# R O H F   –   R E S T R I C T E D   O P E N - S H E L L
# ======================================================================
class ROHF(HartreeFock):
    """Restricted Open-Shell Hartree-Fock.

    Doubly-occupied (closed) orbitals share the same spatial part;
    singly-occupied (open) orbitals carry unpaired electrons:

    .. math::

        N_{open}   = M - 1

        N_{closed} = (N_{elec} - N_{open}) / 2

    Verifies :math:`M = N_{open} + 1`.

    :param molecule: Molecular system (see :class:`HartreeFock`).
    :type molecule: Molecule

    Attributes
    ----------
    n_closed : int
        Number of doubly-occupied (closed-shell) spatial orbitals.
    n_open : int
        Number of singly-occupied (open-shell) spatial orbitals.
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__(molecule=molecule, hf_method="ROHF")

        # ── Validate and compute ─────────────────────────────────────
        n_unpaired = self.multiplicity - 1
        n_paired_electrons = self.n_electrons - n_unpaired
        if n_paired_electrons % 2 != 0:
            raise ValueError(
                f"ROHF: after removing {n_unpaired} unpaired electrons, "
                f"the remaining {n_paired_electrons} electrons are not even."
            )

        self.n_closed: int = n_paired_electrons // 2
        self.n_open: int = n_unpaired

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
        return (
            f"ROHF(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, "
            f"n_closed={self.n_closed}, n_open={self.n_open}, "
            f"n_basis={self.n_basis})"
        )
